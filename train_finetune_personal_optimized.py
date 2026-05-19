# load packages
import os

import random
import yaml
import time
from munch import Munch
import numpy as np


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
# Limit PyTorch's CPU thread usage to prevent the server from getting stuck
torch.set_num_threads(4)


from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import warnings
import os
import os.path as osp
import csv
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# .env loading + wandb directory fix
# -----------------------------------------------------------------------------
# Must happen BEFORE importing wandb so that the env vars are already set
# when wandb initialises its internal state on import.
#
# load_dotenv() reads key=value pairs from the .env file and pushes them into
# os.environ so that os.getenv() can find them anywhere in the script.
#
# The .env file lives in Configs/.env relative to this script's directory.
# We build an absolute path using __file__ so the script works regardless of
# the working directory you launch it from.
# =============================================================================
from dotenv import load_dotenv

_env_path = osp.join(osp.dirname(osp.abspath(__file__)), "Configs", ".env")
load_dotenv(dotenv_path=_env_path)

# Diagnostic: confirm the key loaded before we go any further
_wandb_api_key = os.getenv("WANDB_API_KEY")
print(f"[ENV] WANDB_API_KEY loaded: {'YES' if _wandb_api_key else 'NO — check path: ' + _env_path}")

# Fix for servers where HOME points to a non-existent or non-writable mount.
# wandb needs to write a .netrc file to HOME and a cache dir for run files.
# We redirect both to a path we know is writable on this machine.
_wandb_cache_dir = "/mnt/newworkspace/rizoan/StyleTTS2-Fork/wandb_cache"
os.makedirs(_wandb_cache_dir, exist_ok=True)          # create once if missing
os.environ["WANDB_CONFIG_DIR"] = _wandb_cache_dir     # wandb config + .netrc location
os.environ["WANDB_DIR"]        = _wandb_cache_dir     # wandb run files location
os.environ["HOME"]             = "/mnt/newworkspace/rizoan"  # fixes .netrc write path

# Set the API key directly in the environment.
# When WANDB_API_KEY is present in os.environ, wandb.login() reads it
# automatically and skips the .netrc write entirely — avoids the FileNotFoundError.
if _wandb_api_key:
    os.environ["WANDB_API_KEY"] = _wandb_api_key

import wandb  # import AFTER env vars are set

from meldataset import build_dataloader

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


# =============================================================================
# Helper: safe tensor → Python scalar
# -----------------------------------------------------------------------------
# Losses are sometimes plain Python 0 (int) before a training stage activates
# (e.g. loss_sty = 0 before diff_epoch) and sometimes torch.Tensor afterwards.
# wandb.log() requires plain Python numbers so we normalise here.
# Defined once at module level — avoids redefining it inside the hot loop.
# =============================================================================
def to_scalar(x):
    """Return a plain Python float from a torch.Tensor or numeric."""
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


# =============================================================================
# Helper: save waveform tensor to WAV file
# -----------------------------------------------------------------------------
# torchaudio.save() requires shape (C, T). Our decoder produces (B, 1, T) so
# callers index into the batch dimension before passing here.
# =============================================================================
def save_audio(path: str, waveform: torch.Tensor, sample_rate: int):
    """Save a 1-D or 2-D waveform tensor to disk as a WAV file."""
    wav = waveform.detach().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)   # (T,) → (1, T)
    torchaudio.save(path, wav, sample_rate)


# =============================================================================
# Per-epoch validation inference
# -----------------------------------------------------------------------------
# Called once at the end of every epoch (after the metric validation loop).
# Iterates the full validation dataloader, synthesises audio for every sample,
# saves generated + reference WAVs under Output/Epoch_XXXX/, and appends rows
# to a shared routing CSV so all epochs accumulate in one queryable file.
#
# A small number of clips (max_samples_to_log) are also uploaded to W&B as an
# Audio table so you can listen in the browser without downloading files.
# Set max_wandb_audio_samples: 0 in config to disable uploads entirely.
#
# Reference WAVs are only written on the first inference epoch (infer_start_epoch)
# to avoid duplicating identical files across all epochs and wasting disk space.
# The CSV still records the reference path for every epoch pointing back to
# the first-epoch copy so lookups always work.
# =============================================================================
def run_validation_inference(
    epoch: int,
    infer_start_epoch: int,         # the first epoch inference ran (for ref path lookup)
    model,
    val_dataloader,
    stft_loss,
    n_down: int,
    sr: int,
    device: str,
    log_dir: str,
    csv_path: str,
    wandb_run,
    max_samples_to_log: int = 4,
):
    """Synthesise validation audio, save to disk, write routing CSV rows."""

    # Output folder for this epoch's generated files
    epoch_out_dir = osp.join(log_dir, "Output", f"Epoch_{epoch + 1:04d}")
    os.makedirs(epoch_out_dir, exist_ok=True)

    # Reference files always live in the first inference epoch's folder
    ref_epoch_dir = osp.join(log_dir, "Output", f"Epoch_{infer_start_epoch + 1:04d}")

    # CSV: write header only on the very first call when the file doesn't exist
    write_header = not osp.exists(csv_path)
    csv_file   = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow([
            "epoch",           # 1-based epoch index
            "batch_idx",       # batch index within val_dataloader
            "sample_idx",      # sample index within the batch
            "mel_loss",        # STFT mel loss for this batch
            "generated_path",  # path to synthesised WAV (relative to log_dir)
            "reference_path",  # path to reference WAV (relative to log_dir)
        ])

    total_mel_loss  = 0.0
    num_batches     = 0
    wandb_audio_rows = []   # (gen_np, ref_np) pairs for W&B table upload

    is_first_infer_epoch = (epoch == infer_start_epoch)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            logger.info(f"[DEBUG] Inference batch {batch_idx}")
            try:
                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

                mask      = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)

                mask_ST       = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr  = t_en @ s2s_attn_mono
                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                ss, gs = [], []
                for bib in range(len(mel_input_length)):
                    mel = mels[bib, :, :mel_input_length[bib]]
                    ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                    gs.append(model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

                s_pred = torch.stack(ss).squeeze(1)
                gs     = torch.stack(gs).squeeze(1)

                bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                d_en     = model.bert_encoder(bert_dur).transpose(-1, -2)
                d, p     = model.predictor(d_en, s_pred, input_lengths, s2s_attn_mono, text_mask)

                mel_len = int(mel_input_length.min().item() / 2 - 1)
                en_list, p_en_list, gt_list, wav_list = [], [], [], []

                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)
                    if mel_length <= mel_len:
                        continue
                    rstart     = np.random.randint(0, mel_length - mel_len)

                    start = rstart * 2 * 300
                    end   = (rstart + mel_len) * 2 * 300

                    if end > len(waves[bib]):
                        continue
                    en_list.append(asr[bib, :, rstart:rstart + mel_len])
                    p_en_list.append(p[bib, :, rstart:rstart + mel_len])
                    gt_list.append(mels[bib, :, rstart * 2:(rstart + mel_len) * 2])

                    y = waves[bib][start:end]

                    #wav_list.append(torch.from_numpy(y).to(device))
                    #wav_list.append(torch.tensor(y, dtype=torch.float32, device=device))
                    wav_list.append(
                        torch.from_numpy(y).to(device).float()
                    )


                if len(wav_list) == 0:
                    continue

                if not (
                    len(en_list) ==
                    len(p_en_list) ==
                    len(gt_list) ==
                    len(wav_list)
                ):
                    logger.error(
                        f"[Inference mismatch] "
                        f"en={len(en_list)} "
                        f"p={len(p_en_list)} "
                        f"gt={len(gt_list)} "
                        f"wav={len(wav_list)}"
                    )
                    continue
                wav_gt = torch.stack(wav_list).float()
                en     = torch.stack(en_list)
                p_en   = torch.stack(p_en_list)
                gt     = torch.stack(gt_list).detach()

                s_clip     = model.style_encoder(gt.unsqueeze(1))
                s_dur_clip = model.predictor_encoder(gt.unsqueeze(1))

                F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur_clip)
                y_rec = model.decoder(en, F0_fake, N_fake, s_clip)

                batch_mel_loss  = stft_loss(y_rec.squeeze(1), wav_gt.detach()).item()
                total_mel_loss += batch_mel_loss
                num_batches    += 1

                # Save per-sample audio files and write CSV rows
                for sample_idx in range(y_rec.shape[0]):
                    gen_filename = f"generated_batch{batch_idx:03d}_sample{sample_idx:02d}.wav"
                    ref_filename = f"reference_batch{batch_idx:03d}_sample{sample_idx:02d}.wav"
                    gen_path     = osp.join(epoch_out_dir, gen_filename)
                    ref_path     = osp.join(ref_epoch_dir, ref_filename)  # always points to first epoch

                    # Generated audio: written every epoch
                    save_audio(gen_path, y_rec[sample_idx], sr)

                    # Reference audio: written only on the first inference epoch
                    # to avoid duplicating identical files for every subsequent epoch
                    if is_first_infer_epoch:
                        os.makedirs(ref_epoch_dir, exist_ok=True)
                        save_audio(ref_path, wav_gt[sample_idx], sr)

                    rel_gen = osp.relpath(gen_path, log_dir)
                    rel_ref = osp.relpath(ref_path, log_dir)
                    csv_writer.writerow([
                        epoch + 1,
                        batch_idx,
                        sample_idx,
                        f"{batch_mel_loss:.5f}",
                        rel_gen,
                        rel_ref,
                    ])

                    # Collect clips for W&B audio table (capped at max_samples_to_log)
                    if len(wandb_audio_rows) < max_samples_to_log:
                        wandb_audio_rows.append((
                            y_rec[sample_idx].squeeze().detach().cpu().numpy(),
                            wav_gt[sample_idx].squeeze().detach().cpu().numpy(),
                        ))

            except Exception as e:
                logger.warning(f"[Inference] Batch {batch_idx} failed: {e}")
                continue

    csv_file.close()
    avg_mel_loss = total_mel_loss / max(num_batches, 1)

    # Upload audio table to W&B (skipped if max_samples_to_log == 0)
    if wandb_run is not None and wandb_audio_rows:
        audio_table = wandb.Table(columns=["epoch", "sample", "generated", "reference"])
        for idx, (gen_np, ref_np) in enumerate(wandb_audio_rows):
            audio_table.add_data(
                epoch + 1,
                idx,
                wandb.Audio(gen_np, sample_rate=sr, caption=f"ep{epoch+1}_gen_{idx}"),
                wandb.Audio(ref_np, sample_rate=sr, caption=f"ep{epoch+1}_ref_{idx}"),
            )
        wandb_run.log(
            {f"val_inference/audio_epoch_{epoch+1}": audio_table},
            commit=False,
        )

    logger.info(
        f"[Inference] Epoch {epoch+1}: avg mel loss = {avg_mel_loss:.5f}, "
        f"saved to {epoch_out_dir}"
    )
    return avg_mel_loss


@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    # =========================================================================
    # W&B initialisation
    # -------------------------------------------------------------------------
    # wandb.login() reads WANDB_API_KEY from os.environ (set above at module
    # level) so no key argument is needed here — avoids the .netrc write path.
    # wandb.init() creates the run and logs the entire YAML config as
    # hyper-parameters visible under the Config tab in the W&B dashboard.
    # resume="allow" means if the run crashes and you restart with the same
    # run_id it will continue the existing run rather than creating a new one.
    # =========================================================================
    wandb.login()
    wandb_run = wandb.init(
        project=config.get("wandb_project", "styletts2-finetune"),
        name=config.get("wandb_run_name", f"run_{int(time.time())}"),
        config=config,      # logs all YAML keys as hyper-parameters
        resume="allow",
    )
    logger.info(f"W&B run: {wandb_run.name}  id: {wandb_run.id}")

    # =========================================================================
    # Everything below is IDENTICAL to the original code.
    # The only additions are:
    #   1. wandb.log() calls inside the log_interval block (training metrics)
    #   2. wandb.log() call after validation metrics
    #   3. run_validation_inference() call after validation loop (audio saving)
    #   4. wandb.run.summary updates inside the checkpoint block (best loss)
    #   5. wandb.finish() at the very end
    # Nothing else has been changed.
    # =========================================================================

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    # Routing CSV path — one file for the whole run, appended each epoch
    routing_csv_path = osp.join(log_dir, "validation_routing.csv")

    batch_size = config.get('batch_size', 10)

    epochs = config.get('epochs', 200)
    save_freq = config.get('save_freq', 1)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    # infer_start_epoch: skip audio inference for the first N epochs where the
    # model output is still noise — saves disk space. Default 0 = always infer.
    infer_start_epoch = config.get('infer_start_epoch', 0)

    # max_wandb_audio_samples: how many clips to upload to W&B per epoch.
    # Keep this small (≤ 8) to stay within the free 5 GB quota.
    # Set to 0 to disable W&B audio uploads entirely (files still saved locally).
    max_wandb_audio = config.get('max_wandb_audio_samples', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=0,   # 0 = no subprocess workers, avoids CUDA context deadlock on single GPU
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})
    
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    
    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion'])

            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.') 

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
    
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))
        
    n_down = model.text_aligner.n_down

    best_loss = float('inf')
    loss_train_record = list([])
    loss_test_record = list([])
    iters = 0
    
    criterion = nn.L1Loss()
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    start_ds = False
    
    running_std = []
    
    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, sampler, 
                                slmadv_params.min_len, 
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter, 
                                sig=slmadv_params.sig
                               )
    
    
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]
        
        model.text_aligner.train()
        model.text_encoder.train()
        
        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)
                
            try:
                ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)
            except Exception as e:
                logger.exception(e)
                continue

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            t_en = model.text_encoder(texts, input_lengths, text_mask)
            
            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)

            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze()
            gs = torch.stack(gs).squeeze()
            s_trg = torch.cat([gs, s_dur], dim=-1).detach()

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
            
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)
                
                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item()
                    running_std.append(model.diffusion.module.diffusion.sigma_data)
                    
                if multispeaker:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                                   features=ref,
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean()
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
                else:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)                    
                    loss_diff = model.diffusion.module.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean()
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
            else:
                loss_sty = 0
                loss_diff = 0

            s_loss = 0

            d, p = model.predictor(d_en, s_dur, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    text_mask)
                
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            en = []
            gt = []
            p_en = []
            wav = []
            st = []
            
            for bib in range(len(mel_input_length)):

                mel_length = int(mel_input_length[bib].item() / 2)

                if mel_length <= mel_len:
                    continue

                random_start = np.random.randint(0, mel_length - mel_len)

                start = (random_start * 2) * 300
                end   = ((random_start + mel_len) * 2) * 300

                if end > len(waves[bib]):
                    continue

                y = waves[bib][start:end]

                en.append(
                    asr[bib, :, random_start:random_start+mel_len]
                )

                p_en.append(
                    p[bib, :, random_start:random_start+mel_len]
                )

                gt.append(
                    mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)]
                )

                wav.append(
                    torch.from_numpy(y).to(device).float()
                )

                random_start_st = np.random.randint(
                    0,
                    mel_length - mel_len_st
                )

                st.append(
                    mels[bib, :, (random_start_st * 2):
                    ((random_start_st+mel_len_st) * 2)]
                )
            
            if len(wav) == 0:
                logger.warning("[EMPTY BATCH] wav list empty")
                continue

            if not (len(en) == len(p_en) == len(gt) == len(wav)):
                logger.error(
                    f"[BATCH MISMATCH] "
                    f"en={len(en)} "
                    f"p_en={len(p_en)} "
                    f"gt={len(gt)} "
                    f"wav={len(wav)}"
                )
                continue
            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            
            if gt.size(-1) < 80:
                continue
            
            s = model.style_encoder(gt.unsqueeze(1))           
            s_dur = model.predictor_encoder(gt.unsqueeze(1))
                
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                
                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

                wav = y_rec_gt

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, F0_fake, N_fake, s)
            # 🔴 ADD THIS BLOCK RIGHT HERE
            if torch.isnan(y_rec).any() or torch.isinf(y_rec).any():
                print(f"[NaN DETECTED] Skipping batch at epoch {epoch}, step {i}")
                continue

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            optimizer.zero_grad()
            d_loss = dl(wav.detach(), y_rec.detach()).mean()
            d_loss.backward()
            optimizer.step('msd')
            optimizer.step('mpd')

            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean()
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                       _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)
            
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            g_loss = loss_params.lambda_mel * loss_mel + \
                     loss_params.lambda_F0 * loss_F0_rec + \
                     loss_params.lambda_ce * loss_ce + \
                     loss_params.lambda_norm * loss_norm_rec + \
                     loss_params.lambda_dur * loss_dur + \
                     loss_params.lambda_gen * loss_gen_all + \
                     loss_params.lambda_slm * loss_lm + \
                     loss_params.lambda_sty * loss_sty + \
                     loss_params.lambda_diff * loss_diff + \
                    loss_params.lambda_mono * loss_mono + \
                    loss_params.lambda_s2s * loss_s2s
            
            running_loss += loss_mel.item()
            # ── Pre-backward shape + NaN diagnostic ─────────────────────────
            
            _diag = {
                "t_en":           t_en,
                "s2s_attn":       s2s_attn,
                "s2s_attn_mono":  s2s_attn_mono,
                "asr":            asr,
                "en":             en,
                "p_en":           p_en,
                "F0_fake":        F0_fake,
                "N_fake":         N_fake,
                "y_rec":          y_rec,
                "s":              s,
                "s_dur":          s_dur,
            }
            for _name, _t in _diag.items():
                if isinstance(_t, torch.Tensor):
                    _has_nan = torch.isnan(_t).any().item()
                    _has_inf = torch.isinf(_t).any().item()
                    logger.debug(
                        f"[PRE-BWD] {_name:20s} shape={str(_t.shape):30s} "
                        f"dtype={_t.dtype}  nan={_has_nan}  inf={_has_inf}  "
                        f"min={_t.float().min().item():.4f}  max={_t.float().max().item():.4f}"
                    )

            g_loss.backward()
            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()

            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('predictor_encoder')
            optimizer.step('style_encoder')
            optimizer.step('decoder')
            
            optimizer.step('text_encoder')
            optimizer.step('text_aligner')
            
            if epoch >= diff_epoch:
                optimizer.step('diffusion')

            d_loss_slm, loss_gen_lm = 0, 0
            if epoch >= joint_epoch:
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts
                    
                slm_out = slmadv(i, 
                                 y_rec_gt, 
                                 y_rec_gt_pred, 
                                 waves, 
                                 mel_input_length,
                                 ref_texts, 
                                 ref_lengths, use_ind, s_trg.detach(), ref if multispeaker else None)

                if slm_out is not None:
                    d_loss_slm, loss_gen_lm, y_pred = slm_out

                    optimizer.zero_grad()
                    loss_gen_lm.backward()

                    total_norm = {}
                    for key in model.keys():
                        total_norm[key] = 0
                        parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    if total_norm['predictor'] > slmadv_params.thresh:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= (1 / total_norm['predictor'])

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale
                    
                    optimizer.step('bert_encoder')
                    optimizer.step('bert')
                    optimizer.step('predictor')
                    optimizer.step('diffusion')

                    if d_loss_slm != 0:
                        optimizer.zero_grad()
                        d_loss_slm.backward(retain_graph=True)
                        optimizer.step('wd')

            iters = iters + 1
            
            if (i+1)%log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f, SLoss: %.5f, S2S Loss: %.5f, Mono Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff, d_loss_slm, loss_gen_lm, s_loss, loss_s2s, loss_mono))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/slm_loss', loss_lm, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                writer.add_scalar('train/d_loss_slm', d_loss_slm, iters)
                writer.add_scalar('train/gen_loss_slm', loss_gen_lm, iters)

                # ── W&B training metrics ──────────────────────────────────────
                # step=iters keeps x-axis aligned with global step count so
                # training and validation curves share the same x-axis in W&B.
                wandb.log({
                    "train/mel_loss":     running_loss / log_interval,
                    "train/gen_loss":     to_scalar(loss_gen_all),
                    "train/d_loss":       to_scalar(d_loss),
                    "train/ce_loss":      to_scalar(loss_ce),
                    "train/dur_loss":     to_scalar(loss_dur),
                    "train/slm_loss":     to_scalar(loss_lm),
                    "train/norm_loss":    to_scalar(loss_norm_rec),
                    "train/F0_loss":      to_scalar(loss_F0_rec),
                    "train/sty_loss":     to_scalar(loss_sty),
                    "train/diff_loss":    to_scalar(loss_diff),
                    "train/d_loss_slm":   to_scalar(d_loss_slm),
                    "train/gen_loss_slm": to_scalar(loss_gen_lm),
                    "train/s2s_loss":     to_scalar(loss_s2s),
                    "train/mono_loss":    to_scalar(loss_mono),
                }, step=iters, commit=True)
                
                running_loss = 0
                
                print('Time elasped:', time.time()-start_time)
            
        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]
        logger.info("[DEBUG] Validation loop starting...")
        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze()
                    gs = torch.stack(gs).squeeze()
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
                    d, p = model.predictor(d_en, s, 
                                                        input_lengths, 
                                                        s2s_attn_mono, 
                                                        text_mask)
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []

                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)
                        if mel_length <= mel_len:
                            continue

                        random_start = np.random.randint(0, mel_length - mel_len)

                        #y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        start = (random_start * 2) * 300
                        end   = ((random_start + mel_len) * 2) * 300

                        if end > len(waves[bib]):
                            continue
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        p_en.append(p[bib, :, random_start:random_start+mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                        y = waves[bib][start:end]
                        #wav.append(torch.from_numpy(y).to(device))
                        #wav.append(torch.tensor(y, dtype=torch.float32, device=device))
                        wav.append(
                            torch.from_numpy(y).to(device).float()
                        )
                    
                    if len(wav) == 0:
                        logger.warning("[EMPTY BATCH] wav list empty")
                        continue

                    if not (len(en) == len(p_en) == len(gt) == len(wav)):
                        logger.error(
                            f"[BATCH MISMATCH] "
                            f"en={len(en)} "
                            f"p_en={len(p_en)} "
                            f"gt={len(gt)} "
                            f"wav={len(wav)}"
                        )
                        continue
                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()
                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                               _text_input[1:_text_length-1])

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    if torch.isnan(y_rec).any() or torch.isinf(y_rec).any():
                        logger.warning(f"[Inference] NaN detected at batch {batch_idx} → skipping")
                        continue
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1)) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except Exception as e:
                    logger.exception(e)
                    continue

        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n')
        print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)

        # ── W&B validation metrics ────────────────────────────────────────────
        # commit=False — we flush together with the audio table below so all
        # end-of-epoch metrics appear as one atomic step in W&B.
        wandb.log({
            "eval/mel_loss": (loss_test  / iters_test).item(),
            "eval/dur_loss": (loss_align / iters_test).item(),
            "eval/F0_loss":  (loss_f     / iters_test).item(),
        }, step=iters, commit=False)

        # ── Per-epoch validation inference + audio saving ─────────────────────
        # Runs full synthesis on the validation set, saves WAV files under
        # Output/Epoch_XXXX/, appends rows to validation_routing.csv, and
        # uploads a small audio table to W&B.
        if epoch >= infer_start_epoch:
            logger.info("[DEBUG] Validation inference starting...")
            infer_mel_loss = run_validation_inference(
                epoch=epoch,
                infer_start_epoch=infer_start_epoch,
                model=model,
                val_dataloader=val_dataloader,
                stft_loss=stft_loss,
                n_down=n_down,
                sr=sr,
                device=device,
                log_dir=log_dir,
                csv_path=routing_csv_path,
                wandb_run=wandb_run,
                max_samples_to_log=max_wandb_audio,
            )
            wandb.log({"eval/infer_mel_loss": infer_mel_loss}, step=iters, commit=False)
            writer.add_scalar('eval/infer_mel_loss', infer_mel_loss, epoch + 1)

        # Flush all pending end-of-epoch W&B metrics in one call
        wandb.log({}, step=iters, commit=True)
        
        if (epoch + 1) % save_freq == 0:
            # Update best loss tracker
            current_val_loss = loss_test / iters_test
            is_best = current_val_loss < best_loss
            if is_best:
                best_loss = current_val_loss

            logger.info("[DEBUG] Checkpoint saving starting...")
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }

            # Periodic checkpoint — named by epoch number for easy rollback
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)

            #logger.info("[DEBUG] torch.save completed successfully")

            #logger.info(f'[Save] Periodic checkpoint → {save_path}')
            # 🔴 DEBUG: verify checkpoint was actually written
            if os.path.exists(save_path):
                logger.info(f"[Save OK] Checkpoint successfully written → {save_path}")
            else:
                logger.error(f"[Save FAIL] Checkpoint NOT found after saving → {save_path}")

            # Best model — overwrite a single file so you always know which
            # checkpoint had the lowest validation loss without hunting epoch numbers
            if is_best:
                best_path = osp.join(log_dir, 'best_model.pth')
                torch.save(state, best_path)
                # 🔴 DEBUG: verify checkpoint was actually written
                if os.path.exists(best_path):          # ← FIXED: now correctly checks best_path
                    logger.info(f"[Save OK] Best model written → {best_path} (val_loss={to_scalar(best_loss):.5f})")
                else:
                    logger.error(f"[Save FAIL] Best model NOT found after saving → {best_path}")
                #logger.info(f'[Save] New best val loss {best_loss:.5f} → {best_path}')
                # Summary metrics persist across steps in W&B run comparison table
                wandb.run.summary["best_val_mel_loss"] = to_scalar(best_loss)
                wandb.run.summary["best_epoch"]        = epoch + 1

            # if estimate sigma, save the estimated sigma
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    writer.close()
    wandb.finish()   # flush remaining W&B data and mark run as finished
    logger.info('Training complete.')

                            
if __name__=="__main__":
    main()