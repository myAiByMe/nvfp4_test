#!/usr/bin/env python3
"""
HessGPT Pre-Training v10 — NVfp4 Blackwell + Healing

NOUVEAUTÉS v10 :
  ✅ NVfp4 pretraining natif (RTX Pro 6000 Blackwell SM_120)
      - use_nvfp4=True dans CONFIG active te.Linear dans tous les GEMMs
        (q/k/v/out_proj, gate/up/down_proj) via NVIDIA Transformer Engine
      - Éléments protégés en BF16 : embeddings, RMSNorm, attention scores,
        output_head, RoPE
      - Enveloppe te.fp8_autocast autour du forward → NVfp4 Tensor Core activé
      - Fallback automatique BF16 si transformer_engine non installé

  ✅ Healing (précision recovery en fin de run)
      - À partir de healing_ratio * TOTAL_STEPS (défaut 90%) :
        désactivation automatique du contexte FP4 → forward repassé en BF16
      - Réduit l'erreur relative de ~1.5% à ~0.5% (stratégie NVIDIA paper arXiv 2509.25149)
      - Log clair lors du basculement

  ✅ Fix Sequence Packing — chaînage cu_seqlens complet
      - HessGPT.forward → TransformerBlock → MultiHeadAttention
        cu_seqlens_q/k et max_seqlen_q/k correctement propagés à chaque layer

  Tout le reste identique v9 (Muon+MARS, WSD, FA4/FA2, Benchmark, Checkpoint, etc.)
"""

import os
import contextlib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="transformer_engine")

os.environ["TORCHINDUCTOR_CACHE_DIR"]      = "./CompileCache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.makedirs("./CompileCache", exist_ok=True)

import torch
# ── TF32 : +20% vitesse sur Ampere/Hopper/Blackwell, précision quasi-identique BF16
torch.set_float32_matmul_precision('high')

from torch.utils.data import DataLoader, Dataset
import sys
import time
import math
import json
import gc
import threading
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime
import traceback
import numpy as np

# ── NVIDIA Transformer Engine — NVfp4 (Blackwell SM_120) ────────
_TE_AVAILABLE   = False
_nvfp4_recipe   = None

try:
    import transformer_engine
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as _te_recipe

    # Recette NVFP4BlockScaling — nom exact dans TE 2.15.0 (Blackwell SM_120)
    try:
        _nvfp4_recipe = _te_recipe.NVFP4BlockScaling()
        print("  ⚡ Transformer Engine : recette NVFP4BlockScaling disponible")
    except Exception as _e:
        # Fallback FP8 E4M3 si NVFP4BlockScaling indisponible
        _nvfp4_recipe = _te_recipe.DelayedScaling(
            margin=0,
            fp8_format=_te_recipe.Format.E4M3,
        )
        print(f"  ⚡ Transformer Engine : NVFP4BlockScaling échoué ({_e}) → fallback FP8 E4M3")

    _TE_AVAILABLE = True
    # FIX : te = transformer_engine.pytorch n'a pas __version__ → lire depuis le module racine
    _te_version = getattr(transformer_engine, "__version__", "?")
    print(f"  ✅ Transformer Engine chargé ({_te_version})")

except ImportError:
    print("  ⚠️  transformer_engine non installé — NVfp4 désactivé, entraînement en BF16")
    print("       Installer : pip install transformer-engine[pytorch]")

sys.path.append('./Core/Model')
sys.path.append('./Core/Attention')
sys.path.append('./Core/FeedForward')
sys.path.append('./Core/TransformerBlock')

print("=" * 80)
print("HessGPT v10 — NVfp4 Blackwell + Healing | FA2 + Sequence Packing")
print("=" * 80)

import argparse as _argparse
_parser = _argparse.ArgumentParser(description='HessGPT Pretrain', add_help=False)
_parser.add_argument('--batch-size', type=int, default=None,
                     help='Override CONFIG batch_size (ex: --batch-size 64)')
_parser.add_argument('--hf-token', type=str, default=None,
                     help='Hugging Face token (lecture + écriture)')
_parser.add_argument('--hf-repo', type=str, default='silyan/nvfp4_test_9Btokens',
                     help='Repo HF dataset (données + checkpoint partagé)')
_args, _ = _parser.parse_known_args()

_HF_TOKEN      = _args.hf_token
_HF_REPO       = _args.hf_repo
_HF_PUSH_INTERVAL = 60 * 30   # push checkpoint toutes les 1h

CONFIG = {
    'vocab_size':            None,
    'embed_dim':             1280,
    'num_heads':             20,
    'num_layers':            24,
    'max_seq_len':           1024,
    'dropout':               0.0,
    'use_rope':              True,
    'use_yarn':              False,
    'yarn_scale':            4.0,
    'yarn_original_max_len': 512,
    'use_swiglu':            True,
    'n_kv_heads':            5,
    'use_qk_norm':           True,
    'soft_cap':              None,
    'use_flash_attn':        True,
    'batch_size':            97,
    'gradient_accumulation': 8,
    'max_grad_norm':         1.0,
    'learning_rate':         4e-4,
    'weight_decay':          0.1,
    'adam_beta1':            0.9,
    'adam_beta2':            0.95,
    'adam_eps':              1e-8,
    'num_epochs':            1,
    'data_file':             './pretrain_data.bin',
    'val_tokens':            15_000_000,
    'warmup_ratio':          0.03,
    'decay_ratio':           0.15,
    'min_lr_ratio':          0.1,
    'validate_every_steps':  500,
    'val_batches':           50,
    'shuffle_seed':          42,
    'save_every_steps':      2000,
    'checkpoint_file':       './Model/HessGpt_pretrain.pt',
    'use_compile':           True,
    'compile_mode':          'default',
    'num_workers':           1,
    # ── Sequence Packing ────────────────────────────────────────
    'use_packing':           True,   # False = comportement v8 (padding classique)
    # ── Benchmark ───────────────────────────────────────────────
    'benchmark_steps':       20,     # steps de warmup mesurés au démarrage
    # ── NVfp4 (Blackwell SM_120 — RTX Pro 6000) ─────────────────
    # True  → te.Linear dans les GEMMs + te.fp8_autocast en training
    # False → BF16 classique (si GPU non-Blackwell ou TE non installé)
    'use_nvfp4':             True,
    # ── Healing ─────────────────────────────────────────────────
    # À partir de healing_ratio * TOTAL_STEPS, le contexte FP4 est
    # désactivé → forward repassé en BF16 pour recovery de précision.
    # Stratégie issue de l'article NVIDIA arXiv 2509.25149.
    'healing_ratio':         0.90,
    # ── Gradient Checkpointing (healing) ────────────────────────
    # True → active le gradient checkpointing automatiquement au
    # déclenchement du healing (derniers 10%) pour compenser la
    # hausse de VRAM liée au passage en BF16.
    # Réduit la VRAM des activations de ~30-40% au prix d'un
    # recalcul du forward pendant la backward.
    'gc_on_healing':         True,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Override batch_size via CLI ──────────────────────────────────
if _args.batch_size is not None:
    print(f"  CLI override : batch_size {CONFIG['batch_size']} → {_args.batch_size}")
    CONFIG['batch_size'] = _args.batch_size

_nvfp4_active_global = (
    CONFIG['use_nvfp4'] and _TE_AVAILABLE and _nvfp4_recipe is not None
)

print(f"\nCONFIG :")
print(f"  embed={CONFIG['embed_dim']}  layers={CONFIG['num_layers']}  "
      f"heads={CONFIG['num_heads']}  kv={CONFIG['n_kv_heads']}")
print(f"  packing={'ON' if CONFIG['use_packing'] else 'OFF'}  "
      f"seq_len={CONFIG['max_seq_len']}")
print(f"  NVfp4={'ON ✅' if _nvfp4_active_global else 'OFF (BF16 classique)'}  "
      f"healing_ratio={CONFIG['healing_ratio']:.0%}")
if device == 'cuda':
    print(f"  GPU={torch.cuda.get_device_name(0)}  "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    cap = torch.cuda.get_device_capability()
    print(f"  Compute capability: SM{cap[0]}{cap[1]}")


# ============================================================
# HUGGING FACE — Download données + checkpoint
# ============================================================
def _hf_download():
    """
    1. Télécharge le repo HF (données + checkpoint éventuel) dans le répertoire courant.
    2. Déplace le checkpoint .pt (et _info.json) vers ./Model/ pour que
       CheckpointManager le trouve à l'emplacement CONFIG['checkpoint_file'].
    """
    if not _HF_TOKEN:
        print("  --hf-token absent : skip download HF (mode local)")
        return

    # Si le checkpoint et les données existent déjà localement → skip download
    _pt_local   = CONFIG['checkpoint_file']
    _data_local = CONFIG['data_file']
    if os.path.exists(_pt_local) and os.path.exists(_data_local):
        print(f"  Fichiers locaux présents — skip download HF")
        return

    try:
        import shutil
        from huggingface_hub import snapshot_download, list_repo_files
        print(f"\nHugging Face — download depuis {_HF_REPO}")
        snapshot_download(
            repo_id   = _HF_REPO,
            repo_type = 'dataset',
            local_dir = '.',
            token     = _HF_TOKEN,
            ignore_patterns = ['*.md', '*.txt', '.gitattributes'],
        )
        print("  Download terminé")

        # ── Déplacement checkpoint vers ./Model/ ─────────────────
        model_dir = os.path.dirname(CONFIG['checkpoint_file'])   # './Model'
        os.makedirs(model_dir, exist_ok=True)

        pt_name   = os.path.basename(CONFIG['checkpoint_file'])  # 'HessGpt_pretrain.pt'
        json_name = pt_name.replace('.pt', '_info.json')

        for fname in (pt_name, json_name):
            src = os.path.join('.', fname)
            dst = os.path.join(model_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"  Checkpoint déplacé : {src} → {dst}")
            elif os.path.exists(src) and os.path.exists(dst):
                # Le fichier local est plus récent → on écrase
                shutil.move(src, dst)
                print(f"  Checkpoint mis à jour : {dst}")

    except Exception as e:
        print(f"  WARN HF download : {e}")

_hf_download()

def hf_push_checkpoint(local_pt_path, step, epoch):
    """Envoie le checkpoint .pt et le _info.json dans le repo HF dataset."""
    if not _HF_TOKEN:
        return
    try:
        from huggingface_hub import HfApi
        api      = HfApi(token=_HF_TOKEN)
        pt_name  = os.path.basename(local_pt_path)
        json_path = local_pt_path.replace('.pt', '_info.json')
        api.upload_file(
            path_or_fileobj = local_pt_path,
            path_in_repo    = pt_name,
            repo_id         = _HF_REPO,
            repo_type       = 'dataset',
            commit_message  = f'checkpoint step={step:,} epoch={epoch}',
        )
        if os.path.exists(json_path):
            api.upload_file(
                path_or_fileobj = json_path,
                path_in_repo    = os.path.basename(json_path),
                repo_id         = _HF_REPO,
                repo_type       = 'dataset',
                commit_message  = f'info step={step:,} epoch={epoch}',
            )
        print(f"  HF push OK → {_HF_REPO}  step={step:,}")
    except Exception as e:
        print(f"  WARN HF push : {e}")

# ============================================================
# CHARGEMENT DONNÉES — pretrain_data.bin
# ============================================================
_data_file = CONFIG['data_file']
if not os.path.exists(_data_file):
    print(f"ERREUR : fichier introuvable → {_data_file}")
    sys.exit(1)

print(f"\nChargement données : {_data_file}")
# memmap — données lues depuis le disque à la demande, pas de copie RAM
_memmap_probe = np.memmap(_data_file, dtype=np.uint16, mode='r')
_total_tok    = len(_memmap_probe)
del _memmap_probe   # fermé ici — chaque worker rouvrira lazily
print(f"  {_total_tok/1e9:.3f}B tokens  ({_total_tok*2/1e9:.1f}GB sur disque, memmap lazy)")

# .bin déjà shufflé — indices séquentiels, pas de shuffle supplémentaire
_seq_len_1  = CONFIG['max_seq_len'] + 1
_n_seqs     = _total_tok // _seq_len_1
_idx        = np.arange(_n_seqs)

_val_seqs   = min(CONFIG['val_tokens'] // _seq_len_1, int(_n_seqs * 0.05))
_val_seqs   = max(_val_seqs, 1)
_val_size   = _val_seqs * _seq_len_1
_train_size = (_n_seqs - _val_seqs) * _seq_len_1

TRAIN_IDX = _idx[_val_seqs:]   # indices train (séquentiels)
VAL_IDX   = _idx[:_val_seqs]   # indices val
print(f"  train={_train_size/1e9:.3f}B  val={_val_size/1e6:.0f}M tokens  (memmap lazy, no shuffle)")

_samples_per_epoch = len(TRAIN_IDX)
_batches_per_epoch = math.ceil(_samples_per_epoch / CONFIG['batch_size'])
STEPS_PER_EPOCH    = max(math.ceil(_batches_per_epoch / CONFIG['gradient_accumulation']), 1)
TOTAL_STEPS        = STEPS_PER_EPOCH * CONFIG['num_epochs']
print(f"  steps/epoch={STEPS_PER_EPOCH:,}  total={TOTAL_STEPS:,}")


# ============================================================
# TOKENIZER
# ============================================================
print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG['vocab_size'] = len(tokenizer)
print(f"  vocab={len(tokenizer)}")


# ============================================================
# WSD SCHEDULER
# ============================================================
class WSDScheduler:
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.03, decay_ratio=0.15, min_lr_ratio=0.1):
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0

    def get_lr(self):
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * (s / max(self.warmup_steps, 1))
        elif s < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            d = s - self.warmup_steps - self.stable_steps
            p = min(d / max(self.decay_steps, 1), 1.0)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    def step(self):
        lr = self.get_lr()
        self.current_step += 1
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg['lr'] = lr * 5.0 if pg.get('is_muon', False) else lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, sd):
        self.current_step = sd['current_step']


# ============================================================
# DATASETS — Standard + Packed
# ============================================================

class ChunkSubset(Dataset):
    """Dataset standard (padding classique, comportement v8)."""
    def __init__(self, data_path, idx, seq_len, pad_token_id):
        self.data_path    = data_path
        self._data        = None      # ouvert lazily par chaque worker
        self.idx          = idx
        self.seq_len      = seq_len
        self.pad_token_id = pad_token_id
        self.num_samples  = len(idx)

    def _get_data(self):
        if self._data is None:
            self._data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        return self._data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        start = int(self.idx[i]) * (self.seq_len + 1)
        chunk = torch.from_numpy(
            self._get_data()[start : start + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1].clone(), chunk[1:].clone()


class PackedChunkDataset(Dataset):
    """
    Sequence Packing : pack plusieurs documents dans un bloc de max_seq_len tokens.

    Principe :
      - Les tokens sont lus depuis data_path (memmap lazy, séquences back-to-back)
      - On découpe en blocs de max_seq_len tokens (chaque bloc = 1 sample)
      - Le collate_fn calcule les cu_seqlens à partir des positions EOS dans chaque bloc

    Pas de padding → 100% tokens utiles.
    """
    def __init__(self, data_path, idx, seq_len, eos_token_id):
        self.data_path    = data_path
        self._data        = None      # ouvert lazily par chaque worker
        self.idx          = idx
        self.seq_len      = seq_len
        self.eos_token_id = eos_token_id
        self.num_samples  = len(idx)

    def _get_data(self):
        if self._data is None:
            self._data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        return self._data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        start = int(self.idx[i]) * (self.seq_len + 1)
        block = torch.from_numpy(
            self._get_data()[start : start + self.seq_len + 1].astype(np.int64)
        )
        x = block[:-1].clone()   # [seq_len]
        y = block[1:].clone()    # [seq_len]
        return x, y


def packed_collate_fn(batch, eos_token_id, seq_len):
    """
    Collate pour Sequence Packing.
    Calcule cu_seqlens_q en détectant les EOS dans chaque séquence du batch.

    Retourne :
      x           : [batch, seq_len]
      y           : [batch, seq_len]
      cu_seqlens  : [batch * n_seqs_per_sample + 1] int32 — pour flash_attn_varlen
      max_seqlen  : int — longueur max d'une sous-séquence dans le batch
    """
    xs, ys = zip(*batch)
    x = torch.stack(xs)   # [B, seq_len]
    y = torch.stack(ys)   # [B, seq_len]

    # Calcul cu_seqlens pour chaque item du batch
    # On concatène les séquences de tout le batch en une seule dim
    # (c'est ce que flash_attn_varlen_func attend)
    all_cu = [0]
    max_sl = 1

    for i in range(x.size(0)):
        seq = x[i]
        # Positions des EOS dans cette séquence (début de nouvelle doc)
        eos_pos = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) == 0:
            # Pas d'EOS → toute la séquence est un seul doc
            all_cu.append(all_cu[-1] + seq_len)
            max_sl = max(max_sl, seq_len)
        else:
            prev = 0
            for pos in eos_pos.tolist():
                l = pos - prev + 1   # +1 pour inclure l'EOS
                if l > 0:
                    all_cu.append(all_cu[-1] + l)
                    max_sl = max(max_sl, l)
                prev = pos + 1
            # Dernier segment après le dernier EOS
            remaining = seq_len - prev
            if remaining > 0:
                all_cu.append(all_cu[-1] + remaining)
                max_sl = max(max_sl, remaining)

    cu_seqlens = torch.tensor(all_cu, dtype=torch.int32)
    return x, y, cu_seqlens, max_sl


class SeededSampler(torch.utils.data.Sampler):
    def __init__(self, n, seed, skip_samples=0):
        self.n            = n
        self.seed         = seed
        self.skip_samples = min(skip_samples, n)
        rng               = np.random.default_rng(seed)
        indices           = rng.permutation(n)
        self._indices     = indices[self.skip_samples:]
        print(f"  SeededSampler : n={n:,}  seed={seed}  "
              f"skip={self.skip_samples:,}  restant={len(self._indices):,}")

    def __iter__(self):
        return iter(self._indices.tolist())

    def __len__(self):
        return len(self._indices)




# ============================================================
# BENCHMARK — tokens/sec + MFU
# ============================================================
def estimate_model_flops(model, seq_len):
    """
    Estimation FLOPs par forward pass (formule Chinchilla approximative).
    6 * N * seq_len pour un transformer dense (attention + FFN).
    """
    N = sum(p.numel() for p in model.parameters())
    return 6 * N * seq_len


@torch.no_grad()
def run_benchmark(model, vocab_size, seq_len, batch_size, steps=20,
                  dtype=torch.bfloat16, use_nvfp4=False):
    """
    Mesure tokens/sec et MFU sur B200.
    use_nvfp4=True : wrappe avec te.fp8_autocast pour mesurer les vraies perfs NVfp4.
    Retourne un dict de métriques.
    """
    model.eval()
    gpu_tflops = 1.0
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap == (12, 0) or cap[0] > 12:
            # B200 SM120 : 9000 TFLOPs NVfp4, 4500 TFLOPs BF16
            gpu_tflops = 9000.0 if use_nvfp4 else 4500.0
        elif cap[0] >= 9:
            # H100 SXM : ~3958 TFLOPs FP8, ~1979 TFLOPs BF16
            gpu_tflops = 3958.0 if use_nvfp4 else 1979.0
        elif cap[0] >= 8:
            # A100 : ~624 TFLOPs FP8, ~312 TFLOPs BF16
            gpu_tflops = 624.0 if use_nvfp4 else 312.0

    flops_per_fwd = estimate_model_flops(model, seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # FA2/FA3/FA4 refusent float32 — cast le modèle en bf16 pour le benchmark
    model_dtype = next(model.parameters()).dtype
    if dtype == torch.bfloat16 and model_dtype == torch.float32:
        model = model.to(torch.bfloat16)

    # Factory NVfp4 — recréé à chaque appel (un GeneratorContextManager ne peut pas être réutilisé)
    def fp4_ctx():
        if use_nvfp4 and _TE_AVAILABLE and _nvfp4_recipe is not None:
            return te.fp8_autocast(enabled=True, fp8_recipe=_nvfp4_recipe)
        return contextlib.nullcontext()

    # Warmup
    for _ in range(3):
        with fp4_ctx():
            with torch.amp.autocast(device, dtype=dtype):
                model(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        with fp4_ctx():
            with torch.amp.autocast(device, dtype=dtype):
                model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_tokens = batch_size * seq_len * steps
    tokens_per_sec = total_tokens / elapsed
    flops_per_sec  = flops_per_fwd * batch_size * steps / elapsed
    mfu = flops_per_sec / (gpu_tflops * 1e12) * 100  # %

    model.train()
    return {
        'tokens_per_sec': tokens_per_sec,
        'mfu_pct':        mfu,
        'elapsed_steps':  steps,
        'dtype':          str(dtype),
        'use_nvfp4':      use_nvfp4,
    }


def print_benchmark(label, metrics):
    print(f"\n{'─'*60}")
    print(f"  BENCHMARK : {label}")
    print(f"  tokens/sec : {metrics['tokens_per_sec']:,.0f}")
    print(f"  MFU        : {metrics['mfu_pct']:.1f}%")
    print(f"  dtype      : {metrics['dtype']}")
    print(f"{'─'*60}")


# ============================================================
# CHECKPOINT MANAGER
# ============================================================
class CheckpointManager:
    def __init__(self, path):
        self.path          = path
        self._last_hf_push = 0.0
        self._save_thread  = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _write(self, cp, info_snapshot, json_path, step, epoch):
        new_path = json_path + '.new'
        with open(new_path, 'w') as f:
            json.dump(info_snapshot, f, indent=2, default=str)
        tmp = self.path + '.tmp'
        torch.save(cp, tmp)
        os.replace(tmp, self.path)
        if os.path.exists(json_path):
            os.remove(json_path)
        os.replace(new_path, json_path)
        # ── Push HF (dans le thread background) ──────────────────
        if _HF_TOKEN:
            hf_push_checkpoint(self.path, step=step, epoch=epoch)

    def save(self, model, optimizers, scheduler, metadata):
        # Attendre la sauvegarde précédente si encore en cours
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        muon_opt, adamw_opt = optimizers
        info_snapshot = {**metadata, 'last_save': datetime.now().isoformat(), 'config': CONFIG}
        # state_dict() copie les tenseurs en CPU — snapshot atomique dans le thread principal
        cp = {
            'model_state_dict':     m.state_dict(),
            'muon_state_dict':      muon_opt.state_dict(),
            'adamw_state_dict':     adamw_opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metadata':             info_snapshot,
        }
        json_path = self.path.replace('.pt', '_info.json')
        print(f"  SAVE → epoch={metadata['current_epoch']}  "
              f"step={metadata['global_step']:,}  [{self.path}] (async)")
        # Écriture disque dans un thread background — le GPU continue à tourner
        self._save_thread = threading.Thread(
            target=self._write,
            args=(cp, info_snapshot, json_path,
                  metadata['global_step'], metadata['current_epoch']),
            daemon=True,
        )
        self._save_thread.start()

    def wait(self):
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f"\nCheckpoint trouvé : {self.path}")
        cp = torch.load(self.path, map_location='cpu', weights_only=False)
        json_path = self.path.replace('.pt', '_info.json')
        new_path  = json_path + '.new'
        if os.path.exists(new_path):
            if os.path.exists(json_path):
                os.remove(json_path)
            os.replace(new_path, json_path)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                info = json.load(f)
            for k in ('global_step', 'current_epoch', 'epoch_start_step',
                      'skip_batches', 'total_training_time', 'training_history'):
                default = 1 if k == 'current_epoch' else (0.0 if k == 'total_training_time' else 0)
                cp[k] = info.get(k, default)
        else:
            cp.update({'global_step': 0, 'current_epoch': 1,
                       'epoch_start_step': 0, 'skip_batches': 0,
                       'total_training_time': 0.0,
                       'training_history': {'validations': [], 'epochs': []}})

        # ── Sanity check skip_batches ─────────────────────────────
        # skip_batches doit toujours être égal à global_step * gradient_accumulation
        # Si ce n'est pas le cas (sauvegarde corrompue / bug), on corrige automatiquement
        _expected_skip = cp.get('global_step', 0) * CONFIG['gradient_accumulation']
        _actual_skip   = cp.get('skip_batches', 0)
        if _actual_skip != _expected_skip:
            print(f"  ⚠️  skip_batches corrigé : {_actual_skip:,} → {_expected_skip:,} "
                  f"(step={cp.get('global_step',0):,} × grad_acc={CONFIG['gradient_accumulation']})")
            cp['skip_batches'] = _expected_skip
            # Mettre à jour le json aussi pour la cohérence
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    _info = json.load(f)
                _info['skip_batches'] = _expected_skip
                with open(json_path, 'w') as f:
                    json.dump(_info, f, indent=2, default=str)
        else:
            print(f"  ✅ skip_batches OK : {_actual_skip:,} "
                  f"(step={cp.get('global_step',0):,} × grad_acc={CONFIG['gradient_accumulation']})")

        return cp


# ============================================================
# VALIDATION
# ============================================================
@torch.no_grad()
def validate(model, val_loader, max_batches=50):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    n = 0
    ae  = (device == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32
    try:
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            # Val toujours en mode standard (pas de cu_seqlens)
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast(device, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
            total_loss += loss.detach()
            n += 1
    finally:
        model.train()
    avg = (total_loss / max(n, 1)).item()  # unique sync CPU-GPU
    return math.exp(min(avg, 10)), avg


# ============================================================
# MARS-M + MUON
# ============================================================
def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        use_mars=use_mars, mars_gamma=mars_gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, nesterov = group['lr'], group['momentum'], group['nesterov']
            ns_steps, wd           = group['ns_steps'], group['weight_decay']
            use_mars, mars_gamma   = group.get('use_mars', True), group.get('mars_gamma', 0.025)
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    continue
                state = self.state[p]
                if use_mars:
                    if 'prev_grad' not in state:
                        state['prev_grad'] = torch.zeros_like(g)
                    prev_g    = state['prev_grad']
                    norm_g    = g.norm() + 1e-8
                    norm_prev = prev_g.norm() + 1e-8
                    c_t       = torch.clamp((mars_gamma / (1.0 - mars_gamma)) * (norm_g / norm_prev), max=1.0)
                    g         = g + c_t * (g - prev_g)
                    state['prev_grad'].copy_(p.grad)
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = (g + momentum * buf) if nesterov else buf
                g     = zeropower_via_newtonschulz5(g, steps=ns_steps)
                scale = max(g.size(0), g.size(1)) ** 0.5
                g     = g * scale
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(model, lr, weight_decay, betas, eps):
    MUON_EXCLUDE = {'token_embeddings.weight', 'output_head.weight',
                    'position_embeddings.weight'}
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn in MUON_EXCLUDE:
            (adamw_decay if p.dim() >= 2 else adamw_nodecay).append(p)
            continue
        if p.dim() >= 2 and pn.startswith('blocks.'):
            muon_params.append(p)
        elif p.dim() < 2 and pn.startswith('blocks.'):
            adamw_nodecay.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)

    lr_muon  = lr * 5.0
    muon_opt = Muon(
        [{'params': muon_params, 'is_muon': True}],
        lr=lr_muon, momentum=0.95, nesterov=True,
        ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025,
    )
    muon_opt.param_groups[0]['is_muon'] = True
    adamw_opt = torch.optim.AdamW(
        [
            {'params': adamw_decay,   'weight_decay': weight_decay, 'is_muon': False},
            {'params': adamw_nodecay, 'weight_decay': 0.0,          'is_muon': False},
        ],
        lr=lr, betas=betas, eps=eps, fused=(device == 'cuda'),
        capturable=(device == 'cuda'),
    )
    print(f"\nOptimizer Muon+MARS-M + AdamW :")
    print(f"  Muon : {len(muon_params)} tenseurs  lr={lr_muon:.2e}")
    print(f"  AdamW: {len(adamw_decay)} decay + {len(adamw_nodecay)} no-decay  lr={lr:.2e}")
    return muon_opt, adamw_opt


# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_epoch(
    model, optimizers, scheduler,
    checkpoint_manager, training_history,
    global_step, total_training_time,
    current_epoch, epoch_start_step,
    skip_batches=0,
    raw_model=None,
):
    muon_opt, adamw_opt = optimizers
    label = f"Epoch {current_epoch}/{CONFIG['num_epochs']}"
    print(f"\n{'='*80}\n  {label}\n{'='*80}")

    # ── Datasets ────────────────────────────────────────────────
    if CONFIG['use_packing']:
        train_ds = PackedChunkDataset(
            _data_file, TRAIN_IDX, CONFIG['max_seq_len'],
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        train_ds = ChunkSubset(_data_file, TRAIN_IDX, CONFIG['max_seq_len'], tokenizer.pad_token_id)

    val_ds  = ChunkSubset(_data_file, VAL_IDX, CONFIG['max_seq_len'], tokenizer.pad_token_id)
    total_seqs = len(train_ds)

    if skip_batches >= math.ceil(total_seqs / CONFIG['batch_size']):
        print(f"  Epoch déjà traitée, skip.")
        return global_step, total_training_time, epoch_start_step

    # .bin déjà shufflé — sampler séquentiel avec skip pour reprise checkpoint
    _skip_samples = skip_batches * CONFIG['batch_size']
    sampler       = torch.utils.data.SequentialSampler(
        range(_skip_samples, total_seqs)
    )
    print(f"  Sampler séquentiel : n={total_seqs - _skip_samples:,}  skip={_skip_samples:,}")

    # ── Collate selon mode ───────────────────────────────────────
    if CONFIG['use_packing']:
        from functools import partial
        _collate = partial(
            packed_collate_fn,
            eos_token_id = tokenizer.eos_token_id,
            seq_len      = CONFIG['max_seq_len'],
        )
    else:
        _collate = None

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], sampler=sampler,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=3,
        drop_last=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=3,
    )

    total_batches = total_seqs // CONFIG['batch_size']
    num_batches   = len(train_loader)
    print(f"  train={total_batches:,} batches | restant={num_batches:,} | "
          f"val={len(val_loader):,} | packing={'ON' if CONFIG['use_packing'] else 'OFF'}")

    model.train()
    epoch_loss_t    = torch.zeros(1, device=device)
    valid_batches   = 0
    accumulated_steps = 0
    running_loss_t  = torch.zeros(1, device=device)
    running_batches = 0
    t_start = time.time()
    ae  = (device == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32

    healing_step    = int(TOTAL_STEPS * CONFIG['healing_ratio'])
    _healing_logged = False

    pbar = tqdm(train_loader, desc=label, leave=True,
                initial=total_batches - num_batches, total=total_batches)

    for batch_idx, batch in enumerate(pbar):
        try:
            if CONFIG['use_packing'] and len(batch) == 4:
                x, y, cu_seqlens, max_seqlen = batch
                x          = x.to(device, non_blocking=True)
                y          = y.to(device, non_blocking=True)
                cu_seqlens = cu_seqlens.to(device, non_blocking=True)
            else:
                x, y       = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                cu_seqlens = None
                max_seqlen = None

            use_fp4_now = _nvfp4_active_global and global_step < healing_step
            if _nvfp4_active_global and not use_fp4_now and not _healing_logged:
                print(f"\n  HEALING activé à step={global_step:,} "
                      f"(seuil={healing_step:,}) — forward repassé en BF16")
                _healing_logged = True
                # ── Gradient Checkpointing pour compenser la hausse VRAM BF16 ──
                if CONFIG.get('gc_on_healing', True) and raw_model is not None:
                    raw_model.enable_gradient_checkpointing()
                    print(f"  Gradient checkpointing activé (gc_on_healing=True) — "
                          f"VRAM activations réduite ~30-40%")

            fp4_ctx = (
                te.fp8_autocast(enabled=True, fp8_recipe=_nvfp4_recipe)
                if use_fp4_now else contextlib.nullcontext()
            )

            with fp4_ctx:
                with torch.amp.autocast(device, dtype=adt, enabled=ae):
                    _, loss, _ = model(
                        x, targets=y,
                        pad_token_id = tokenizer.pad_token_id,
                        cu_seqlens_q = cu_seqlens,
                        cu_seqlens_k = cu_seqlens,
                        max_seqlen_q = max_seqlen,
                        max_seqlen_k = max_seqlen,
                    )
                    loss = loss / CONFIG['gradient_accumulation']

            loss.backward()
            accumulated_steps += 1

            is_last = (batch_idx + 1 == num_batches)
            if (accumulated_steps % CONFIG['gradient_accumulation'] == 0) or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'], foreach=True)
                muon_opt.step()
                adamw_opt.step()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                scheduler.step()
                accumulated_steps = 0
                global_step += 1

                if global_step % CONFIG['validate_every_steps'] == 0:
                    val_ppl, val_loss = validate(model, val_loader, CONFIG['val_batches'])
                    avg = (running_loss_t / max(running_batches, 1)).item()  # sync unique ici
                    print(f"\n  step={global_step:,} | "
                          f"train={avg:.4f} ppl={math.exp(min(avg,10)):.1f} | "
                          f"val={val_loss:.4f} ppl={val_ppl:.1f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e}\n")
                    training_history['validations'].append({
                        'step': global_step, 'current_epoch': current_epoch,
                        'val_loss': val_loss, 'val_ppl': val_ppl,
                        'train_loss': avg, 'lr': scheduler.get_last_lr()[0],
                    })
                    running_loss_t  = torch.zeros(1, device=device)
                    running_batches = 0

                if global_step % CONFIG['save_every_steps'] == 0:
                    checkpoint_manager.save(model, optimizers, scheduler, metadata={
                        'current_epoch':       current_epoch,
                        'global_step':         global_step,
                        'epoch_start_step':    epoch_start_step,
                        'skip_batches':        batch_idx + 1,
                        'total_training_time': total_training_time + (time.time() - t_start),
                        'training_history':    training_history,
                    })

            raw_t = loss.detach() * CONFIG['gradient_accumulation']
            epoch_loss_t    += raw_t
            running_loss_t  += raw_t
            valid_batches   += 1
            running_batches += 1

            if batch_idx % 20 == 0:
                # Sync CPU-GPU seulement toutes les 20 itérations
                raw_f = raw_t.item()
                avg   = (running_loss_t / max(running_batches, 1)).item()
                pbar.set_postfix(
                    loss=f'{raw_f:.4f}', avg=f'{avg:.4f}',
                    ppl=f'{math.exp(min(avg,10)):.1f}',
                    lr=f'{scheduler.get_last_lr()[0]:.2e}',
                    step=f'{global_step:,}',
                )

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n  OOM batch {batch_idx} — skip")
                torch.cuda.empty_cache()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                accumulated_steps = 0
                gc.collect()
                model.train()
                continue
            raise

    pbar.close()
    train_loader._iterator = None
    del train_loader
    val_loader._iterator = None
    del val_loader

    elapsed = time.time() - t_start
    total_training_time += elapsed
    avg_loss = (epoch_loss_t / max(valid_batches, 1)).item()  # unique sync fin d'epoch
    print(f"\n  Epoch {current_epoch} terminée | loss={avg_loss:.4f} | {elapsed/60:.1f}min")

    training_history['epochs'].append({
        'epoch': current_epoch, 'train_loss': avg_loss,
        'time_sec': elapsed, 'global_step': global_step,
    })

    return global_step, total_training_time, epoch_start_step


# ============================================================
# MAIN
# ============================================================
def main():
    from HessGpt import HessGPT

    print('\n' + '='*80 + '\nCREATION MODELE\n' + '='*80)

    ckpt_mgr = CheckpointManager(CONFIG['checkpoint_file'])

    model = HessGPT(
        vocab_size=CONFIG['vocab_size'], embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'], num_layers=CONFIG['num_layers'],
        max_seq_len=CONFIG['max_seq_len'], dropout=CONFIG['dropout'],
        use_rope=CONFIG['use_rope'], use_yarn=CONFIG['use_yarn'],
        yarn_scale=CONFIG['yarn_scale'],
        yarn_original_max_len=CONFIG['yarn_original_max_len'],
        use_swiglu=CONFIG['use_swiglu'], n_kv_heads=CONFIG['n_kv_heads'],
        use_qk_norm=CONFIG['use_qk_norm'], soft_cap=CONFIG['soft_cap'],
        use_flash_attn=CONFIG['use_flash_attn'],
        use_nvfp4=_nvfp4_active_global,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params : {total_params/1e6:.1f}M")
    print(f"  Précision GEMMs : {'NVfp4 (te.Linear)' if _nvfp4_active_global else 'BF16 (nn.Linear)'}")
    if _nvfp4_active_global:
        healing_step_main = int(TOTAL_STEPS * CONFIG['healing_ratio'])
        print(f"  Healing BF16 à partir du step {healing_step_main:,} "
              f"({CONFIG['healing_ratio']:.0%} de {TOTAL_STEPS:,} steps)")
        if CONFIG.get('gc_on_healing', True):
            print(f"  Gradient checkpointing activé automatiquement au healing "
                  f"(gc_on_healing=True)")

    # ── BENCHMARK PHASE 0 (baseline avant compile) ───────────────
    if device == 'cuda':
        print(f"\n{'='*60}")
        _bm0_label = f"Phase 0 — {'NVfp4' if _nvfp4_active_global else 'BF16'} no compile"
        print(f"  BENCHMARK PHASE 0 — Baseline {_bm0_label}")
        bm_baseline = run_benchmark(
            model, CONFIG['vocab_size'],
            CONFIG['max_seq_len'], min(CONFIG['batch_size'], 32),
            steps=CONFIG['benchmark_steps'],
            use_nvfp4=_nvfp4_active_global,
        )
        print_benchmark(_bm0_label, bm_baseline)

    if CONFIG['use_compile'] and device == 'cuda':
        print('torch.compile...')
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors  = True
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print('  OK')
        except Exception as e:
            print(f'  FAIL : {e}')

    # ── BENCHMARK PHASE 1 (après compile + FA) ───────────────────
    if device == 'cuda':
        print(f"\n{'='*60}")
        _bm1_label = f"Phase 1 — {'NVfp4' if _nvfp4_active_global else 'BF16'} + FA2 + compile"
        print(f"  BENCHMARK PHASE 1 — {_bm1_label}")
        # Warmup compile (premier forward lent)
        dummy = torch.randint(0, CONFIG['vocab_size'],
                              (4, CONFIG['max_seq_len']), device=device)
        _fp4_warm_ctx = (
            te.fp8_autocast(enabled=True, fp8_recipe=_nvfp4_recipe)
            if _nvfp4_active_global else contextlib.nullcontext()
        )
        with _fp4_warm_ctx:
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                model(dummy)
        torch.cuda.synchronize()

        bm_phase1 = run_benchmark(
            model, CONFIG['vocab_size'],
            CONFIG['max_seq_len'], min(CONFIG['batch_size'], 32),
            steps=CONFIG['benchmark_steps'],
            use_nvfp4=_nvfp4_active_global,
        )
        print_benchmark(_bm1_label, bm_phase1)

        if 'bm_baseline' in dir():
            speedup = bm_phase1['tokens_per_sec'] / bm_baseline['tokens_per_sec']
            print(f"\n  Speedup Phase1 vs Phase0 : {speedup:.2f}x")
            print(f"  MFU Phase0 : {bm_baseline['mfu_pct']:.1f}%")
            print(f"  MFU Phase1 : {bm_phase1['mfu_pct']:.1f}%")

    raw_model  = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizers = configure_optimizers(
        raw_model, CONFIG['learning_rate'], CONFIG['weight_decay'],
        (CONFIG['adam_beta1'], CONFIG['adam_beta2']), CONFIG['adam_eps'],
    )
    muon_opt, adamw_opt = optimizers

    scheduler = WSDScheduler(
        list(optimizers), max_lr=CONFIG['learning_rate'],
        total_steps=TOTAL_STEPS, warmup_ratio=CONFIG['warmup_ratio'],
        decay_ratio=CONFIG['decay_ratio'], min_lr_ratio=CONFIG['min_lr_ratio'],
    )

    training_history = {
        'config': CONFIG, 'total_params': total_params,
        'total_steps': TOTAL_STEPS, 'validations': [],
        'epochs': [], 'start_time': datetime.now().isoformat(),
        'benchmarks': {},
    }
    if device == 'cuda' and 'bm_baseline' in dir():
        training_history['benchmarks']['phase0'] = bm_baseline
        training_history['benchmarks']['phase1'] = bm_phase1

    global_step         = 0
    current_epoch       = 1
    epoch_start_step    = 0
    skip_batches        = 0
    total_training_time = 0.0

    cp = ckpt_mgr.load()
    if cp:
        print('\nREPRISE')
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(cp['model_state_dict'])
        if 'muon_state_dict' in cp:
            muon_opt.load_state_dict(cp['muon_state_dict'])
            adamw_opt.load_state_dict(cp['adamw_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])
        current_epoch       = cp.get('current_epoch', 1)
        global_step         = cp.get('global_step', 0)
        epoch_start_step    = cp.get('epoch_start_step', 0)
        skip_batches        = cp.get('skip_batches', 0)
        total_training_time = cp.get('total_training_time', 0.0)
        training_history    = cp.get('training_history', training_history)
        if current_epoch > CONFIG['num_epochs']:
            print('Training déjà terminé.')
            return

    print('\n' + '='*80)
    print(f'TRAINING START — packing={"ON" if CONFIG["use_packing"] else "OFF"}')
    print('='*80)

    for epoch in range(current_epoch, CONFIG['num_epochs'] + 1):
        _skip = skip_batches if epoch == current_epoch else 0
        if epoch != current_epoch:
            epoch_start_step = global_step

        try:
            global_step, total_training_time, epoch_start_step = train_epoch(
                model=model, optimizers=optimizers, scheduler=scheduler,
                checkpoint_manager=ckpt_mgr, training_history=training_history,
                global_step=global_step, total_training_time=total_training_time,
                current_epoch=epoch, epoch_start_step=epoch_start_step,
                skip_batches=_skip,
                raw_model=raw_model,
            )
            cp = None
            skip_batches = 0
        except KeyboardInterrupt:
            print('\nCTRL+C')
            ckpt_mgr.save(model, optimizers, scheduler, metadata={
                'current_epoch': epoch, 'global_step': global_step,
                'epoch_start_step': epoch_start_step, 'skip_batches': 0,
                'total_training_time': total_training_time,
                'training_history': training_history,
            })
            return
        except Exception:
            print(f'\nERREUR:\n{traceback.format_exc()}')
            ckpt_mgr.save(model, optimizers, scheduler, metadata={
                'current_epoch': epoch, 'global_step': global_step,
                'epoch_start_step': epoch_start_step, 'skip_batches': 0,
                'total_training_time': total_training_time,
                'training_history': training_history,
            })
            raise

    print(f'\n{"="*80}\nTRAINING TERMINÉ\n{"="*80}')
    print(f'  Steps : {global_step:,}  Temps : {total_training_time/3600:.2f}h')

    ckpt_mgr.save(model, optimizers, scheduler, metadata={
        'current_epoch': CONFIG['num_epochs'] + 1, 'global_step': global_step,
        'epoch_start_step': global_step, 'skip_batches': 0,
        'total_training_time': total_training_time,
        'training_history': training_history,
    })
    ckpt_mgr.wait()  # attendre que la dernière sauvegarde async soit terminée
    history_path = CONFIG['checkpoint_file'].replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2, default=str)
    print(f'  History : {history_path}')
    print('DONE')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompu')
    except Exception:
        print(traceback.format_exc())
    finally:
        print('\nBye')
