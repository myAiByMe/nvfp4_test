# inference.py — HessGPT v10
"""
Script d'inférence pour HessGPT.
Charge le modèle depuis HuggingFace Hub ou un fichier local.

Gère automatiquement les checkpoints entraînés avec use_nvfp4=True
(poids te.Linear avec clés _extra_state) en les convertissant pour
une inférence standard BF16/FP32 sans Transformer Engine.

Usage :
  python inference.py
  python inference.py --checkpoint ./HessGpt_pretrain.pt
  python inference.py --prompt "Il était une fois"
  python inference.py --hf_repo silyan/nvfp4_test_9Btokens --hf_file HessGpt_pretrain.pt
"""

import argparse
import sys
import os

# ── Résolution des imports depuis Core/ ──────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "Core", "Model"))
sys.path.insert(0, os.path.join(_HERE, "Core", "Attention"))
sys.path.insert(0, os.path.join(_HERE, "Core", "TransformerBlock"))
sys.path.insert(0, os.path.join(_HERE, "Core", "FeedForward"))

import torch
from transformers import AutoTokenizer

from HessGpt import HessGPT

# ─────────────────────────────────────────────────────────────────────────────
# Config du modèle par défaut (identique à pretrain.py)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_CONFIG = {
    "embed_dim"             : 1280,
    "num_heads"             : 20,
    "num_layers"            : 24,
    "max_seq_len"           : 1024,
    "dropout"               : 0.0,
    "use_rope"              : True,
    "use_yarn"              : False,
    "yarn_scale"            : 4.0,
    "yarn_original_max_len" : 512,
    "use_swiglu"            : True,
    "n_kv_heads"            : 5,
    "use_qk_norm"           : True,
    "soft_cap"              : None,
    "use_flash_attn"        : True,
    "use_nvfp4"             : False,   # toujours False en inférence
}

HF_REPO_DEFAULT = "silyan/nvfp4_test_9Btokens"
HF_FILE_DEFAULT = "HessGpt_pretrain.pt"
TOKENIZER_ID    = "HuggingFaceTB/cosmo2-tokenizer"


# ─────────────────────────────────────────────────────────────────────────────
# Nettoyage du state_dict (poids te.Linear → nn.Linear compatibles)
# ─────────────────────────────────────────────────────────────────────────────

def clean_state_dict(state_dict: dict) -> dict:
    """
    Supprime les clés _extra_state générées par te.Linear (Transformer Engine).
    Ces clés sont des métadonnées FP8 inutiles en inférence BF16/FP32.

    te.Linear et nn.Linear partagent le même format pour 'weight' et 'bias',
    donc seules les clés _extra_state sont à retirer.
    """
    cleaned = {}
    removed = 0
    for k, v in state_dict.items():
        if k.endswith("._extra_state"):
            removed += 1
            continue
        cleaned[k] = v

    if removed > 0:
        print(f"    🧹  Suppression de {removed} clés _extra_state (te.Linear → nn.Linear)")

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def download_from_hf(repo_id: str, filename: str, cache_dir: str = ".hf_cache") -> str:
    """Télécharge le fichier depuis HuggingFace Hub et retourne le chemin local."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌  huggingface_hub non installé. Faites : pip install huggingface-hub")
        sys.exit(1)

    print(f"📥  Téléchargement depuis HuggingFace ...")
    print(f"    repo : {repo_id}")
    print(f"    file : {filename}")

    local_path = hf_hub_download(
        repo_id   = repo_id,
        filename  = filename,
        repo_type = "dataset",
        cache_dir = cache_dir,
    )
    print(f"    ✅  Fichier disponible : {local_path}")
    return local_path


def load_checkpoint(path: str, device: torch.device) -> tuple:
    """Charge un checkpoint PyTorch et retourne (state_dict, meta)."""
    print(f"\n🔧  Chargement du checkpoint : {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}

        # Affiche la config d'entraînement si disponible
        train_cfg = meta.get("metadata", {}).get("training_history", {}).get("config", {})
        if train_cfg:
            print(f"    ℹ️   Config d'entraînement détectée :")
            print(f"         nvfp4={train_cfg.get('use_nvfp4')}  "
                  f"vocab={train_cfg.get('vocab_size')}  "
                  f"layers={train_cfg.get('num_layers')}  "
                  f"embed={train_cfg.get('embed_dim')}")

    elif isinstance(ckpt, dict) and any(k.startswith("token_embeddings") for k in ckpt):
        state_dict = ckpt
        meta = {}
    else:
        raise ValueError(
            "Format de checkpoint non reconnu. "
            "Attendu : dict avec 'model_state_dict' ou state_dict brut."
        )

    # ── Nettoyage des clés te.Linear (_extra_state) ──────────────────────────
    state_dict = clean_state_dict(state_dict)

    return state_dict, meta


def build_model(state_dict: dict, vocab_size: int, device: torch.device) -> HessGPT:
    """Instancie HessGPT (use_nvfp4=False) et charge les poids."""
    cfg = dict(DEFAULT_MODEL_CONFIG)
    cfg["vocab_size"]  = vocab_size
    cfg["use_nvfp4"]   = False  # inférence toujours en BF16/FP32

    print(f"\n🏗️   Construction du modèle (use_nvfp4=False — inférence standard) :")
    print(f"    embed={cfg['embed_dim']}  layers={cfg['num_layers']}  "
          f"heads={cfg['num_heads']}  kv={cfg['n_kv_heads']}  "
          f"vocab={cfg['vocab_size']}")

    model = HessGPT(**cfg)

    # Résolution vocab_size depuis le checkpoint si différent du tokenizer
    emb_key = "token_embeddings.weight"
    if emb_key in state_dict:
        ckpt_vocab = state_dict[emb_key].shape[0]
        if ckpt_vocab != vocab_size:
            print(f"    ⚠️   vocab_size checkpoint ({ckpt_vocab}) ≠ tokenizer ({vocab_size}). "
                  f"Resize → {ckpt_vocab}")
            model.resize_token_embeddings(ckpt_vocab)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    ⚠️   Clés manquantes ({len(missing)}) : {missing[:3]} ...")
    if unexpected:
        print(f"    ⚠️   Clés inconnues ({len(unexpected)}) : {unexpected[:3]} ...")

    model = model.to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"    ✅  Modèle chargé — {total / 1e6:.1f}M paramètres sur {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inférence
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    model          : HessGPT,
    tokenizer,
    prompt         : str,
    max_new_tokens : int   = 200,
    temperature    : float = 0.8,
    top_k          : int   = 50,
    top_p          : float = 0.95,
    device         : torch.device = torch.device("cpu"),
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p,
            eos_token_id   = tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Boucle interactive
# ─────────────────────────────────────────────────────────────────────────────

def interactive_loop(model: HessGPT, tokenizer, device: torch.device, args):
    print("\n" + "═" * 60)
    print("  HessGPT — Mode interactif")
    print("  Tapez votre prompt, puis Entrée.")
    print("  Commandes : :quit | :temp 0.8 | :topk 50 | :topp 0.95 | :max 200")
    print("═" * 60 + "\n")

    temperature    = args.temperature
    top_k          = args.top_k
    top_p          = args.top_p
    max_new_tokens = args.max_new_tokens

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not prompt:
            continue

        if prompt == ":quit":
            print("Au revoir !")
            break
        if prompt.startswith(":temp "):
            temperature = float(prompt.split()[1])
            print(f"  temperature → {temperature}")
            continue
        if prompt.startswith(":topk "):
            top_k = int(prompt.split()[1])
            print(f"  top_k → {top_k}")
            continue
        if prompt.startswith(":topp "):
            top_p = float(prompt.split()[1])
            print(f"  top_p → {top_p}")
            continue
        if prompt.startswith(":max "):
            max_new_tokens = int(prompt.split()[1])
            print(f"  max_new_tokens → {max_new_tokens}")
            continue

        print(f"\n[temp={temperature}  top_k={top_k}  top_p={top_p}  max={max_new_tokens}]")
        print("─" * 40)
        output = generate(
            model, tokenizer, prompt,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p,
            device         = device,
        )
        print(prompt + output)
        print("─" * 40 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inférence HessGPT v10")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", type=str, default=None,
                     help="Chemin local vers le fichier .pt")
    src.add_argument("--hf_repo", type=str, default=HF_REPO_DEFAULT,
                     help=f"Repo HuggingFace (défaut: {HF_REPO_DEFAULT})")

    p.add_argument("--hf_file",   type=str, default=HF_FILE_DEFAULT,
                   help=f"Nom du fichier dans le repo HF (défaut: {HF_FILE_DEFAULT})")
    p.add_argument("--cache_dir", type=str, default=".hf_cache",
                   help="Répertoire de cache local pour HF Hub")

    p.add_argument("--prompt",         type=str,  default=None,
                   help="Prompt unique (mode non-interactif)")
    p.add_argument("--max_new_tokens", type=int,   default=200)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--top_p",          type=float, default=0.95)

    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cpu', 'cuda', 'cuda:0', etc.")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🖥️   Device : {device}")

    # ── Tokenizer ────────────────────────────────────────────────
    print(f"\n📚  Chargement du tokenizer : {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"    ✅  vocab_size = {vocab_size}")

    # ── Checkpoint ───────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.exists(ckpt_path):
            print(f"❌  Fichier introuvable : {ckpt_path}")
            sys.exit(1)
    else:
        ckpt_path = download_from_hf(args.hf_repo, args.hf_file, args.cache_dir)

    state_dict, meta = load_checkpoint(ckpt_path, device)

    # ── Modèle ───────────────────────────────────────────────────
    model = build_model(state_dict, vocab_size, device)

    # Les poids du checkpoint sont en BF16 ; on caste tout le modèle
    # en BF16 pour garantir l'homogénéité dtype (poids + activations).
    model = model.to(torch.bfloat16)
    if device.type == "cuda":
        print("    ↳  Cast → bfloat16 (GPU)")
    else:
        print("    ↳  Cast → bfloat16 (CPU)")

    # ── Génération ───────────────────────────────────────────────
    if args.prompt:
        print(f"\n📝  Prompt : {args.prompt!r}")
        output = generate(
            model, tokenizer, args.prompt,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
            device         = device,
        )
        print("\n" + "─" * 40)
        print(args.prompt + output)
        print("─" * 40)
    else:
        interactive_loop(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
