# HessGPT

Transformer decoder-only from scratch, style LLaMA/Mistral. Pré-entraînement uniquement.

---

## Architecture

| Composant | Choix | Pourquoi |
|---|---|---|
| Normalisation | RMSNorm | Plus rapide que LayerNorm (pas de centrage) |
| Encodage positionnel | RoPE + YaRN | Extension de contexte sans réentraînement |
| Activation FFN | SwiGLU | ~15% mieux que GELU à iso-paramètres |
| Attention | GQA + FlashAttention | Mémoire KV réduite + vitesse |
| Stabilité | QK-Norm (optionnel) | Entraînement stable à fort LR |
| Inférence | KV Cache | Génération autoregressive rapide |

**Configuration par défaut :**
```
embed_dim=1280  num_heads=20  num_layers=24
n_kv_heads=5    max_seq_len=512
tokenizer: cosmo2 (~128k vocab)
soft_cap=None   use_flash_attn=True
```

> **Note soft_cap** : toujours `None`. Toute valeur non-nulle désactive FlashAttention.

---

## Structure du projet

```
test_fp4_TE/
├── Core/
│   ├── Attention/
│   │   └── attention.py          # RMSNorm, RoPE/YaRN, MultiHeadAttention
│   ├── FeedForward/
│   │   └── feedforward.py        # FFN avec SwiGLU ou GELU
│   ├── TransformerBlock/
│   │   └── transformer_block.py  # RMSNorm + MHA + FFN + résiduel
│   └── Model/
│       └── HessGpt.py            # Modèle complet + génération
│
├── pretrain.py                   # Pré-entraînement (NVfp4 + Sequence Packing)
├── bench.py                      # Évaluation Qwen2.5 (22 benchmarks)
├── speedTest.py                  # Diagnostic vitesse complet (10 tests)
└── requi.txt                     # Dépendances Python
```

---

## Installation

```bash
pip install -r requi.txt
```

**Optionnel (accélération) :**
```bash
# FlashAttention (FA2/FA3/FA4 selon le GPU)
pip install flash-attn --no-build-isolation

# NVfp4 — Blackwell uniquement (SM120)
pip install transformer-engine[pytorch]

# Évaluation benchmarks
pip install lm-eval
```

**Compatibilité GPU :**
| GPU | FlashAttention | NVfp4 |
|---|---|---|
| B200 (SM120) | FA4 | ✅ natif |
| H100 (SM90) | FA3 | FP8 fallback |
| A100 (SM80) | FA2 | — |
| Autres | SDPA PyTorch | — |

---

## Pré-entraînement

```bash
python pretrain.py
python pretrain.py --batch-size 64   # override batch size via CLI
```

**Ce que fait le script :**
- Charge `./pretrain_data.bin` (tokens uint16 flat binary)
- Tokenizer : cosmo2 (`HuggingFaceTB/cosmo2-tokenizer`)
- Sequence Packing activé (`use_packing=True`) — zéro padding, 100% tokens utiles
- Optimiseur : **Muon + MARS** pour les poids 2D, **AdamW fused** pour le reste
- Scheduler : **WSD** (Warmup → Stable → Decay cosine)
- **NVfp4 Healing** : à 90% des steps, retour en BF16 pour recovery de précision
- Benchmark tokens/sec + MFU au démarrage (Phase 0 sans compile, Phase 1 avec)
- Checkpoint automatique toutes les 2000 steps → `./Model/HessGpt_pretrain.pt`
- Reprise automatique : relancer la même commande reprend là où c'était arrêté

**Format des données :**
```
pretrain_data.bin   ← flat binary, tokens uint16 concaténés
```

Pour générer ce fichier depuis un dataset tokenisé :
```python
import numpy as np
tokens = np.array(all_token_ids, dtype=np.uint16)
tokens.tofile('./pretrain_data.bin')
```

**Principaux hyperparamètres :**
```python
CONFIG = {
    'learning_rate':         4e-4,
    'batch_size':            140,
    'gradient_accumulation': 8,       # batch effectif = 1120
    'num_epochs':            5,
    'max_seq_len':           512,
    'warmup_ratio':          0.03,
    'decay_ratio':           0.15,
    'use_packing':           True,
    'use_compile':           True,    # torch.compile mode='default'
    'use_nvfp4':             True,    # NVfp4 si Blackwell + TE installé
    'healing_ratio':         0.90,    # retour BF16 à 90% des steps
}
```

---

## Évaluation

```bash
# Suite Qwen2.5 complète (22 benchmarks, shots officiels)
python bench.py --checkpoint ./Model/HessGpt_pretrain.pt

# Debug rapide (10% des exemples)
python bench.py --checkpoint ./Model/HessGpt_pretrain.pt --limit 0.1

# Sauvegarder les résultats
python bench.py --checkpoint ./Model/HessGpt_pretrain.pt --output results.json
```

**22 benchmarks Qwen2.5 :**

| Catégorie | Benchmarks |
|---|---|
| Commonsense | HellaSwag (10-shot), ARC-E/C (25-shot), WinoGrande (5-shot), PIQA, BoolQ, OpenBookQA, CommonsenseQA (7-shot) |
| Connaissances | TriviaQA (5-shot), NQ-Open (5-shot), TruthfulQA |
| Compréhension | LAMBADA, RACE, DROP (3-shot) |
| Raisonnement | BBH-CoT (3-shot) |
| Mathématiques | GSM8K (8-shot), MATH (4-shot) |
| Code | HumanEval, MBPP (3-shot) |
| Encyclopédique | MMLU (5-shot), MMLU-Pro (5-shot) |
| Science avancée | GPQA-Main |

---

## Diagnostic vitesse

```bash
python speedTest.py
python speedTest.py --seq-len 512 --batch-size 28
python speedTest.py --no-compile
```

**10 tests effectués :**
1. GPU info + capacité théorique (TFLOPs, params estimés)
2. FlashAttention disponibilité (SDPA, FA2/3/4, flex_attention)
3. Attention : Flash vs Manuel vs Flex
4. Forward seul
5. Forward + Backward
6. Forward + Backward + Optimizer step (Muon + AdamW)
7. Muon Newton-Schulz : ns_steps=3 vs 5
8. DataLoader : temps de chargement / prefetch
9. torch.compile : gain réel mesuré
10. Résumé + recommandations automatiques

---

## Utilisation du modèle

```python
import sys
sys.path.extend(['./Core/Model', './Core/Attention',
                 './Core/FeedForward', './Core/TransformerBlock'])

from HessGpt import HessGPT

model = HessGPT(
    vocab_size     = 128000,   # cosmo2 tokenizer
    embed_dim      = 1280,
    num_heads      = 20,
    num_layers     = 24,
    max_seq_len    = 512,
    use_rope       = True,
    use_swiglu     = True,
    n_kv_heads     = 5,
    use_qk_norm    = True,
    soft_cap       = None,     # toujours None
    use_flash_attn = True,
)

# Training
logits, loss, _ = model(input_ids, targets=labels, pad_token_id=pad_id)
loss.backward()

# Génération avec KV Cache
output = model.generate(
    input_ids,
    max_new_tokens = 200,
    temperature    = 0.8,
    top_p          = 0.9,
    eos_token_id   = tokenizer.eos_token_id,
)
```

---

## KV Cache

**Prefill** (prompt complet) :
```python
logits, _, past_kv = model(prompt_ids, use_kv_cache=True)
next_token = logits[:, -1, :].argmax(-1)
```

**Decode** (1 token à la fois) :
```python
logits, _, past_kv = model(next_token, past_kv=past_kv, use_kv_cache=True)
next_token = logits[:, -1, :].argmax(-1)
```

---

## Extension de contexte (YaRN)

```python
model = HessGPT(
    max_seq_len           = 2048,
    use_yarn              = True,
    yarn_scale            = 4.0,
    yarn_original_max_len = 512,
)
```
