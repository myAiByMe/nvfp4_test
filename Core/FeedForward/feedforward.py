# feedforward.py - v10 — NVfp4 via Transformer Engine
"""
NOUVEAUTÉS v10 :
  ✅ NVfp4 / FP8 via NVIDIA Transformer Engine
      - use_nvfp4=True  → gate/up/down_proj (SwiGLU) ou fc1/fc2 (GELU)
        remplacés par te.Linear, participent au contexte te.fp8_autocast
      - use_nvfp4=False → comportement identique v9 (nn.Linear, BF16)
      - Fallback silencieux si transformer_engine non installé

  Tout le reste identique v9 (SwiGLU, GELU fallback, dropout).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Transformer Engine — import optionnel ───────────────────────
try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) avec SwiGLU ou GELU.

    Args:
        embed_dim  : dimension des embeddings
        dropout    : taux de dropout
        use_swiglu : True = SwiGLU (défaut), False = GELU
        use_nvfp4  : True = te.Linear pour les GEMMs (NVfp4/FP8 via fp8_autocast)
    """
    def __init__(self, embed_dim, dropout=0.1, use_swiglu=True, use_nvfp4=False):
        super().__init__()

        self.embed_dim  = embed_dim
        self.use_swiglu = use_swiglu
        self.use_nvfp4  = use_nvfp4 and _TE_AVAILABLE

        # Choix de la classe Linear
        Linear = te.Linear if self.use_nvfp4 else nn.Linear

        if use_swiglu:
            # SwiGLU avec 8/3 * embed_dim (compensation gate)
            self.hidden_dim = int(8 * embed_dim / 3)
            self.hidden_dim = (self.hidden_dim + 63) // 64 * 64

            self.gate_proj = Linear(embed_dim, self.hidden_dim, bias=False)
            self.up_proj   = Linear(embed_dim, self.hidden_dim, bias=False)
            self.down_proj = Linear(self.hidden_dim, embed_dim, bias=False)
        else:
            # Fallback GELU
            self.hidden_dim = 4 * embed_dim
            self.fc1 = Linear(embed_dim, self.hidden_dim, bias=False)
            self.fc2 = Linear(self.hidden_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ── Cast BF16 — te.Linear (NVfp4) exige du bfloat16 en entrée ──
        if self.use_nvfp4 and x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        if self.use_swiglu:
            gate = F.silu(self.gate_proj(x))
            value = self.up_proj(x)
            x = gate * value
            x = self.down_proj(x)
            x = self.dropout(x)
        else:
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)
            x = self.dropout(x)
        return x
