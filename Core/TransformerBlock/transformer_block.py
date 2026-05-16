# transformer_block.py - v10 — NVfp4
"""
NOUVEAUTÉS v10 :
  ✅ Propagation de use_nvfp4 vers MultiHeadAttention et FeedForward
  ✅ Propagation de cu_seqlens / max_seqlen vers MultiHeadAttention
     (correction du chaînage manquant en v9)
  Tout le reste identique v9 (RMSNorm pre-norm, résidus, KV Cache).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from attention import MultiHeadAttention, RMSNorm, KVCache
from feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block : RMSNorm + RoPE + SwiGLU + GQA + Flash Attention
                        + KV Cache + NVfp4 (optionnel).

    Args:
      use_nvfp4 : True = te.Linear dans les GEMMs de l'attention et du FFN.
                  Ignoré silencieusement si transformer_engine non installé.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 use_swiglu=True, n_kv_heads=None, use_qk_norm=False,
                 use_flash_attn=True, soft_cap=None, use_nvfp4=False):
        super().__init__()

        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.use_rope       = use_rope
        self.use_swiglu     = use_swiglu
        self.n_kv_heads     = n_kv_heads
        self.use_qk_norm    = use_qk_norm
        self.use_flash_attn = use_flash_attn
        self.soft_cap       = soft_cap
        self.use_nvfp4      = use_nvfp4

        # RMSNorm : toujours BF16 (non remplacé par te.*)
        self.ln1 = RMSNorm(embed_dim)

        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout,
            use_rope              = use_rope,
            max_seq_len           = max_seq_len,
            use_yarn              = use_yarn,
            yarn_scale            = yarn_scale,
            yarn_original_max_len = yarn_original_max_len,
            n_kv_heads            = n_kv_heads,
            use_qk_norm           = use_qk_norm,
            use_flash_attn        = use_flash_attn,
            soft_cap              = soft_cap,
            use_nvfp4             = use_nvfp4,
        )

        self.ln2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout,
                               use_swiglu=use_swiglu,
                               use_nvfp4=use_nvfp4)

    def forward(
        self,
        x            : torch.Tensor,
        mask         : Optional[torch.Tensor] = None,
        past_kv      : Optional[KVCache]      = None,
        use_kv_cache : bool                   = False,
        cu_seqlens_q : Optional[torch.Tensor] = None,
        cu_seqlens_k : Optional[torch.Tensor] = None,
        max_seqlen_q : Optional[int]          = None,
        max_seqlen_k : Optional[int]          = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x            : [batch, seq_len, embed_dim]
            mask         : [seq_len, seq_len] bool — fallback manuel uniquement
            past_kv      : cache KV de ce layer depuis les steps précédents
            use_kv_cache : si True, retourne le cache KV mis à jour
            cu_seqlens_q : offsets séquences packées (Sequence Packing)
            cu_seqlens_k : idem pour K
            max_seqlen_q : longueur max dans le batch packé
            max_seqlen_k : idem pour K

        Returns:
            output : [batch, seq_len, embed_dim]
            new_kv : cache KV mis à jour, ou None
        """
        # Attention block (pre-norm)
        residual = x
        x, new_kv = self.attention(
            self.ln1(x),
            mask         = mask,
            past_kv      = past_kv,
            use_kv_cache = use_kv_cache,
            cu_seqlens_q = cu_seqlens_q,
            cu_seqlens_k = cu_seqlens_k,
            max_seqlen_q = max_seqlen_q,
            max_seqlen_k = max_seqlen_k,
        )
        x = residual + x

        # FFN block (pre-norm)
        residual = x
        x        = self.ffn(self.ln2(x))
        x        = residual + x

        return x, new_kv
