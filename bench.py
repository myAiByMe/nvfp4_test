#!/usr/bin/env python3
"""
HessGPT — Evaluation Qwen2.5 (22 benchmarks, shots officiels du tech report)
Source : https://arxiv.org/abs/2412.15115

USAGE :
    python bench.py --checkpoint ./Model/HessGpt_pretrain.pt
    python bench.py --checkpoint ./Model/HessGpt_pretrain.pt --output results.json
    python bench.py --checkpoint ./Model/HessGpt_pretrain.pt --limit 0.1   # debug 10%%

INSTALLATION :
    pip install lm-eval
"""

import sys
import os
import json
import time
import argparse
import traceback
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn.functional as F

sys.path.append('./Core/Model')
sys.path.append('./Core/Attention')
sys.path.append('./Core/FeedForward')
sys.path.append('./Core/TransformerBlock')

# ── 22 benchmarks Qwen2.5 avec leurs shots officiels ─────────────
# Chaque entrée : (nom_lm_eval, num_fewshot, métrique_principale)
QWEN_TASKS = [
    # Commonsense
    ('hellaswag',       10, 'acc_norm'),
    ('arc_easy',        25, 'acc_norm'),
    ('arc_challenge',   25, 'acc_norm'),
    ('winogrande',       5, 'acc'),
    ('piqa',             0, 'acc_norm'),
    ('boolq',            0, 'acc'),
    ('openbookqa',       0, 'acc_norm'),
    ('commonsenseqa',    7, 'acc'),
    # Connaissances / QA
    ('triviaqa',         5, 'exact_match'),
    ('nq_open',          5, 'exact_match'),
    ('truthfulqa_mc2',   0, 'acc'),
    # Compréhension
    ('lambada_openai',   0, 'acc'),
    ('race',             0, 'acc'),
    ('drop',             3, 'f1'),
    # Raisonnement
    ('bbh_cot_fewshot',  3, 'acc_norm'),
    # Mathématiques
    ('gsm8k',            8, 'exact_match'),
    ('math',             4, 'exact_match'),
    # Code
    ('humaneval',        0, 'pass@1'),
    ('mbpp',             3, 'pass@1'),
    # Connaissances encyclopédiques
    ('mmlu',             5, 'acc'),
    ('mmlu_pro',         5, 'acc'),
    # Science avancée
    ('gpqa_main',        0, 'acc_norm'),
]

TASKS       = [t[0] for t in QWEN_TASKS]
FEWSHOT_MAP = {t[0]: t[1] for t in QWEN_TASKS}
METRIC_MAP  = {t[0]: t[2] for t in QWEN_TASKS}


# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser(description='HessGPT — Qwen2.5 Benchmark Suite')
parser.add_argument('--checkpoint', type=str,
                    default='./Model/HessGpt_pretrain.pt',
                    help='Chemin vers le checkpoint .pt')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size pour l\'évaluation')
parser.add_argument('--max-seq-len', type=int, default=None,
                    help='Override max_seq_len du modèle')
parser.add_argument('--output', type=str, default=None,
                    help='Fichier JSON pour sauvegarder les résultats')
parser.add_argument('--device', type=str, default=None,
                    help='Device (cuda / cpu). Détection auto si absent.')
parser.add_argument('--dtype', type=str, default='bfloat16',
                    choices=['bfloat16', 'float16', 'float32'],
                    help='Dtype d\'inférence')
parser.add_argument('--limit', type=float, default=None,
                    help='Fraction d\'exemples par tâche (debug, ex: 0.1)')
args = parser.parse_args()

_device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
_dtype  = {'bfloat16': torch.bfloat16,
            'float16':  torch.float16,
            'float32':  torch.float32}[args.dtype]

print('=' * 80)
print('HessGPT — Qwen2.5 Benchmark Suite (22 tâches)')
print('=' * 80)
print(f'  Checkpoint : {args.checkpoint}')
print(f'  Tâches     : {len(TASKS)}  (shots Qwen2.5 officiels)')
print(f'  Batch size : {args.batch_size}')
print(f'  Device     : {_device}  |  Dtype : {args.dtype}')
if args.limit:
    print(f'  Limit      : {args.limit}')
print()
for name, shots, metric in QWEN_TASKS:
    print(f'    {name:<22} {shots:>2}-shot  [{metric}]')


# ============================================================
# CHARGEMENT MODÈLE
# ============================================================
print(f'\n{"="*60}\nCHARGEMENT MODÈLE\n{"="*60}')

try:
    from HessGpt import HessGPT
except ImportError as e:
    print(f'ERREUR import HessGPT : {e}')
    sys.exit(1)

if not os.path.exists(args.checkpoint):
    print(f'ERREUR : checkpoint introuvable → {args.checkpoint}')
    sys.exit(1)

checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

# Config depuis checkpoint _info.json si dispo
ckpt_config = None
info_path = args.checkpoint.replace('.pt', '_info.json')
if os.path.exists(info_path):
    with open(info_path) as f:
        ckpt_config = json.load(f).get('config', None)
    print(f'  Config chargée depuis : {info_path}')

DEFAULT_CONFIG = {
    'vocab_size':            None,
    'embed_dim':             1280,
    'num_heads':             20,
    'num_layers':            24,
    'max_seq_len':           512,
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
}
cfg = {**DEFAULT_CONFIG, **(ckpt_config or {})}
if args.max_seq_len:
    cfg['max_seq_len'] = args.max_seq_len

from transformers import AutoTokenizer
print('  Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/cosmo2-tokenizer')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
cfg['vocab_size'] = len(tokenizer)
print(f'  vocab={len(tokenizer)}')

model = HessGPT(
    vocab_size            = cfg['vocab_size'],
    embed_dim             = cfg['embed_dim'],
    num_heads             = cfg['num_heads'],
    num_layers            = cfg['num_layers'],
    max_seq_len           = cfg['max_seq_len'],
    dropout               = 0.0,
    use_rope              = cfg['use_rope'],
    use_yarn              = cfg.get('use_yarn', False),
    yarn_scale            = cfg.get('yarn_scale', 1.0),
    yarn_original_max_len = cfg.get('yarn_original_max_len', cfg['max_seq_len']),
    use_swiglu            = cfg['use_swiglu'],
    n_kv_heads            = cfg['n_kv_heads'],
    use_qk_norm           = cfg.get('use_qk_norm', False),
    soft_cap              = cfg.get('soft_cap', None),
    use_flash_attn        = cfg['use_flash_attn'],
    use_nvfp4             = False,
)

state_dict = checkpoint.get('model_state_dict', checkpoint)
model.load_state_dict(state_dict, strict=True)
model = model.to(_device).to(_dtype)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f'  Params : {total_params / 1e6:.1f}M  |  max_seq_len : {cfg["max_seq_len"]}')
print('  Modèle prêt')


# ============================================================
# WRAPPER LM EVAL
# ============================================================
try:
    from lm_eval.api.model import LM
    import lm_eval
except ImportError:
    print('\nERREUR : lm-eval non installé.')
    print('  pip install lm-eval')
    sys.exit(1)


class HessGPTLM(LM):
    """Wrapper LM Eval Harness pour HessGPT."""

    def __init__(self, model, tokenizer, device, dtype, batch_size, max_seq_len):
        super().__init__()
        self._model      = model
        self._tokenizer  = tokenizer
        self._device     = device
        self._dtype      = dtype
        self._batch_size = batch_size
        self._max_length = max_seq_len

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str) -> List[int]:
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)

    def _encode_pair(self, context: str, continuation: str):
        ctx_ids  = self.tok_encode(context)
        cont_ids = self.tok_encode(continuation)
        max_ctx  = max(self._max_length - len(cont_ids) - 1, 1)
        ctx_ids  = ctx_ids[-max_ctx:]
        return ctx_ids + cont_ids, len(ctx_ids)

    @torch.no_grad()
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        results = []
        batches = [list(requests[i:i + self._batch_size])
                   for i in range(0, len(requests), self._batch_size)]

        for batch in batches:
            encoded = []
            for req in batch:
                ctx, cont = req.args
                ids, ctx_len = self._encode_pair(ctx, cont)
                encoded.append((ids, ctx_len, len(self.tok_encode(cont))))

            max_len      = max(len(e[0]) for e in encoded)
            pad_id       = self._tokenizer.pad_token_id
            input_tensor = torch.full(
                (len(encoded), max_len), pad_id, dtype=torch.long, device=self._device
            )
            for i, (ids, _, _) in enumerate(encoded):
                input_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            with torch.amp.autocast(self._device, dtype=self._dtype,
                                    enabled=(self._device == 'cuda')):
                logits, _, _ = self._model(input_tensor)

            log_probs = F.log_softmax(logits, dim=-1)

            for i, (ids, ctx_len, cont_len) in enumerate(encoded):
                cont_start = ctx_len
                cont_end   = min(ctx_len + cont_len, max_len)
                target_ids = torch.tensor(
                    ids[cont_start:cont_end], dtype=torch.long, device=self._device
                )
                logit_slice = log_probs[i, cont_start - 1:cont_end - 1]

                if logit_slice.shape[0] == 0 or target_ids.shape[0] == 0:
                    results.append((float('-inf'), False))
                    continue

                n         = min(logit_slice.shape[0], target_ids.shape[0])
                token_ll  = logit_slice[:n].gather(1, target_ids[:n].unsqueeze(1)).squeeze(1)
                total_ll  = token_ll.sum().item()
                is_greedy = (logit_slice[:n].argmax(dim=-1) == target_ids[:n]).all().item()
                results.append((total_ll, bool(is_greedy)))

        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests) -> List[float]:
        results = []
        for req in requests:
            ids      = self.tok_encode(req.args[0])
            total_ll = 0.0
            for start in range(0, max(len(ids) - 1, 1), self._max_length - 1):
                chunk = ids[start:start + self._max_length]
                if len(chunk) < 2:
                    continue
                inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=self._device)
                tgt = torch.tensor(chunk[1:],   dtype=torch.long, device=self._device)
                with torch.amp.autocast(self._device, dtype=self._dtype,
                                        enabled=(self._device == 'cuda')):
                    logits, _, _ = self._model(inp)
                lp        = F.log_softmax(logits[0], dim=-1)
                total_ll += lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item()
            results.append(total_ll)
        return results

    @torch.no_grad()
    def generate_until(self, requests) -> List[str]:
        results = []
        for req in requests:
            ctx, gen_kwargs = req.args
            ids       = self.tok_encode(ctx)
            max_ctx   = max(self._max_length - self.max_gen_toks, 1)
            ids       = ids[-max_ctx:]
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)

            until    = gen_kwargs.get('until', [])
            max_toks = gen_kwargs.get('max_gen_toks', self.max_gen_toks)
            temp     = gen_kwargs.get('temperature', 0.0)
            top_p    = gen_kwargs.get('top_p', None)

            with torch.amp.autocast(self._device, dtype=self._dtype,
                                    enabled=(self._device == 'cuda')):
                out_ids = self._model.generate(
                    input_ids,
                    max_new_tokens = max_toks,
                    temperature    = temp if temp > 0 else 0.0,
                    top_p          = top_p,
                    eos_token_id   = self._tokenizer.eos_token_id,
                )

            gen_text = self._tokenizer.decode(
                out_ids[0, len(ids):].tolist(), skip_special_tokens=True
            )
            for stop in until:
                if stop in gen_text:
                    gen_text = gen_text[:gen_text.index(stop)]
            results.append(gen_text)
        return results


# ============================================================
# ÉVALUATION
# ============================================================
print(f'\n{"="*60}\nÉVALUATION\n{"="*60}')

lm = HessGPTLM(
    model       = model,
    tokenizer   = tokenizer,
    device      = _device,
    dtype       = _dtype,
    batch_size  = args.batch_size,
    max_seq_len = cfg['max_seq_len'],
)

t0 = time.time()
try:
    results = lm_eval.simple_evaluate(
        model       = lm,
        tasks       = TASKS,
        num_fewshot = FEWSHOT_MAP,
        batch_size  = args.batch_size,
        limit       = args.limit,
        log_samples = False,
    )
except Exception:
    print(f'\nERREUR pendant l\'évaluation :\n{traceback.format_exc()}')
    sys.exit(1)

elapsed = time.time() - t0


# ============================================================
# AFFICHAGE RÉSULTATS
# ============================================================
print(f'\n{"="*80}')
print(f'RÉSULTATS  ({elapsed/60:.1f} min)')
print(f'{"="*80}')
print(f'{"Tâche":<24} {"Shot":>4}  {"Metric":<16} {"Score":>8}')
print(f'{"─"*60}')

summary = {}
for task_name, shots, preferred_metric in QWEN_TASKS:
    task_res = results['results'].get(task_name)
    if task_res is None:
        print(f'{task_name:<24} {shots:>4}  {"—":<16} {"—":>8}')
        continue

    # Cherche la métrique préférée, puis fallback sur les métriques standards
    score = None
    metric_used = None
    for key in (f'{preferred_metric},none',
                'acc_norm,none', 'acc,none', 'exact_match,none',
                'f1,none', 'pass@1,none', 'perplexity,none'):
        if key in task_res and isinstance(task_res[key], float):
            score       = task_res[key]
            metric_used = key.split(',')[0]
            break

    if score is None:
        print(f'{task_name:<24} {shots:>4}  {"(no metric)":<16} {"—":>8}')
        continue

    stderr_key = f'{metric_used}_stderr,none'
    stderr     = task_res.get(stderr_key)
    score_str  = f'{score * 100:.2f}%' if score <= 1.0 else f'{score:.3f}'
    stderr_str = f' ±{stderr*100:.2f}' if stderr and stderr <= 1.0 else ''
    print(f'{task_name:<24} {shots:>4}  {metric_used:<16} {score_str:>8}{stderr_str}')
    summary[task_name] = {'metric': metric_used, 'score': score,
                          'stderr': stderr, 'num_fewshot': shots}

print(f'{"─"*60}')

acc_scores = [
    v['score'] for v in summary.values()
    if v['metric'] in ('acc', 'acc_norm') and v['score'] is not None
]
if acc_scores:
    print(f'\n  Moyenne acc/acc_norm : {sum(acc_scores)/len(acc_scores)*100:.2f}%'
          f'  ({len(acc_scores)} tâches)')


# ============================================================
# SAUVEGARDE
# ============================================================
output_data = {
    'timestamp':      datetime.now().isoformat(),
    'checkpoint':     args.checkpoint,
    'suite':          'Qwen2.5 (22 tâches)',
    'dtype':          args.dtype,
    'elapsed_min':    round(elapsed / 60, 2),
    'model_params_M': round(total_params / 1e6, 1),
    'config':         cfg,
    'summary':        summary,
    'full_results':   results['results'],
}

if args.output:
    out_path = args.output
else:
    base     = os.path.splitext(os.path.basename(args.checkpoint))[0]
    ts       = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = f'./results_{base}_{ts}.json'

with open(out_path, 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

print(f'\n  Résultats sauvegardés → {out_path}')
print('DONE')
