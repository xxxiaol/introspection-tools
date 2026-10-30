"""Local Qwen3-8B with attention-head probes: token-level political bias readout and activation steering."""
import os
import pickle
from dataclasses import dataclass

import numpy as np
from einops import rearrange

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root, holds data/
MODEL = 'Qwen/Qwen3-8B'


@dataclass
class HF:
    tok: object
    model: object
    probes: dict   # {'heads': [(layer, head)], 'coef': {(l, h): vec}, 'std': {(l, h): float}}
    layers: dict   # layer -> [heads]
    steer: dict    # layer -> (head index tensor, [n_heads, d] steering vectors = coef * std)
    n_heads: int
    device: object


def load_hf(cache_dir=None, probes_path=f'{HERE}/data/probes_qwen3_8b.pkl'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=cache_dir, torch_dtype=torch.float16, device_map='auto')
    with open(probes_path, 'rb') as f:
        return build_hf(tok, model.eval(), pickle.load(f))


def build_hf(tok, model, probes):
    """Attach the probes to a Qwen3 model: group heads by layer and precompute steering vectors (coef * std)."""
    import torch
    tok.padding_side = 'left'
    layers = {}
    for l, h in probes['heads']:
        layers.setdefault(l, []).append(h)
    steer = {}
    for l, hs in layers.items():
        w = model.model.layers[l].self_attn.o_proj.weight
        vecs = torch.stack([torch.tensor(probes['coef'][(l, h)] * probes['std'][(l, h)], device=w.device, dtype=w.dtype) for h in hs])
        steer[l] = (torch.tensor(hs, device=w.device), vecs)
    return HF(tok, model, probes, layers, steer, model.config.num_attention_heads, model.model.layers[0].self_attn.o_proj.weight.device)


def _steering_hook(hf, l, alphas):
    heads, vecs = hf.steer[l]
    a = alphas.to(vecs.device, vecs.dtype)

    def hook(module, args):  # add alpha * (coef * std) to the selected heads' output at the current position
        x = rearrange(args[0], 'b s (h d) -> b s h d', h=hf.n_heads)
        x[:, -1, heads, :] += a[:, None, None] * vecs[None]
        return (rearrange(x, 'b s h d -> b s (h d)'),)
    return hook


def generate(hf, convs, max_new_tokens, continue_final=False, alphas=None, batch=32, temperature=1.0):
    """Sample one reply per conversation. `alphas` (one per conversation) turns steering on; `continue_final` extends a prefilled assistant turn."""
    import torch
    outs = []
    for s in range(0, len(convs), batch):
        enc = hf.tok.apply_chat_template(convs[s:s + batch], return_tensors='pt', padding=True, return_dict=True, enable_thinking=False,
                                         continue_final_message=continue_final, add_generation_prompt=not continue_final).to(hf.device)
        handles = []
        if alphas is not None:
            al = torch.tensor(alphas[s:s + batch], dtype=torch.float32)
            handles = [hf.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(_steering_hook(hf, l, al)) for l in hf.steer]
        try:
            with torch.no_grad():
                out = hf.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, pad_token_id=hf.tok.eos_token_id)
        finally:
            for h in handles:
                h.remove()
        outs += [hf.tok.decode(r, skip_special_tokens=True) for r in out[:, enc['input_ids'].shape[1]:]]
    return outs


def _ids(x):
    return x['input_ids'] if not hasattr(x, 'shape') else x


def probe_tokens(hf, messages, response_text):
    """Per-token political bias in [-1, 1] (-1 left, +1 right) of `response_text`, averaged over the probed heads.

    Returns ([(token, score)], mean score).
    """
    import torch
    tpl = hf.tok.apply_chat_template
    full = _ids(tpl(messages + [{'role': 'assistant', 'content': response_text}], return_tensors='pt', add_generation_prompt=False, enable_thinking=False))
    prompt_len = _ids(tpl(messages, return_tensors='pt', add_generation_prompt=True, enable_thinking=False)).shape[1]
    full = full.to(hf.device)
    store = {}

    def hook(l):
        def f(module, inp, out):
            x = rearrange(inp[0].detach(), 'b s (h d) -> b s h d', h=hf.n_heads)
            for h in hf.layers[l]:
                store[(l, h)] = x[0, :, h, :].float().cpu().numpy()
        return f

    handles = [hf.model.model.layers[l].self_attn.o_proj.register_forward_hook(hook(l)) for l in hf.steer]
    try:
        with torch.no_grad():
            hf.model(full, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    data = []
    for pos in range(prompt_len, full.shape[1]):
        pred = np.mean([np.dot(hf.probes['coef'][k], store[k][pos]) for k in store])
        data.append((hf.tok.decode([full[0, pos].item()]), float(np.clip(pred / 2, -1, 1))))
    return data, (float(np.mean([s for _, s in data])) if data else 0.0)
