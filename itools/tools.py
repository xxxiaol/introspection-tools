"""The two introspection tools: a sensing tool (token logprobs) and a steering tool (activation steering)."""
import json
import re

import numpy as np

from .core import Tool

CONTEXT_WINDOW = 5
STEER_SCALE = 50      # beta in the paper: largest value that keeps generation fluent
STEER_CLIP = 30       # |alpha * beta| is clipped to this


# ---------------------------------------------------------------- logprob tool (calibration)

def select_span(tokens, text_part):
    """Tokens covering the first occurrence of `text_part` (case-insensitive fallback); all tokens if not found."""
    if not text_part:
        return tokens
    full = ''.join(t for t, _ in tokens)
    start = full.find(text_part)
    if start == -1:
        start = full.lower().find(text_part.lower())
    if start == -1:
        return tokens
    end, pos, first, last = start + len(text_part), 0, None, None
    for i, (t, _) in enumerate(tokens):
        if first is None and pos + len(t) > start:
            first = i
        if pos + len(t) >= end:
            last = i + 1
            break
        pos += len(t)
    return tokens[first:last] if first is not None and last is not None else tokens


def summarize_logprobs(tokens):
    """Uncertainty summary of a token span: size, mean and min logprob, and the least confident token in context."""
    if not tokens:
        raise ValueError('empty token list')
    pos = min(range(len(tokens)), key=lambda i: tokens[i][1])
    before = ''.join(t for t, _ in tokens[max(0, pos - CONTEXT_WINDOW):pos])
    after = ''.join(t for t, _ in tokens[pos + 1:pos + CONTEXT_WINDOW + 1])
    return {'total_tokens': len(tokens),
            'average_logprob': round(sum(lp for _, lp in tokens) / len(tokens), 4),
            'minimum_logprob': round(tokens[pos][1], 4),
            'minimum_token': tokens[pos][0],
            'context': f'...{before}[{tokens[pos][0]}]{after}...'}


def _analyze_token_logprobs(args, state):
    tokens = state.get('last_logprobs')
    if not tokens:
        return json.dumps({'error': 'No log probabilities available from the previous model response.'}, indent=2)
    return json.dumps(summarize_logprobs(select_span(tokens, args.get('text_part'))), indent=2)


LOGPROB_DESCRIPTION = """Analyze token log probabilities to quantify model output uncertainty.

This tool analyzes log probabilities from the most recent model response. You can analyze either the entire response or a specific part of it.
The log probabilities are injected automatically.

Returns JSON with the following fields:
- total_tokens: Number of tokens analyzed
- average_logprob: Overall confidence across all tokens
- minimum_logprob: The lowest log probability (most uncertain token)
- minimum_token: Which token was least confident
- context: Surrounding text showing the uncertain token in brackets

Interpretation guide:
- Higher (closer to 0) = more confident. Lower (more negative) = less confident.
- Around 0 to -0.5: High confidence
- Around -1 to -2: Moderate confidence
- Below -3: Low confidence

To use this tool, you should first generate the text in one response, and then call the tool in the next response."""


def logprob_tool():
    return Tool('analyze_token_logprobs', LOGPROB_DESCRIPTION,
                {'text_part': {'type': 'string', 'description': 'A substring from your response to analyze. Default: full response'}},
                _analyze_token_logprobs)


# ---------------------------------------------------------------- perspective tools (misinformation)

def bias_table(data, max_display=50):
    """Token-by-token political bias report shown to the model (data: list of (token, score in [-1, 1]))."""
    lines = ['=' * 70, 'TOKEN-BY-TOKEN BIAS ANALYSIS', '=' * 70, f"{'Token':<30} | {'Bias Score':>12} | {'Direction':>12}", '-' * 70]
    rows = data if len(data) <= max_display else data[:max_display // 2] + [('...', 0.0)] + data[-(max_display // 2):]
    for tok, s in rows:
        d = tok.replace('\n', '\\n').replace('\t', '\\t')
        d = d[:25] + '...' if len(d) > 28 else d
        direction = '← Dem' if s < -0.001 else ('Rep →' if s > 0.1 else 'Neutral')
        color = '\033[94m' if s < -0.3 else ('\033[91m' if s > 0.3 else '\033[0m')
        lines.append(f'{d:<30} | {color}{s:>+12.3f}\033[0m | {direction:>12}')
    lines.append('-' * 70)
    scores = [s for t, s in data if t != '...']
    if scores:
        a = float(np.mean(scores))
        lines.append(f'SUMMARY: Avg={a:+.3f}, Std={np.std(scores):.3f}, Min={np.min(scores):+.3f}, Max={np.max(scores):+.3f}')
        lines.append(f"Overall tendency: {'Democrat-leaning' if a < -0.1 else ('Republican-leaning' if a > 0.1 else 'Neutral')}")
    lines.append('=' * 70)
    return '\n'.join(lines)


def parse_decision(text):
    """Parse the model's KEEP/STEER reply -> (decision | None, steer_magnitude | None). Tolerant of ```json fences and <think>."""
    t = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    m = re.search(r'```json\s*(\{.*?\})\s*```', t, re.DOTALL) or re.search(r'\{[^{}]*\}', t, re.DOTALL)
    dec, mag = None, None
    if m:
        try:
            d = json.loads(re.sub(r':\s*\+(\d)', r': \1', m.group(1) if '```' in m.group(0) else m.group(0)))
            v = str(d.get('decision', '')).upper()
            dec = 'STEER' if 'STEER' in v else ('KEEP' if 'KEEP' in v else None)
            if d.get('steer_magnitude') is not None:
                mag = float(str(d['steer_magnitude']).replace('+', ''))
        except Exception:
            pass
    if dec is None:
        dec = 'STEER' if 'STEER' in t.upper() else ('KEEP' if 'KEEP' in t.upper() else None)
    if dec == 'STEER' and mag is None:
        m2 = re.search(r'magnitude[:\s]*([+-]?\d*\.?\d+)', t, re.IGNORECASE)
        if m2:
            mag = float(m2.group(1))
    return dec, mag


def steering_alpha(value):
    """Model-chosen steering value in [-1, 1] -> activation-addition coefficient (alpha * beta, clipped)."""
    return float(np.clip((value or 0.0) * STEER_SCALE, -STEER_CLIP, STEER_CLIP))


def steer_perspective_tool(steered_generate):
    """`steered_generate(prompt, alpha) -> str` regenerates a response from a clean context with activation steering."""
    def fn(args, state):
        return steered_generate(args['prompt'], steering_alpha(float(args['steering_value'])))
    return Tool('steer_perspective', "Steer the model's internal representations along the ideological dimension and regenerate a response.",
                {'prompt': {'type': 'string', 'description': 'The original user prompt to regenerate a response for'},
                 'steering_value': {'type': 'number', 'description': 'Steering direction and strength from -1.0 (left) to +1.0 (right)'}},
                fn, required=['prompt', 'steering_value'])
