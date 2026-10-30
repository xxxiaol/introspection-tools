"""Experiment 1: uncertainty calibration. Does access to token logprobs reduce ECE? (tool vs. reflection-only control)"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from . import math_judge
from . import stats
from .api_backend import api_chat
from .core import run_episode
from .tools import logprob_tool

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root, holds data/
MAX_TURNS = 10

# dataset -> (role line, answer-field line in the JSON format)
FORMATS = {'olympiadbench': ('You are a math expert.', '"answer": "your final answer without any context"'),
           'theoremqa': ('You are an expert in scientific reasoning.', '"answer": your_numerical_or_textual_answer')}


def load_problems(dataset, limit=0):
    """OlympiadBench: 674 English math competition problems (OE_TO_maths_en_COMP). TheoremQA: the 747 text-only questions."""
    if dataset == 'olympiadbench':
        with open(f'{HERE}/data/olympiadbench.jsonl', encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        probs = [{'id': r['unique_id'], 'question': r['problem'], 'answer': r['answer']} for r in rows]
    else:
        with open(f'{HERE}/data/theoremqa_test.json', encoding='utf-8') as f:
            rows = json.load(f)
        probs = [{'id': r['id'].strip(), 'question': r['Question'].strip(), 'answer': r['Answer']}
                 for r in rows if r.get('Picture') in (None, 'null') and r.get('Question', '').strip() and r.get('Answer') is not None]
    return probs[:limit] if limit else probs


def build_prompt(dataset, question, condition, max_turns=MAX_TURNS):
    """`tool`: may call analyze_token_logprobs. `control`: may reflect for the same number of rounds, without the tool."""
    role, answer_field = FORMATS[dataset]
    if condition == 'tool':
        mid = (f"\nYou have access to tools and can use them for up to {max_turns} rounds of interaction.\n"
               "You can call the tool 'analyze_token_logprobs' after each step to check the uncertainty.")
    else:
        mid = f"\nYou can reflect on your reasoning process after each step. You should finish reasoning in up to {max_turns} rounds."
    return (f"{role} Please analyze the following question and provide a clear, step-by-step solution.\n{mid}\n\n"
            "After solving the problem, output your final response in the following JSON format:\n```json\n{\n"
            f'  {answer_field},\n  "calibration": your_likelihood_estimate_between_0_and_1\n}}\n```\n\nQuestion:\n{question}')


def generate(args):
    """Run every problem through the episode loop in a thread pool; append to a jsonl so interrupted runs resume."""
    chat = api_chat(args.model, args.base_url, args.api_key_env)
    tools = [logprob_tool()] if args.condition == 'tool' else []
    probs = load_problems(args.dataset, args.limit)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)['id'] for l in f if l.strip()}
    todo = [p for p in probs if p['id'] not in done]

    def solve(p):
        try:
            final, messages, status = run_episode(chat, build_prompt(args.dataset, p['question'], args.condition), tools, MAX_TURNS)
        except Exception as e:
            final, messages, status = None, [], f'error: {e}'
        return {'id': p['id'], 'dataset': args.dataset, 'condition': args.condition, 'gold': p['answer'], 'final': final, 'status': status,
                'n_tool_calls': sum(len(m.get('tool_calls', [])) for m in messages), 'messages': messages}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with ThreadPoolExecutor(args.workers) as ex, open(args.out, 'a') as f:
        for row in tqdm(ex.map(solve, todo), total=len(todo)):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()


def extract_json_answer(text):
    """Last ```json block (or last {...} span) containing an "answer" key, else None."""
    if not text:
        return None
    for s in [m for m in re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)[-1:]] + [text[text.rfind('{'):text.rfind('}') + 1]]:
        try:
            d = json.loads(s)
        except ValueError:
            continue
        if isinstance(d, dict) and 'answer' in d:
            return d
    return None


def score_row(row):
    """Adds `pred`, `confidence` (None if missing/invalid) and `correct` to a generated row."""
    d = extract_json_answer(row['final'])
    pred = str(d['answer']) if d and d.get('answer') not in (None, '') else None
    try:
        conf = float(d['calibration']) if d and d.get('calibration') is not None else None
    except (TypeError, ValueError):
        conf = None
    judge = math_judge.is_correct_olympiad if row['dataset'] == 'olympiadbench' else math_judge.is_correct_theoremqa
    return {**row, 'pred': pred, 'confidence': conf, 'correct': bool(pred is not None and judge(row['gold'], pred))}


def score(args):
    with open(args.inp) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with open(args.out, 'w') as f:
        for r in tqdm(rows):
            f.write(json.dumps(score_row(r), ensure_ascii=False) + '\n')


def load_scored(path):
    with open(path) as f:
        return {r['id']: r for r in map(json.loads, filter(str.strip, f))}


def summarize(rows):
    """Accuracy over all problems; ECE over the ones that gave a parsable answer and confidence (as in the paper)."""
    cal = [r for r in rows if r['pred'] is not None and r['confidence'] is not None]
    return {'n': len(rows), 'accuracy': stats.accuracy([r['correct'] for r in rows]),
            'n_calibrated': len(cal), 'ece': stats.ece([r['correct'] for r in cal], [r['confidence'] for r in cal]) if cal else None}


def report(args):
    tool, ctrl = load_scored(args.tool), load_scored(args.control)
    for name, rows in (('tool', tool), ('control', ctrl)):
        s = summarize(list(rows.values()))
        print(f"{name:8s} n={s['n']}  acc={s['accuracy']:.1%}  ECE={s['ece']:.1%} (n={s['n_calibrated']})")
    ids = sorted(set(tool) & set(ctrl), key=str)
    both = [i for i in ids if all(r[i]['pred'] is not None and r[i]['confidence'] is not None for r in (tool, ctrl))]
    c = stats.compare_calibration([tool[i]['correct'] for i in both], [tool[i]['confidence'] for i in both],
                                  [ctrl[i]['correct'] for i in both], [ctrl[i]['confidence'] for i in both], args.n_boot)
    b = c['ece']
    print(f"paired on {c['n']} problems, tool - control:  ECE {b['diff']:+.2%} [{b['ci_low']:+.2%}, {b['ci_high']:+.2%}] p={b['p']:.3f}")
    b = c['accuracy']
    print(f"{'':54s}acc {b['diff']:+.2%} [{b['ci_low']:+.2%}, {b['ci_high']:+.2%}] p={b['p']:.3f}; McNemar +{c['mcnemar'][0]}/-{c['mcnemar'][1]} p={c['mcnemar'][2]:.3f}")
