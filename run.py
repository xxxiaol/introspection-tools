"""CLI for both experiments.  python run.py {calibration,misinfo} {generate,score,report} ...

  calibration generate --dataset olympiadbench --model Qwen3-8B --condition tool    --out out/ob_qwen_tool.jsonl
  calibration generate --dataset olympiadbench --model Qwen3-8B --condition control --out out/ob_qwen_control.jsonl
  calibration score  --in out/ob_qwen_tool.jsonl --out out/ob_qwen_tool.scored.jsonl     (same for control)
  calibration report --tool out/ob_qwen_tool.scored.jsonl --control out/ob_qwen_control.scored.jsonl

  misinfo generate --condition tool|control --out out/mis_tool.jsonl      (needs a GPU)
  misinfo score    --in out/mis_tool.jsonl --out out/mis_tool.scored.jsonl (calls the GPT-5.2 judge)
  misinfo report   --tool out/mis_tool.scored.jsonl --control out/mis_control.scored.jsonl
"""
import argparse
import importlib


def add_api_args(p):
    p.add_argument('--base-url', default=None, help='OpenAI-compatible endpoint (default: OpenRouter for gpt-*, localhost:8005 otherwise)')
    p.add_argument('--api-key-env', default=None, help='env var holding the API key (default: OPENROUTER_API_KEY)')


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tasks = ap.add_subparsers(dest='task', required=True)
    for task in ('calibration', 'misinfo'):
        stages = tasks.add_parser(task).add_subparsers(dest='stage', required=True)
        gen, sc, rep = stages.add_parser('generate'), stages.add_parser('score'), stages.add_parser('report')
        gen.add_argument('--condition', choices=['tool', 'control'], required=True)
        gen.add_argument('--out', required=True)
        gen.add_argument('--limit', type=int, default=0, help='only the first N problems/claims (smoke test)')
        sc.add_argument('--in', dest='inp', required=True)
        sc.add_argument('--out', required=True)
        rep.add_argument('--tool', required=True)
        rep.add_argument('--control', required=True)
        if task == 'calibration':
            gen.add_argument('--dataset', choices=['olympiadbench', 'theoremqa'], required=True)
            gen.add_argument('--model', required=True, help='gpt-4.1 (OpenRouter) or Qwen3-8B (local vLLM)')
            gen.add_argument('--workers', type=int, default=16)
            add_api_args(gen)
            rep.add_argument('--n-boot', type=int, default=10000)
        else:
            gen.add_argument('--batch', type=int, default=32)
            gen.add_argument('--cache-dir', default=None, help='Hugging Face cache for Qwen3-8B')
            sc.add_argument('--judge-model', default='openai/gpt-5.2')
            add_api_args(sc)
    return ap


def main():
    args = build_parser().parse_args()
    getattr(importlib.import_module(f'itools.{args.task}'), args.stage)(args)


if __name__ == '__main__':
    main()
