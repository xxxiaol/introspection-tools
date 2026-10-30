# Introspection tools: proof-of-concept experiments

Code for the two proof-of-concept experiments in *Introspection Tools Help LLMs Understand and Control Themselves* (NeurIPS 2026 position paper).


## Layout
```
run.py           CLI
itools/
  core.py          Tool dataclass + run_episode (multi-turn tool-calling loop)
  tools.py         logprob tool; KEEP/STEER parsing, bias report, steer_perspective tool
  api_backend.py   OpenAI-compatible chat (OpenRouter / OpenAI / local vLLM) returning token logprobs
  hf_backend.py    Qwen3-8B + attention-head probes: token-level bias readout and activation steering
  calibration.py   experiment 1: generate / score / report
  misinfo.py       experiment 2: generate / score / report
  stats.py         ECE, paired bootstrap, exact McNemar (shared)
  math_judge.py, theoremqa_judge.py   answer-equivalence judges (vendored from OlympiadBench / TheoremQA)
data/            problem sets, PolitiFact claims, and the 32 trained probes (probes_qwen3_8b.pkl)
```

## Setup
```
pip install -r requirements.txt
```

## Run
Every task has three stages: `generate` (model calls) -> `score` (correctness / judge) -> `report` (tool vs. control). Outputs are jsonl.

```bash
# Calibration. Datasets: olympiadbench, theoremqa. Models: open-source model (needs a vLLM server on localhost:8005, or pass --base-url)
# and closed-source model (OpenRouter, OPENROUTER_API_KEY).
for d in olympiadbench theoremqa; do
  for c in tool control; do
    python run.py calibration generate --dataset $d --model Qwen3-8B --condition $c --out out/${d}_qwen_$c.jsonl
    python run.py calibration score --in out/${d}_qwen_$c.jsonl --out out/${d}_qwen_$c.scored.jsonl
  done
  python run.py calibration report --tool out/${d}_qwen_tool.scored.jsonl --control out/${d}_qwen_control.scored.jsonl
done

# Misinformation (GPU; HF_CACHE optional). The judge defaults to openai/gpt-5.2 on OpenRouter;
# for OpenAI directly: --judge-model gpt-5.2 --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
for c in tool control; do
  python run.py misinfo generate --condition $c --out out/mis_$c.jsonl --cache-dir $HF_CACHE
  python run.py misinfo score --in out/mis_$c.jsonl --out out/mis_$c.scored.jsonl
done
python run.py misinfo report --tool out/mis_tool.scored.jsonl --control out/mis_control.scored.jsonl
```
