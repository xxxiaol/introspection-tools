"""OpenAI-compatible chat backend (OpenRouter, OpenAI, or a local vLLM server) that returns token logprobs."""
import json
import os

from .core import Reply

OPENROUTER = 'https://openrouter.ai/api/v1'
LOCAL_VLLM = 'http://localhost:8005/v1'


def client_for(model, base_url=None, api_key_env=None):
    """GPT models go through OpenRouter by default; anything else (e.g. Qwen3-8B) is expected on a local vLLM server."""
    from openai import OpenAI
    hosted = model.startswith(('gpt', 'openai/'))
    base_url = base_url or (OPENROUTER if hosted else LOCAL_VLLM)
    key = os.environ.get(api_key_env or 'OPENROUTER_API_KEY') if hosted or api_key_env else 'EMPTY'
    if not key:
        raise SystemExit(f'set {api_key_env or "OPENROUTER_API_KEY"}')
    return OpenAI(base_url=base_url, api_key=key)


def api_chat(model, base_url=None, api_key_env=None, temperature=0.7, max_tokens=4096):
    """Returns `chat(messages, tool_schemas) -> Reply`."""
    client = client_for(model, base_url, api_key_env)
    served = model
    if model.startswith('gpt'):
        served = 'openai/' + model
    elif not model.startswith('openai/') and base_url is None:
        served = client.models.list().data[0].id  # local vLLM serves a single model

    def chat(messages, tools):
        kw = dict(model=served, messages=messages, max_tokens=max_tokens, temperature=temperature, logprobs=True, top_logprobs=1)
        if tools:
            kw.update(tools=tools, tool_choice='auto')
        if 'qwen' in (model + served).lower():
            kw['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
        choice = client.chat.completions.create(**kw).choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except ValueError:
                args = {}
            calls.append((tc.id, tc.function.name, args))
        lp = [(t.token, t.logprob) for t in choice.logprobs.content] if choice.logprobs and choice.logprobs.content else None
        return Reply(choice.message.content, calls, lp)

    return chat
