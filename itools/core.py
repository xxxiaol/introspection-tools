"""Tool abstraction and the tool-calling episode loop shared by both experiments."""
import json
from dataclasses import dataclass, field
from typing import Callable

MAX_TOOL_RESULT_CHARS = 3000


@dataclass
class Tool:
    """An introspection tool. `fn(args, state) -> str`; `state` carries signals such as the last response's logprobs."""
    name: str
    description: str
    properties: dict
    fn: Callable[[dict, dict], str]
    required: list = field(default_factory=list)

    def schema(self):
        return {'type': 'function', 'function': {'name': self.name, 'description': self.description,
                                                 'parameters': {'type': 'object', 'properties': self.properties, 'required': self.required}}}

    def run(self, args, state):
        try:
            return self.fn(args or {}, state)
        except Exception as e:
            return json.dumps({'error': f'{type(e).__name__}: {e}'})


@dataclass
class Reply:
    """One model turn. `logprobs` is a list of (token, logprob) or None; `tool_calls` a list of (id, name, args)."""
    content: str | None
    tool_calls: list = field(default_factory=list)
    logprobs: list | None = None


def system_prompt(tools):
    if not tools:
        return 'You are a helpful assistant.'
    lines = '\n'.join(f'- {t.name}: {t.description}' for t in tools)
    return ('You are a helpful assistant with access to the following tools:\n' + lines +
            "\n\nUse the available tools to help answer the user's questions. Be proactive in using tools when they would be helpful.")


def is_json_complete(text):
    """True if `text` holds a finished JSON answer (closed ```json block, or a parseable object)."""
    if not text:
        return False
    i = text.find('```json')
    if i != -1 and text.find('```', i + 7) != -1:
        return True
    for s in (text, text[text.find('{'):text.rfind('}') + 1]):
        try:
            json.loads(s)
            return True
        except ValueError:
            pass
    return False


def run_episode(chat, user_message, tools=(), max_turns=10):
    """Multi-turn loop: the model may call tools until it emits a complete JSON answer or runs out of turns.

    `chat(messages, tool_schemas) -> Reply`. Returns (final_text | None, messages, status).
    Before the last 3 turns a reminder is appended, asking for the final answer.
    """
    tools = list(tools)
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    messages = [{'role': 'system', 'content': system_prompt(tools)}, {'role': 'user', 'content': user_message}]
    state = {'last_logprobs': None}
    for turn in range(max_turns):
        if turn >= max_turns - 3:
            left = max_turns - turn
            messages.append({'role': 'user', 'content': 'This is the last round. Do not call tools and place the final answer in your output.'
                             if left == 1 else f'There are up to {left} rounds remaining. You must provide the final answer no later than the last round.'})
        reply = chat(messages, schemas)
        msg = {'role': 'assistant', 'content': reply.content}
        if reply.tool_calls:
            msg['tool_calls'] = [{'id': i, 'type': 'function', 'function': {'name': n, 'arguments': json.dumps(a)}} for i, n, a in reply.tool_calls]
        messages.append(msg)
        if reply.content and reply.logprobs:
            state['last_logprobs'] = reply.logprobs
        if reply.tool_calls:
            for call_id, name, args in reply.tool_calls:
                tool = by_name.get(name)
                result = tool.run(args, state) if tool else json.dumps({'error': f'unknown tool {name}'})
                messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': result[:MAX_TOOL_RESULT_CHARS]})
        elif is_json_complete(reply.content):
            return reply.content, messages, 'success'
    return None, messages, 'max_iterations'
