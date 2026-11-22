"""Conservative checks: tool execution is not proof of a model's interpretation."""
import re
import json
from decimal import Decimal, InvalidOperation


def numbers(text: str) -> set[Decimal]:
    values = set()
    for raw in re.findall(r'(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?', text):
        try:
            values.add(Decimal(raw.replace(',', '')))
        except InvalidOperation:
            pass
    return values


def check_answer(answer: str, evidence: list[dict], goal: str) -> tuple[str, str]:
    """Reject unsupported calculation figures; present pure calculations verbatim.

    Mixed and web synthesis is evidence-linked, not declared independently verified.
    This deliberately does not pretend numeric membership proves semantic agreement.
    """
    runs = [entry for entry in evidence if entry['tool'] == 'run_python']
    outputs = []
    for entry in runs:
        observation = entry['observation']
        if observation.get('result_truncated'):
            raise ValueError('Python result was truncated. Run a smaller calculation before answering.')
        stdout = str(observation.get('stdout') or '').strip()
        value = str(observation.get('result') or '').strip()
        output = '\n'.join(part for part in (stdout, value if value != 'None' else '') if part)
        if output:
            outputs.append(output)
    if runs and not outputs:
        raise ValueError('Python produced no visible result. Print or return the result before answering.')
    if runs:
        observed = '\n'.join(json.dumps(item['observation'], ensure_ascii=False) for item in evidence)
        if numbers(answer) - (numbers(observed) | numbers(goal)):
            raise ValueError('Answer adds figures absent from the cited tool results. Use the actual results.')
    if runs and len(runs) == len(evidence):
        supported = numbers('\n'.join(outputs)) | numbers(goal)
        if numbers(answer) - supported:
            raise ValueError('Answer contradicts or adds figures absent from the Python output. Use the actual result.')
        # No model paraphrase can invert labels, signs, booleans or units here.
        return 'Ran the numbers: ' + '\n\n'.join(outputs) + '.', 'tool_output'
    return answer, 'evidence_linked'
