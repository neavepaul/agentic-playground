import pytest
from app.capabilities.tools import CapabilityTools
from app.capabilities.schemas import AnswerAction
from app.tasks.models import TaskContext
from app.tasks.manager import answer_notes


def calculation(output='42', **extra):
    context = TaskContext(goal='Calculate 17 + 25')
    context.record('run_python', {}, {'success': True, 'evidence_id': 'run',
                   'observation': {'stdout': output, 'result': 'None', **extra}})
    return context


def finish(context, answer):
    return CapabilityTools.finish_answer(AnswerAction(tool='finish_answer', answer=answer,
                                        evidence_ids=['run']), context)


def test_rejects_wrong_calculation():
    with pytest.raises(ValueError, match='figures'):
        finish(calculation(), '17 plus 25 is 99.')


def test_uses_actual_output_instead_of_paraphrase():
    result = finish(calculation(), '17 plus 25 is 42.')
    assert result['answer'] == 'Ran the numbers: 42.'
    assert result['verification'] == 'tool_output'


def test_web_citation_cannot_hide_wrong_calculation():
    context = calculation()
    context.record('read_webpage', {}, {'success': True, 'evidence_id': 'page',
                   'observation': {'url': 'https://example.com', 'text': 'Background information'}})
    with pytest.raises(ValueError, match='figures'):
        CapabilityTools.finish_answer(AnswerAction(tool='finish_answer', answer='The answer is 99',
                                      evidence_ids=['run', 'page']), context)


def test_cannot_invert_labels_or_boolean_results():
    result = finish(calculation('approved=False'), 'Approved!')
    assert result['answer'] == 'Ran the numbers: approved=False.'


@pytest.mark.parametrize('output,extra', [('', {}), ('42', {'result_truncated': True})])
def test_incomplete_outputs_do_not_finish(output, extra):
    with pytest.raises(ValueError):
        finish(calculation(output, **extra), '42')


def test_synthesis_does_not_claim_independent_verification():
    context = TaskContext(goal='Research')
    context.record('read_webpage', {}, {'success': True, 'evidence_id': 'run',
                   'observation': {'url': 'https://example.com', 'text': 'A source'}})
    context.answer = finish(context, 'An interpretation')
    assert context.answer['verification'] == 'evidence_linked'
    assert 'not been independently verified' in answer_notes(context)[0]
