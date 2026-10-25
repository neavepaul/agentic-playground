"""Static instructions describe capabilities, never answers to our test cases."""
import re

from app.agents import prompts


def test_static_prompts_do_not_include_test_links_questions_or_answers():
    instructions = '\n'.join(value for name, value in vars(prompts).items()
                             if name.isupper() and isinstance(value, str))
    assert not re.search(r'https?://', instructions, re.I)
    for fixture in ('example.com', 'aqi.in', 'mumbai', '6 * 7', '6 × 7', 'print(42)',
                    '"code":"a short program"', '"url":"the supplied URL"'):
        assert fixture not in instructions.casefold()
