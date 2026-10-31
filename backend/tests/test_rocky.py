"""Kip as Rocky's younger brother: the deterministic parts of the character.

Whether replies actually sound like Rocky is up to the model and is not testable here.
These check what code decides: who Kip is in every spoken view, which words carry his
voice (speech and answers, never tool arguments or the chat action line), practical care
that still yields to a request for space, celebrating only real achievements, and that
saved state from the previous character keeps what it learned.
"""
import json

import pytest

from app.agents import prompts
from app.chat.prompt import CHAT_SYSTEM, build_messages
from app.persona import Persona
from app.persona.cues import read_cues
from app.persona.profile import BASELINE, DRIFT, IDENTITY, PREVIOUS_BASELINES, TONES, VOICE
from app.persona.state import CHARACTER_VERSION, PersonaState
from tests.fakes import ScriptedLLM, complete, tool
from tests.test_character import said
from tests.test_persona import Clock, _manager, given


# ── who he is ─────────────────────────────────────────────────────────────────
def test_identity_is_rockys_younger_brother_without_borrowed_memories():
    assert "Rocky's younger brother" in IDENTITY and "Eridian engineer" in IDENTITY
    assert "robot" in IDENTITY and "share no memories" in IDENTITY and "invent none" in IDENTITY
    assert "Never fake a result" in IDENTITY and "investigate, never invent" in IDENTITY
    assert "respect their choices" in IDENTITY  # protective, not controlling
    # The speech pattern, with the guard against playing it as dim or babyish.
    assert "question?" in VOICE and "Grammar imperfect, reasoning precise" in VOICE
    assert "never babyish, robotic or dim" in VOICE and "Praise only real achievements" in VOICE
    assert all(not tone.startswith("dry") for tone in TONES.values())


def test_every_spoken_view_carries_the_same_character_even_when_trimmed():
    clock = Clock()
    persona = Persona(now=clock)
    for index in range(5):
        persona.observe(given(person=f"person_{index}", thing="a very long object name " * 3))
    # Speech during an errand, a written answer, and conversation: one character.
    speech = persona.voice_view([f"person_{i}" for i in range(5)])
    chat = persona.voice_view(speaker="person_0", message="I only slept four hours. I passed my exam! "
                                                          "Thanks. Going to bed.")
    for view in (speech, chat, persona.voice_view()):
        assert view["identity"] == IDENTITY and view["voice"] == VOICE
    system = build_messages("Kip", chat, [], {"room": "hall"}, [], "hi")[0]["content"]
    assert IDENTITY in system and VOICE in system and "never an impersonal help desk" in system


# ── his voice is for people, never for machinery ────────────────────────────────
def test_the_voice_is_scoped_to_words_for_people():
    assert "command_id, IDs, summary and every other field stay plain" in prompts.PERSONA_SPEECH
    assert "word for word" in prompts.PERSONA_SPEECH
    assert "tool arguments" in prompts.PERSONA_ANSWER and "citation labels" in prompts.PERSONA_ANSWER
    assert "plain standard English, not your voice" in CHAT_SYSTEM
    for name in ("COORDINATOR", "CRITIC", "GOAL_PLANNER", "EXPLORER", "CAPABILITY_EXPLORER", "COMMON"):
        text = getattr(prompts, name)
        assert "Rocky" not in text and "question?" not in text, name


@pytest.mark.asyncio
async def test_rocky_speech_reaches_people_while_planning_calls_never_see_the_character():
    persona = Persona(now=Clock())
    styled = "Neave. Dinner is ready. Food is hot. Good, good."
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Notify.", "task": "Tell Neave dinner is ready."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message=styled),
        {"action": "report", "summary": "Message delivered."}, complete,
        {"approved": True, "summary": "Notification evidenced."},
    ])
    mgr, bus = _manager(fake, persona)
    spoken = []
    bus.add_listener(lambda e: spoken.append(e) if e["type"] == "person_spoken_to" else None)
    task = mgr.start("Tell Neave dinner is ready.")
    await mgr.runner
    assert task.status == "completed", task.summary
    carrying = [sent for sent, _ in fake.calls if "persona" in json.loads(sent[1]["content"])]
    assert carrying, "the speech decision carries the character"
    for sent in carrying:
        assert json.loads(sent[1]["content"])["persona"]["identity"] == IDENTITY
        assert "every other field stay plain" in sent[0]["content"]
    for sent, _ in fake.calls:
        if "persona" not in json.loads(sent[1]["content"]):
            assert IDENTITY not in sent[0]["content"] and IDENTITY not in sent[1]["content"]
    assert spoken and spoken[-1]["data"]["arguments"] == {"person": "neave", "message": styled}


# ── care is practical, and still yields to a request for space ─────────────────
def test_short_rest_or_food_is_named_once_and_left_to_their_choice():
    stance = Persona(now=Clock()).stance("neave", "I only slept four hours, back to it.")
    assert "They said 'slept four hours'" in stance and "their choice" in stance
    assert read_cues("I haven't eaten since breakfast").rest_neglected
    assert read_cues("Skipped lunch again").neglect == "skipped lunch"
    assert not read_cues("I slept eight hours, feel great").rest_neglected
    assert not read_cues("I'm not working late tonight").rest_neglected


def test_leave_me_alone_still_wins_over_care_and_celebration():
    clock = Clock()
    persona = Persona(now=clock)
    persona.observe(given())  # a care delivery that would normally earn a check-in
    said(persona, "Leave me alone. I was up all night and I passed my exam.")
    stance = persona.stance("neave", "Leave me alone. I was up all night and I passed my exam.")
    assert "want space" in stance and "not control" in stance
    assert "rest or food" not in stance and "celebrate" not in stance
    # Code still enforces it, whatever the character would like to do.
    assert persona.state.followups == [] and persona.wants_space("neave")
    assert persona.mind_view()["relationships"]["neave"]["do_not_disturb"] is True
    assert "rest or food" not in persona.stance("neave", "Skipped lunch.")  # the quiet period holds


def test_only_real_achievements_are_celebrated_and_never_over_bad_news():
    persona = Persona(now=Clock())
    assert "celebrate it openly" in persona.stance("neave", "I passed my driving test!")
    assert "celebrate" not in persona.stance("neave", "Thanks for the tea.")  # thanks is not an achievement
    assert "celebrate" not in persona.stance("neave", "I passed my exam but Dad is in hospital.")
    assert "celebrate" not in persona.stance("paul", "I finished the report and I'm exhausted.")
    assert not read_cues("I haven't finished it, I did not pass").achievement


def test_going_to_sleep_gets_a_goodnight_that_promises_nothing_extra():
    stance = Persona(now=Clock()).stance("neave", "Right, I'm off to bed.")
    assert "short goodnight" in stance and "promise nothing more" in stance
    assert not read_cues("I'm not going to bed yet").sleeping


def test_humour_is_literal_and_disagreement_is_plain_not_submissive():
    persona = Persona(now=Clock())
    stance = persona.stance("minu", "What's for dinner?")
    assert "literal, blunt or earnestly serious" in stance and "never at their expense" in stance
    assert "where the evidence disagrees" in persona.stance("minu", "You're wrong about the keys.")
    assert "Take thanks simply" in persona.stance("minu", "Thanks!")


# ── the previous character's experience carries over ──────────────────────────
def test_a_state_saved_by_the_previous_character_keeps_its_earned_drift():
    previous = PREVIOUS_BASELINES[2]
    saved = {"character_version": 2,
             "traits": {**previous, "sarcasm": previous["sarcasm"] + 0.05, "caution": previous["caution"] - 0.02,
                        "curiosity": previous["curiosity"] + 0.5},  # out of range: clamped to the new limit
             "legacy_traits": {"enthusiasm": 0.71},
             "relationships": {"neave": {"familiarity": 0.4, "warmth": 0.7, "interactions": 9}},
             "age_seconds": 500.0}
    state = PersonaState.model_validate(saved)
    assert state.character_version == CHARACTER_VERSION == 3
    assert state.traits["sarcasm"] == pytest.approx(BASELINE["sarcasm"] + 0.05)
    assert state.traits["caution"] == pytest.approx(BASELINE["caution"] - 0.02)
    assert state.traits["curiosity"] == pytest.approx(BASELINE["curiosity"] + DRIFT["curiosity"])
    assert state.traits["expressiveness"] == BASELINE["expressiveness"]
    assert state.traits["honesty"] == BASELINE["honesty"]
    assert state.legacy_traits == {"enthusiasm": 0.71} and state.relationships["neave"].interactions == 9
    again = PersonaState.model_validate(json.loads(state.model_dump_json()))
    assert again.traits == state.traits  # migrating once only
