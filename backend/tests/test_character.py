"""Kip's character: grounded appraisal, per-person relationships learned from conversation,
humour that adapts to people and moments, boundaries, curiosity, and identity continuity.

These test the deterministic layer that shapes behaviour. Whether the model's words are
actually funny or kind is not testable here; see PERSONA.md for what is and isn't covered.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.chat.prompt import CHAT_SYSTEM
from app.chat.session import ChatSession
from app.config import Settings
from app.events.bus import EventBus
from app.main import create_app
from app.memory.recall import MemoryStore
from app.memory.store import PersistentMemory
from app.persona import Persona
from app.persona.cues import read_cues
from app.persona.engine import FOLLOWUP_DELAY, QUIET_SECONDS
from app.persona.profile import BASELINE, IDENTITY, TONES, VOICE
from app.persona.state import CHARACTER_VERSION, PersonaState
from app.tasks.manager import TaskManager
from tests.fakes import WaitingLLM
from tests.test_chat import StreamingModel, reply
from tests.test_persona import Clock, check_in_later, event

BANNED = ("how may i assist", "i hope that helps", "is there anything else", "as an ai language model",
          "successfully completed the requested task")


def said(persona, text, speaker="neave"):
    persona.observe(event("chat_message", turn={"role": "user", "text": text, "speaker": speaker}))


def outcome(persona, ok, goal="Look up the AQI", tools=("web_search",)):
    task = {"goal": goal, "conditions": [{"kind": "answer", "required_tools": list(tools)}]}
    persona.observe(event("task_completed" if ok else "task_failed", task=task))


# ── stable core ─────────────────────────────────────────────────────────────────
def test_foundational_values_never_drift_but_temperament_does():
    persona = Persona(now=Clock())
    for trait in ("honesty", "loyalty"):
        persona.reinforce(trait, -1.0)
        assert persona.trait(trait) == BASELINE[trait]
    persona.reinforce("caution", 1.0)
    assert BASELINE["caution"] < persona.trait("caution") <= BASELINE["caution"] + 0.1 + 1e-9


def test_character_text_avoids_assistant_cliches_and_scripted_lines():
    text = " ".join([IDENTITY, VOICE, CHAT_SYSTEM, *TONES.values()]).casefold()
    for phrase in BANNED:
        # The voice may name what to avoid, but must never instruct Kip to say it.
        assert text.count(phrase) <= text.count("no " + phrase)
    assert "You know you are a robot" in IDENTITY


def test_earlier_character_state_is_carried_over_without_losing_anything():
    saved = {"traits": {"empathy": 0.9, "loyalty": 0.8, "curiosity": 0.703, "enthusiasm": 0.71, "caution": 0.51},
             "mood": {"valence": -0.14, "energy": 0.5, "updated_at": 100.0},
             "relationships": {"moira": {"familiarity": 0.02, "warmth": 0.51, "interactions": 1, "chats": 1}},
             "episodes": [{"summary": "Could not finish: Find the keys", "feeling": "concerned", "at": 90.0}],
             "age_seconds": 100.0}
    state = PersonaState.model_validate(saved)
    assert state.character_version == CHARACTER_VERSION
    assert state.legacy_traits == saved["traits"]
    # Earned drift is carried onto the new baselines; renamed traits follow.
    assert state.traits["curiosity"] == pytest.approx(BASELINE["curiosity"] + 0.003)
    assert state.traits["caution"] == pytest.approx(BASELINE["caution"] + 0.01)
    assert state.traits["initiative"] == pytest.approx(BASELINE["initiative"] + 0.01)
    assert state.relationships["moira"].warmth == 0.51 and state.episodes[0].feeling == "concerned"
    # Saving and loading again never migrates twice.
    again = PersonaState.model_validate_json(state.model_dump_json())
    assert again.traits == state.traits and again.legacy_traits == saved["traits"]


def test_people_met_only_through_house_errands_are_forgotten():
    # Errands in the retired house simulation made bonds without a single chat.
    saved = {"relationships": {"neave": {"familiarity": 0.36, "warmth": 0.5, "chats": 13},
                               "moira": {"familiarity": 0.02, "warmth": 0.51, "interactions": 1}},
             "followups": [{"person": "moira", "reason": "brought her tea", "due_at": 1.0, "expires_at": 9.0},
                           {"person": "neave", "reason": "said he was tired", "due_at": 1.0, "expires_at": 9.0}]}
    state = PersonaState.model_validate(saved)
    assert set(state.relationships) == {"neave"}
    assert [f.person for f in state.followups] == ["neave"]


# ── appraisal: outcomes judged against expectation ─────────────────────────────
def test_a_surprising_failure_stings_more_than_an_expected_one():
    reliable, shaky = Persona(now=Clock()), Persona(now=Clock())
    for _ in range(6):
        outcome(reliable, True)
        outcome(shaky, False)
    before = reliable.state.mood.valence, shaky.state.mood.valence
    outcome(reliable, False)
    outcome(shaky, False)
    assert before[0] - reliable.state.mood.valence > before[1] - shaky.state.mood.valence
    assert reliable.state.skills["looking things up"].attempts == 7


def test_repeated_failure_makes_kip_methodical_then_success_brings_relief():
    persona = Persona(now=Clock())
    for _ in range(3):
        outcome(persona, False)
    assert persona.mood_label() == "determined"
    assert persona.trait("caution") > BASELINE["caution"]
    frustration = persona.state.mood.frustration
    outcome(persona, True)
    assert persona.state.mood.frustration < frustration * 0.5
    assert persona.state.episodes[-1].summary.endswith("(after a run of failures)")
    assert persona.state.skills["looking things up"].streak == 1


def test_routine_success_is_not_a_memorable_moment_but_failures_always_are():
    persona = Persona(now=Clock())
    outcome(persona, True)
    episodes = len(persona.state.episodes)
    outcome(persona, True)
    assert len(persona.state.episodes) == episodes
    outcome(persona, False)
    assert persona.state.episodes[-1].summary == "Could not finish: Look up the AQI"


def test_track_record_is_honest_and_reaches_the_voice():
    persona = Persona(now=Clock())
    for ok in (True, True, False):
        outcome(persona, ok)
    outcome(persona, True, goal="Work out 6 times 7", tools=("run_python",))
    assert persona.track_record() == {"looking things up": "2 of 3 succeeded", "calculations": "1 of 1 succeeded"}
    assert persona.voice_view(speaker="neave", message="How confident are you?")["track_record"]


# ── relationships come from real interaction, per person ───────────────────────
def test_conversation_builds_a_relationship_only_with_a_known_speaker():
    clock = Clock()
    persona = Persona(now=clock)
    said(persona, "Morning, Kip.", speaker=None)
    assert persona.state.relationships == {}
    said(persona, "Morning, Kip.")
    said(persona, "How's it going?")
    bond = persona.state.relationships["neave"]
    assert bond.chats == 2 and bond.interactions == 1  # one visit, two lines
    clock.seconds += 3600
    said(persona, "Back again.")
    assert persona.state.relationships["neave"].interactions == 2
    assert "minu" not in persona.state.relationships


def test_humour_adapts_to_each_person_from_how_they_respond():
    clock = Clock()
    persona = Persona(now=clock)
    for _ in range(10):  # familiarity comes from many separate conversations, not one long one
        said(persona, "hello", speaker="neave")
        said(persona, "haha, good one", speaker="neave")
        said(persona, "That's not funny. Be serious.", speaker="paul")
        clock.seconds += 3600
    neave, paul = persona.stance("neave", "What's for dinner?"), persona.stance("paul", "What's for dinner?")
    assert "Humour suits this" in neave and "teasing" in neave
    assert "humour only if they start it" in paul
    assert persona.state.relationships["paul"].humour_reception < 0.2


def test_friends_get_warmth_and_humour_while_an_unknown_speaker_gets_reserve():
    persona = Persona(now=Clock())
    stance = persona.stance("minu", "hi")  # no shared history yet, but someone Kip can name
    assert "A friend:" in stance and "Humour suits this" in stance
    assert "teasing" not in stance  # teasing is still earned
    unknown = persona.stance(None, "hi")
    assert "don't know who is speaking" in unknown and "Humour suits" not in unknown


# ── care: listen first, respect boundaries ─────────────────────────────────────
def test_distress_puts_humour_away_and_asks_for_listening_not_plans():
    persona = Persona(now=Clock())
    said(persona, "I'm exhausted. Rough day.")
    stance = persona.stance("neave", "I'm exhausted. Rough day.")
    assert "No jokes" in stance and "listen" in stance and "errands unless they ask" in stance
    bond = persona.state.relationships["neave"]
    assert bond.concern > 0 and bond.concern_reason == "exhausted"
    assert persona.mood_label() == "concerned"
    # Later in the same conversation the concern carries over, gently.
    assert "Earlier they said 'exhausted'" in persona.stance("neave", "What's on TV?")


def test_negated_or_absent_feelings_are_never_assumed():
    assert not read_cues("I'm not tired at all").distress
    assert not read_cues("Where are my keys?").sensitive
    assert read_cues("My grandmother is in hospital").serious
    persona = Persona(now=Clock())
    said(persona, "I'm not tired at all")
    assert persona.state.relationships["neave"].concern == 0


def test_serious_topics_turn_humour_off_for_everyone():
    persona = Persona(now=Clock())
    for _ in range(5):
        said(persona, "lol", speaker="neave")
    assert "no humour" in persona.stance("neave", "Dad has been taken to hospital")


def test_asking_for_space_is_honoured_by_check_ins():
    clock = Clock()
    persona = Persona(now=clock)
    said(persona, "Morning.")
    check_in_later(persona)
    said(persona, "Please leave me alone for a bit.")
    assert persona.state.followups == []
    assert "want space" in persona.stance("neave", "ok")
    check_in_later(persona)
    clock.seconds += FOLLOWUP_DELAY
    assert persona.due_followups() == []
    assert persona.voice_view(["neave"])["relationships"]["neave"]["do_not_disturb"] is True
    clock.seconds += QUIET_SECONDS
    assert not persona.wants_space("neave")


def test_a_due_check_in_reaches_the_conversation_and_one_reply_settles_it():
    clock = Clock()
    persona = Persona(now=clock)
    said(persona, "I've got a cold.")
    check_in_later(persona)
    clock.seconds += FOLLOWUP_DELAY
    assert "check-in is natural" in persona.stance("neave", "Hi Kip")
    persona.observe(event("chat_message", turn={"role": "kip", "text": "Feeling better?", "to": "neave"}))
    assert persona.due_followups() == []


# ── curiosity: noticing, not inventing ─────────────────────────────────────────
def test_what_kip_reads_raises_curiosity_but_an_answer_does_not():
    persona = Persona(now=Clock())
    persona.observe(event("capability_observed", tool="finish_answer"))
    assert persona.state.mood.intrigue == pytest.approx(0.2)
    persona.observe(event("capability_observed", tool="read_webpage"))
    assert persona.state.mood.intrigue > 0.2
    assert persona.trait("curiosity") > BASELINE["curiosity"]


# ── in conversation ────────────────────────────────────────────────────────────
def _session(replies):
    bus = EventBus()
    model = StreamingModel(replies)
    manager = TaskManager(model, bus, Settings())
    persona = Persona()
    persona.attach(bus)
    session = ChatSession(model, bus, manager, Settings(), store=MemoryStore(":memory:"), persona=persona)
    session.attach()
    return session, model, manager, persona


@pytest.mark.asyncio
async def test_distress_reaches_the_prompt_and_never_authorizes_an_errand():
    # The model over-eagerly proposes a wellbeing errand; nothing may start from it.
    session, model, manager, persona = _session([reply("Rough day, then.", task="Make Neave some tea")])
    session.submit("I'm exhausted.", speaker="neave")
    await session.drain()
    latest = model.prompts[-1][-1]["content"]
    assert "Stance:" in latest and "No jokes" in latest
    assert "robot" in model.prompts[-1][0]["content"]
    assert manager.active_id is None
    assert session.pending_offer == "Make Neave some tea"  # at most an offer awaiting a yes
    assert session.history()[-1]["to"] == "neave"


@pytest.mark.asyncio
async def test_failure_narration_is_plain_fact_without_stock_reactions():
    session, _, manager, _ = _session([])
    task = {"goal": "Look up the AQI", "status": "failed"}
    session.bus.emit("task_failed", task=task, summary="No page I could read gave today's AQI.")
    text = session.history()[-1]["text"]
    assert text == "Not finish. Bad. No page I could read gave today's AQI."
    assert not any(phrase in text.casefold() for phrase in BANNED)


# ── identity continuity across restarts ────────────────────────────────────────
def test_everything_kip_learned_survives_save_and_reload(tmp_path):
    path = tmp_path / "persistent.json"
    memory = PersistentMemory()
    clock = Clock()
    persona = Persona(memory.persona, now=clock)
    for _ in range(3):
        said(persona, "haha", speaker="neave")
    said(persona, "I'm stressed", speaker="paul")
    outcome(persona, False)
    memory.save(path)
    restored = Persona(PersistentMemory.load(path).persona, now=clock)
    assert restored.public() == persona.public()
    assert restored.stance("paul", "hi") == persona.stance("paul", "hi")


def test_conversation_changes_are_saved_on_shutdown_and_back_after_restart(tmp_path):
    settings = Settings(memory_file=tmp_path / "persistent.json", graph_file=tmp_path / "beliefs.json")
    db = tmp_path / "kip.db"
    with TestClient(create_app(client=WaitingLLM(), settings=settings, memory_store=MemoryStore(db))) as client:
        assert client.post("/api/chat", json={"message": "haha, nice one", "speaker": "neave"}).status_code == 202
    saved = json.loads(MemoryStore(db).get_state("persistent"))
    assert saved["persona"]["relationships"]["neave"]["chats"] == 1
    assert not (tmp_path / "persistent.json").exists()  # nothing is written to JSON any more
    with TestClient(create_app(client=WaitingLLM(), settings=settings, memory_store=MemoryStore(db))) as client:
        bond = client.get("/api/persona").json()["relationships"]["neave"]
    assert bond["chats"] == 1 and bond["humour_reception"] > 0.5
