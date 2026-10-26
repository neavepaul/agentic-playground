# Kip's character: architecture, gaps and test coverage

The character spec (TARS / Rocky / Baymax / Atom, dryly funny, loyal, curious, honest) is
implemented as a **deterministic persona layer** that shapes judgement and pitch, plus a
compact identity in the one chat prompt. It adds **no model calls**: everything below is
plain state updated from events that already exist on the bus.

## Where things were (before this change)

| Spec area | Before | Status now |
|---|---|---|
| Core identity | A different character ("caring nurse-bot, loyal engineer, anxious protocol droid"); warm, upbeat voice | Rewritten: robot that knows it, dry, loyal, curious, honest; compact `IDENTITY` + `VOICE` in `persona/profile.py` |
| Traits | 5 traits, one drift limit for all | 12 traits from the spec; per-trait drift; **honesty and loyalty never drift** |
| Humour | None | Per-reply stance: off for distress/serious topics, earned with familiarity, learned per person from laughter / "not funny" |
| Scripted dialogue | `REACTIONS` table emitted after every task ("All done. I hope that helps.") | Removed. Outcomes are appraised silently; narration is plain fact |
| Emotion | `task_success → valence +0.15` | Outcomes judged against Kip's own track record (surprise), who it mattered to, streaks; resilience dampens dips; frustration → methodical, not upset |
| Relationships | Only from `give`/`talk_to` tool events; chat changed nothing | Conversation builds per-speaker familiarity (per visit, not per line); unknown speakers touch nothing |
| Care / boundaries | Check-in planned after *every* delivery | Check-ins only for care items or expressed concern; "leave me alone" suppresses check-ins and is **enforced in code** for idle initiative |
| Curiosity | Tiny trait bump from web/python results | Novel objects raise intrigue; objects seen in 3+ places become an *unconfirmed* suspicion the idle mind may check |
| Self-memory | Task counts in `self_model` (planning only) | Per-skill track record (`finding things: 5 of 7`) given to chat so confidence claims are grounded |
| Epistemics | "Facts only from memories" | Also: keep apart seen / told / remembered / suspected; never invent a past event, even for a joke |
| Restart continuity | Persona saved only during idle ticks | Saved every mind loop tick when dirty and at shutdown |

Already solid and left unchanged: consent (`chat/consent.py`), unsupported-claim checks,
memory contexts (actual / pretend / hypothetical), corrections superseding memories,
speaker picker, Coordinator / Explorer / Critic, belief graph and distillation.

## Architecture

```
bus events ──► persona.observe ──► appraisal.py (pure) ──► PersonaState (persisted in persistent.json)
                                     │ cues.py: the person's own words only
                                     ▼
          voice_view(speaker, message) ─► chat prompt: identity, voice, mood/tone,
                                           Stance line, track record
          voice_view(people)            ─► Explorer speech / written answers (unchanged contract)
          mind_view() + threshold       ─► idle initiative; boundary check in code before acting
```

* **Stable core** (`profile.py`): values and temperament baselines, identity, voice, tones.
* **Persistent state** (`state.py`): mood (valence, energy, confidence, frustration, intrigue),
  relationships (familiarity, warmth, chats, humour reception, concern + the words that raised it,
  quiet-until), skills, object sightings, episodes, follow-ups. All fields default, so old files load.
* **Appraisal** (`appraisal.py`): events → state changes, judged in context.
* **Stance** (`engine.py`): one deterministic sentence per reply about how to pitch it. It never
  changes facts, consent or what Kip will do.
* **Safety stays elsewhere**: personality can lower or raise the bar for idle initiative, but cannot
  grant permission; consent, claim checks and `autonomous_errands` are untouched.

### Migration of existing state

A persona saved by the earlier character (no `character_version`) is migrated on load: drift each
trait earned is carried onto the new baseline (`enthusiasm` → `initiative`), the old values are kept
verbatim in `legacy_traits`, and relationships, episodes and age are untouched. Verified against the
live `backend/memory/persistent.json` in memory only (file hash unchanged); the file is rewritten in
the new shape the next time the backend saves.

## What is tested vs. aspirational

Directly tested (`tests/test_character.py`, `tests/test_persona.py`, `tests/test_chat.py`):
foundational values don't drift; legacy migration keeps everything; surprising failures sting more;
repeated failure → methodical + higher initiative bar, then relief; routine successes aren't
"moments" but failures always are; self-initiated errands weigh less; honest track record; relationships
per speaker only; humour adapts per person and is held back with strangers; distress → listen, no jokes,
no errands (and the over-eager errand becomes an offer only); negated feelings aren't assumed; serious
topics switch humour off; boundaries suppress check-ins and idle initiative until they lapse; care-only
check-ins; a due check-in surfaces once and one reply settles it; novelty/suspicions; no stock
reactions in narration; full state round-trips through save/load and through an app restart; prompt
size guard.

Not testable here (needs the real model and a person): whether replies are actually funny, warm or
non-repetitive; whether the model follows the stance; long-run feel over months. Speaker identity
still comes from the UI picker. Cue detection is keyword-based and conservative: it will miss
distress expressed indirectly (by design it prefers missing a cue to inventing a feeling).
Embodied expression is limited to choices the simulator supports (what to check, when to act).

## Tuning notes

Persona time is simulated time (60× real time by default), so "8 hours of quiet" after "leave me
alone" is about 8 real minutes in realtime mode. Constants live at the top of `persona/engine.py`.
