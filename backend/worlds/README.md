# World definition and robot boundary

Edit `house.json`, then press Reset. It is loaded at startup and reread on reset.
For another file, set `WORLD_FILE` to its absolute path in `backend/.env` and restart.
Keep IDs lowercase, beginning with a letter, using letters, digits and underscores
(maximum 40 characters). Dictionary keys must equal the corresponding `id` fields.

The house approximates the supplied sketch. The yellow areas are treated as the
bedroom corridor and entrance vestibule. The kitchen opens into the hall; the
bedrooms open into the corridor; the office opens into the entrance vestibule.
Rooms have a short description and fixtures whose state follows the hour (a kettle
steaming in the morning, a lamp on in the evening). People move and change activity on
their schedules. Outside, exact measurements and continuous motion are not simulated.

## Fields

- `name`: scenario title.
- `rooms`: keyed room definitions with `id`, `name`, reciprocal `connections`,
  `outline` (ordered polygon vertices), `anchor` (navigation/display point inside
  the room), and `color`. Coordinates are `[x, z]`, with x right and z down in the
  source sketch. Geometry is optional for headless test fixtures; the renderer
  uses generic tiles when it is absent. Optional `description` (one or two sentences)
  and `features`: fixtures with `id`, `name`, optional `description`, a default `state`
  and timed `states` (`from_hour`, `to_hour`, `state`, wrapping midnight). Features are
  perceived only by being in the room; they never appear in `get_map`.
- `doors`: room pairs, shared boundary `position`, and opening `width`. Keep each
  doorway on both room outlines. The renderer passes the avatar through doorways;
  the simulator currently checks adjacency, not continuous collision geometry.
- `robot`: starting `room`.
- `people`: keyed `id`, `name`, `room`, and optional `dialogue` entries. Each entry
  has `topics` (alternative word phrases) and a spoken `response`. The first
  matching entry answers. Matching ignores case and word order within a phrase.
  Keep topics nonempty. Unmatched messages receive the person's own
  `fallback_response`, or a generic acknowledgement without one. Optional `activity`
  (what they are visibly doing) applies when no schedule entry says otherwise; each
  `schedule` entry may set its own `activity`.
- `objects`: keyed `id`, `name`, optional `portable` (default true), and exactly one
  `location`: `{"kind":"room","id":"office"}`, `{"kind":"person","id":"dad"}`,
  or `{"kind":"robot","id":"robot"}`. Do not store a separate inventory.

For example, add an object `medicine` to an existing room and a person `alice`:

```json
{
  "id": "alice",
  "name": "Alice",
  "room": "office",
  "dialogue": [
    {"topics": ["medicine"], "response": "I need the medicine, please."}
  ]
}
```

This is one person entry, not a complete world file. Put it under `people.alice`
and define the `medicine` object under `objects.medicine`. No Python prompt,
engine branch, command menu or rendering code needs those names added.

## Information available to a future robot

| Interface | Simulator output | Future source |
|---|---|---|
| `get_map` | Static layout only | Surveyed floor plan / mapping |
| `get_status` | Current room, carried items | Localization and manipulation state |
| `look` | Current room's visible semantic entities: people and what they are doing, objects, fixtures and their state, the room's description | Camera scan and perception |
| Senses (no tool) | The same view of the current room, plus arrivals, departures and activity changes there, given to chat every turn | Continuous camera and microphone |
| `talk_to` | Local spoken response | Speaker, microphone, speech transcription |
| `move_to` | Adjacent-room action result | Navigation controller and wheel feedback |
| `pick_up`, `drop`, `give` | Ownership/action result | Arm controller and grasp/release feedback |

These are idealized interfaces. A room scan currently sees all floor objects and
occupants, identifies them correctly, and has no field-of-view or occlusion model.
Movement and manipulation succeed deterministically when preconditions hold.
The simulator does not claim to reproduce realistic sensing or hardware safety.
A real adapter must return uncertainty, partial results and failures rather than
consulting simulator truth. Robot beliefs may be wrong or stale.

The full JSON and observer HTTP/WebSocket snapshots are **not agent inputs**.
In particular, NPC dialogue tables and other-room occupants must never enter model
context. A language-model interpretation of speech is stored with its source
quote and explicitly marked unverified; it is not an extra sensor or hidden fact.

Regression tests retain the original four-room fixture under `tests/fixtures`.
Tests for this house and an unrelated workshop verify scenario loading, observation
isolation, transcript grounding, delivery and a subsequent notification. Scripted
model tests validate orchestration, not real-model planning reliability.
