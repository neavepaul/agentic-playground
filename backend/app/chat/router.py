"""Kip's intent router: what a message is about, read locally before any model reasons over it.

A small classifier names the intent: TF-IDF word and character features, multinomial
logistic regression, trained in numpy at first use from router_data/intents.tsv (plus
reviewed corrections) in well under a second. A domain lexicon and a few patterns fill in
the domain, the people and teams named, the aspect asked about and the time meant. In
sport "where did he finish" asks for a position, not a place, so the aspect is read from
the domain, never from the question word. The previous information request fills in what
a short follow-up leaves out ("What about Max?", "And yesterday?").

The router only describes the message. session.py decides what runs, after cancellation,
permissions, consent and the house checks, none of which the router can override.
"""
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .intent import is_live_fact, read_intent

DATA = Path(__file__).parent / "router_data" / "intents.tsv"
LABELS = ("conversation.casual", "conversation.emotional", "information.sports", "information.weather",
          "information.finance", "information.news", "information.general", "memory.recall", "memory.update",
          "action.request", "reasoning.complex")
# Answers that change by the hour: always looked up, never recited from a model's memory.
LIVE = frozenset({"information.sports", "information.weather", "information.finance", "information.news"})
THRESHOLD = 0.55           # below this the router has no opinion and the usual path decides
FRAME_TURNS = 3            # a follow-up may lean on the information request this many messages back
SPORTS = frozenset({"f1", "cricket", "football", "tennis", "basketball", "motogp"})
DOMAIN_NAMES = {"f1": "Formula 1", "cricket": "cricket", "football": "football", "tennis": "tennis",
                "basketball": "NBA", "motogp": "MotoGP", "weather": "weather", "finance": "markets",
                "news": "news"}
INTENT_OF_DOMAIN = {**{d: "information.sports" for d in SPORTS}, "weather": "information.weather",
                    "finance": "information.finance", "news": "information.news"}

# ── lexicon: alias -> (canonical name, domain, kind). Ambiguous aliases ("max", "city",
# "apple") count only when something else already points at their domain. ──────────────
_F1 = {"lando|norris": "Lando Norris", "oscar|piastri": "Oscar Piastri", "max|verstappen": "Max Verstappen",
       "charles|leclerc": "Charles Leclerc", "lewis|hamilton": "Lewis Hamilton", "george|russell": "George Russell",
       "kimi|antonelli": "Andrea Kimi Antonelli", "fernando|alonso": "Fernando Alonso", "lance|stroll": "Lance Stroll",
       "pierre|gasly": "Pierre Gasly", "franco|colapinto": "Franco Colapinto", "carlos|sainz": "Carlos Sainz",
       "alex|albon": "Alexander Albon", "liam|lawson": "Liam Lawson", "isack|hadjar": "Isack Hadjar",
       "arvid|lindblad": "Arvid Lindblad", "esteban|ocon": "Esteban Ocon", "ollie|oliver|bearman": "Oliver Bearman",
       "nico|hulkenberg|hülkenberg": "Nico Hülkenberg", "gabriel|bortoleto": "Gabriel Bortoleto",
       "checo|sergio|perez|pérez": "Sergio Pérez", "valtteri|bottas": "Valtteri Bottas",
       "mclaren": "McLaren", "ferrari": "Ferrari", "red bull": "Red Bull", "mercedes": "Mercedes",
       "aston martin|aston": "Aston Martin", "alpine": "Alpine", "williams": "Williams", "haas": "Haas",
       "racing bulls": "Racing Bulls", "audi|sauber": "Audi", "cadillac": "Cadillac"}
_CRICKET = {"virat|kohli": "Virat Kohli", "rohit": "Rohit Sharma", "bumrah|jasprit": "Jasprit Bumrah",
            "shubman|gill": "Shubman Gill", "rishabh|pant": "Rishabh Pant", "jadeja": "Ravindra Jadeja",
            "hardik": "Hardik Pandya", "surya|suryakumar": "Suryakumar Yadav", "dhoni": "MS Dhoni",
            "siraj": "Mohammed Siraj", "babar": "Babar Azam", "root": "Joe Root", "stokes": "Ben Stokes",
            "cummins": "Pat Cummins", "starc": "Mitchell Starc", "smith": "Steve Smith",
            "rcb": "Royal Challengers Bengaluru", "csk": "Chennai Super Kings", "mi": "Mumbai Indians",
            "kkr": "Kolkata Knight Riders", "srh": "Sunrisers Hyderabad", "rr": "Rajasthan Royals",
            "india": "India", "australia": "Australia", "england": "England", "pakistan": "Pakistan",
            "new zealand": "New Zealand", "south africa": "South Africa", "sri lanka": "Sri Lanka"}
_FOOTBALL = {"arsenal": "Arsenal", "liverpool": "Liverpool", "chelsea": "Chelsea", "spurs|tottenham": "Tottenham",
             "man city|manchester city|city": "Manchester City", "man united|man utd|manchester united|united":
             "Manchester United", "real madrid|madrid": "Real Madrid", "barcelona|barca": "Barcelona",
             "bayern": "Bayern Munich", "psg": "Paris Saint-Germain", "juventus|juve": "Juventus",
             "messi": "Lionel Messi", "ronaldo": "Cristiano Ronaldo", "haaland": "Erling Haaland",
             "salah": "Mohamed Salah", "mbappe|mbappé": "Kylian Mbappé", "saka": "Bukayo Saka"}
_TENNIS = {"sinner": "Jannik Sinner", "alcaraz": "Carlos Alcaraz", "djokovic": "Novak Djokovic",
           "swiatek": "Iga Świątek", "sabalenka": "Aryna Sabalenka", "gauff": "Coco Gauff"}
_NBA = {"lakers": "Los Angeles Lakers", "warriors": "Golden State Warriors", "celtics": "Boston Celtics",
        "lebron": "LeBron James", "curry": "Stephen Curry"}
_FINANCE = {"apple|aapl": "Apple", "tesla|tsla": "Tesla", "nvidia|nvda": "Nvidia", "microsoft|msft": "Microsoft",
            "amazon|amzn": "Amazon", "google|alphabet": "Alphabet", "meta": "Meta", "reliance": "Reliance Industries",
            "tcs": "TCS", "infosys|infy": "Infosys", "hdfc": "HDFC Bank", "zomato": "Zomato",
            "sensex": "Sensex", "nifty": "Nifty 50", "nasdaq": "Nasdaq", "dow|dow jones": "Dow Jones",
            "bitcoin|btc": "Bitcoin", "ethereum|eth": "Ethereum", "gold": "gold", "silver": "silver",
            "oil|crude": "crude oil"}
_PLACES = ("delhi", "new delhi", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata", "hyderabad", "pune",
           "kochi", "goa", "jaipur", "ahmedabad", "lucknow", "mysore", "manali", "shimla", "ooty", "trivandrum",
           "london", "paris", "new york", "tokyo", "dubai", "singapore", "sydney", "melbourne", "berlin",
           "mexico", "mexico city", "monaco", "silverstone", "monza", "spa", "suzuka", "baku", "las vegas", "miami",
           "abu dhabi", "qatar", "brazil", "interlagos", "austin", "imola", "zandvoort", "singapore")
_AMBIGUOUS = {"max", "george", "charles", "oscar", "lewis", "carlos", "alex", "lance", "nico", "kimi", "ollie",
              "oliver", "sergio", "fernando", "pierre", "franco", "liam", "gabriel", "valtteri", "arvid", "isack",
              "esteban", "aston", "city", "united", "madrid", "india", "australia", "england", "pakistan",
              "mi", "rr", "root", "smith", "curry", "rohit", "surya", "apple", "amazon", "meta", "gold", "silver",
              "oil", "eth", "dow", "spa", "mexico", "brazil", "qatar", "miami", "austin", "monaco"}


def _lexicon() -> dict[str, tuple[str, str]]:
    table = {}
    for domain, names in (("f1", _F1), ("cricket", _CRICKET), ("football", _FOOTBALL), ("tennis", _TENNIS),
                          ("basketball", _NBA), ("finance", _FINANCE)):
        for aliases, canonical in names.items():
            for alias in aliases.split("|"):
                table.setdefault(alias, (canonical, domain))
    return table


LEXICON = _lexicon()
_ALIAS = re.compile(r"\b(" + "|".join(sorted(map(re.escape, LEXICON), key=len, reverse=True)) + r")\b")
_PLACE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, set(_PLACES)), key=len, reverse=True)) + r")\b")

# Words that point at a domain without naming anyone in it.
_CUES = {
    "f1": r"f1|formula (?:1|one)|grand prix|gp|qualif\w*|quali|pole|q[123]|sprint|pit(?:ted|stop)?s?|grid|"
          r"podium|laps?|constructors?|drivers'? championship|fp[123]|practice|race|races|racing",
    "cricket": r"cricket|ipl|odis?|t20s?|test match|the test|innings|wickets?|runs|century|hundred|batting|"
               r"bowling|toss|bowled",
    "football": r"football|soccer|premier league|champions league|la liga|goals?|clasico|derby",
    "tennis": r"tennis|wimbledon|us open|french open|australian open|grand slam",
    "basketball": r"nba|basketball",
    "motogp": r"motogp|moto gp",
    "weather": r"weather|rain\w*|forecast|temperature|hot|cold|warm|cool(?:er)?|humid\w*|umbrella|jacket|sunny|"
               r"cloudy|wind\w*|storm\w*|snow\w*|aqi|air quality|uv|monsoon|degrees?",
    "finance": r"stocks?|shares?|share price|markets?|trading|exchange rate|rupees?|dollars?|euros?|usd|inr|"
               r"crypto|price of|prices?|fed|interest rates?",
    "news": r"news|headlines?|breaking|top stories|trending|election\w*",
}
_CUE = {domain: re.compile(r"\b(?:" + pattern + r")\b") for domain, pattern in _CUES.items()}

# What a sports question asks for, most specific first. "Where" never decides it.
_SPORT_FOCUS = (
    ("qualifying position", r"qualif\w*|quali|pole|q[123]"),
    ("starting position", r"start(?:ed|ing)? from|grid position|on the grid"),
    ("sprint result", r"sprint"),
    ("practice times", r"practice|fp[123]"),
    ("pit stops", r"pit(?:ted|stop)?s?"),
    ("match status", r"already (?:started|begun|over|finished)|started yet|still on|over yet|finished yet|"
                     r"live now|is it on|still going|still playing"),
    ("start time", r"what time|start(?:s|ing)?\b|begin(?:s|ning)?\b|kick ?off|first ball|toss time"),
    ("schedule", r"next (?:race|match|game|grand prix|gp|odi|test)|when(?:'s| is| are| does| do)|what time|"
                 r"fixtures?|schedule|start\b"),
    ("standings", r"standings|championship|points table|table|leading|top of|how many points|title"),
    ("score", r"scores?|scored|runs|how many did|wickets?|innings|batting|bowling|hundred|century"),
    ("result", r"finish\w*|won|wins?|beat|podium|results?|end(?:ed)? up|placed?|came|lost|lose|race|"
               r"how did .* (?:do|get on|go)|how(?:'s| is) .* doing|position"),
)
_SPORT_FOCUS = tuple((name, re.compile(r"\b(?:" + pattern + r")")) for name, pattern in _SPORT_FOCUS)
_RESULT_WORDS = re.compile(r"\b(?:finish\w*|won|wins?|beat|podium|results?|race|lost|lose|end(?:ed)? up|came)\b")
_TIMES = (
    ("now", r"right now|now|currently|at the moment"),
    ("yesterday", r"yesterday(?:'s)?|last night"),
    ("tomorrow", r"tomorrow(?:'s)?"),
    ("today", r"today(?:'s)?|tonight|this (?:morning|afternoon|evening)"),
    ("this weekend", r"(?:this |the )?weekend"),
    ("this week", r"this week"),
    ("last week", r"last week"),
    ("next upcoming", r"next|upcoming|when(?:'s| is| does| do)"),
    ("latest", r"latest|last|most recent|recent"),
)
_TIMES = tuple((name, re.compile(r"\b(?:" + pattern + r")\b")) for name, pattern in _TIMES)
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_PAST = re.compile(r"\b(did|was|were|qualified|finished|won|lost|beat|got|came|ended|scored|took|made)\b")
_PRONOUN = re.compile(r"\b(he|she|him|her|his|hers|they|them|their)\b")
_FOLLOW = re.compile(r"^(?:(?:ok(?:ay)?|thanks|thank you|cool|nice|right|hmm)[,!. ]+)?"
                     r"(?:and|what about|how about|what of|same for|also)\b")
_CORRECTION = re.compile(r"^\W*(?:no|nope|nah|not that|wrong)?[,.!]?\s*(?:i meant(?! to\b)|i mean(?! to\b)|"
                         r"i was asking|i asked (?:about|for)|that'?s not what i (?:asked|meant|wanted)|"
                         r"not what i asked|i wanted)\b|^\W*(?:no|nope|nah)[,.!]?\s+(?:the|his|her|what|which)\b")
# A statement or an opinion has a verb; a typed lookup like "price of gold" often has none.
_VERB = re.compile(r"\b(?:is|are|was|were|am|be|i|i'm|i'd|i'll|you|we|think|love|like|hate|hope|bet|want|"
                   r"watch\w*|feel\w*|got|get|go|going|went|need|tell|let|make|do|does|did|have|has|had)\b")
_DEFINITION = re.compile(r"^(?:what(?:'s| is| are) (?:a|an)\b|what does .+ (?:mean|stand for)|"
                         r"how (?:does|do) .+ work\b|how do (?:they|you) (?:decide|score|choose)|"
                         r"explain (?:what|how)\b)")
_REFERS_BACK = re.compile(r"\b(?:that|this|the same) (?:series|match|game|race|tournament|test|one|session|"
                          r"team|driver|player|stock|share|city|place)\b")
_ASPECT = re.compile(r"\b(?:where|who won|winning|score|result|price|how much|how many|how hot|how cold|"
                     r"forecast|rain)\b")
# A subject of its own: "what time is it?", "how are you?", "do you like it?".
_OWN_SUBJECT = re.compile(r"\b(?:is|are|was|were|am|do|does|did|will|can|have|has) (?:it|you|i|we|that|this|there)\b|"
                          r"\b(?:you|your|i|i'm|me|my|it's)\b")
_QUESTION = re.compile(r"\?\s*$|^(?:who|what|where|when|why|how|which|is|are|did|does|do|will|can|should)\b")
_NEGATED = re.compile(r"\bnot (?:the |his |her |in )?[a-z0-9]+")
_PLACE_AFTER = re.compile(r"\b(?:in|at|for)\s+([a-z][a-z'-]+(?:\s+city)?)")
_NOT_PLACES = {"the", "a", "an", "qualifying", "quali", "practice", "q1", "q2", "q3", "sprint", "race", "match",
               "test", "today", "tomorrow", "yesterday", "tonight", "now", "me", "it", "him", "her", "them",
               "this", "that", "my", "his", "their", "rupees", "dollars", "ten", "general", "points", "india's"}


def normalize(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").split())


_ADDRESS = re.compile(r"[,\s]*\b(?:kip|bud|buddy|mate|man|bro|dude|pal)\b(?=[\s?!.,]*$)")
_FILLER = re.compile(r"^\W*(?:(?:ah+|oh+|aw|shit|damn|dang|wow|hmm+|ok(?:ay)?|so|wait|well|right|yeah|haha|"
                     r"hey|kip)\b[\s,!.]*)+")


def bare(text: str) -> str:
    """The message without who it is said to or leading exclamations: "what time kip?" ->
    "what time?", "ah shit already started?" -> "already started?"."""
    core = _FILLER.sub("", _ADDRESS.sub("", text.strip()))
    return core if core.strip(" ?!.,") else text


_CLOCK = re.compile(r"^(?:what(?:'s|s| is) the time(?: now)?|what time is it(?: now)?|what time(?: now)?|"
                    r"time(?: now| please)?|got the time|do you (?:know|have) the time)\W*$")


def clock_question(text: str) -> bool:
    """ "What time is it?", "what's the time, kip?". The caller checks first that it is not a
    follow-up: "what time?" right after a question about a match asks about the match."""
    return bool(_CLOCK.match(normalize(bare(text))))


def asking_clause(text: str) -> str:
    """The last clause that asks something, else the whole message."""
    clauses = [c for c in re.split(r"(?<=[.;!?])\s+", text.strip()) if c.strip()]
    if len(clauses) < 2:
        return text
    asking = [c for c in clauses if _QUESTION.search(normalize(c))]
    return asking[-1] if asking else text


@dataclass
class Slots:
    """What the words themselves say, before any context."""
    domains: list[str] = field(default_factory=list)
    entities: list[tuple[str, str]] = field(default_factory=list)   # (canonical, domain)
    ambiguous: list[tuple[str, str]] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    sport_focus: str | None = None
    time: str | None = None
    year: str | None = None
    pronoun: bool = False
    follow_cue: bool = False
    past: bool = False
    generic_position: bool = False   # "what position", with nothing saying which one
    words: list[str] = field(default_factory=list)
    order: dict[str, int] = field(default_factory=dict)   # canonical name -> where it was said


def read_slots(text: str) -> Slots:
    lowered = normalize(text)
    slots = Slots(pronoun=bool(_PRONOUN.search(lowered)), follow_cue=bool(_FOLLOW.search(lowered)),
                  past=bool(_PAST.search(lowered)), words=re.findall(r"[a-z0-9]+(?:'[a-z]+)?", lowered))
    slots.domains = [domain for domain, cue in _CUE.items() if cue.search(lowered)]
    for match in _ALIAS.finditer(lowered):
        alias = match.group(1)
        canonical, domain = LEXICON[alias]
        target = slots.ambiguous if alias in _AMBIGUOUS else slots.entities
        slots.order.setdefault(canonical, match.start())
        if (canonical, domain) not in target:
            target.append((canonical, domain))
    places = [m.group(1) for m in _PLACE.finditer(lowered)]
    places += [m.group(1).strip() for m in _PLACE_AFTER.finditer(lowered)
               if m.group(1).split()[0] not in _NOT_PLACES and m.group(1) not in LEXICON]
    slots.places = list(dict.fromkeys(p.title().replace("'S", "'s") for p in places))
    focus_text = _NEGATED.sub(" ", lowered)  # "the race, not qualifying"
    slots.sport_focus = next((name for name, pattern in _SPORT_FOCUS if pattern.search(focus_text)), None)
    slots.generic_position = slots.sport_focus == "result" and not _RESULT_WORDS.search(focus_text)
    slots.time = next((name for name, pattern in _TIMES if pattern.search(lowered)), None)
    year = _YEAR.search(lowered)
    slots.year = year.group(1) if year else None
    return slots


# ── features and classifier ───────────────────────────────────────────────────
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[?!]")
_MASKS = {"f1": "xdriver", "cricket": "xcricketer", "football": "xclub", "tennis": "xplayer",
          "basketball": "xplayer", "finance": "xticker"}


def features(text: str) -> list[str]:
    """Words, word pairs, character n-grams and domain tags. Names in the lexicon become their
    kind ("xdriver"), so one driver's question teaches the classifier about every driver."""
    lowered = normalize(text)
    masked = _ALIAS.sub(lambda m: _MASKS[LEXICON[m.group(1)][1]], lowered)
    masked = _PLACE.sub("xplace", masked)
    words = _WORD.findall(masked)
    out = [f"w:{w}" for w in words]
    out += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    for word in words:
        padded = f"<{word}>"
        out += [f"c:{padded[i:i + n]}" for n in (3, 4) for i in range(len(padded) - n + 1)]
    out += [f"d:{domain}" for domain, cue in _CUE.items() if cue.search(lowered)]
    if _PRONOUN.search(lowered):
        out.append("t:pronoun")
    if re.search(r"\b(i|i'm|i've|my|me)\b", lowered):
        out.append("t:first_person")
    if _QUESTION.search(lowered):
        out.append("t:question")
    return out


class TfidfLogistic:
    """TF-IDF over features(), multinomial logistic regression fitted by gradient descent."""

    def __init__(self, texts: list[str], labels: list[str], epochs: int = 300, rate: float = 4.0,
                 l2: float = 1e-4) -> None:
        documents = [features(t) for t in texts]
        frequency: dict[str, int] = {}
        for doc in documents:
            for token in set(doc):
                frequency[token] = frequency.get(token, 0) + 1
        # Character n-grams seen once are mostly noise; words and tags are kept.
        vocab = sorted(t for t, n in frequency.items() if n > 1 or not t.startswith("c:"))
        self.index = {token: i for i, token in enumerate(vocab)}
        self.idf = np.array([math.log((1 + len(documents)) / (1 + frequency[t])) + 1 for t in vocab], np.float32)
        self.labels = sorted(set(labels))
        X = self._matrix(documents)
        Y = np.zeros((len(labels), len(self.labels)), np.float32)
        Y[np.arange(len(labels)), [self.labels.index(label) for label in labels]] = 1
        self.W = np.zeros((X.shape[1], len(self.labels)), np.float32)
        self.b = np.zeros(len(self.labels), np.float32)
        for _ in range(epochs):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) / len(labels)
            self.W -= rate * (X.T @ G + l2 * self.W)
            self.b -= rate * G.sum(axis=0)

    def _matrix(self, documents: list[list[str]]) -> np.ndarray:
        X = np.zeros((len(documents), len(self.index)), np.float32)
        for row, doc in enumerate(documents):
            for token in doc:
                column = self.index.get(token)
                if column is not None:
                    X[row, column] += 1
        X = np.log1p(X) * self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-9)

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predict(self, text: str) -> dict[str, float]:
        P = self._softmax(self._matrix([features(text)]) @ self.W + self.b)[0]
        return dict(zip(self.labels, P.tolist()))


class EmbeddingLogistic:
    """The same regression over sentence embeddings (the bge-small model memory recall already
    loads). Better at wording it has never seen; it needs the model, so it is optional."""

    def __init__(self, vectors: np.ndarray, labels: list[str], epochs: int = 400, rate: float = 8.0,
                 l2: float = 1e-4) -> None:
        self.labels = sorted(set(labels))
        Y = np.zeros((len(labels), len(self.labels)), np.float32)
        Y[np.arange(len(labels)), [self.labels.index(label) for label in labels]] = 1
        self.W = np.zeros((vectors.shape[1], len(self.labels)), np.float32)
        self.b = np.zeros(len(self.labels), np.float32)
        for _ in range(epochs):
            P = TfidfLogistic._softmax(vectors @ self.W + self.b)
            G = (P - Y) / len(labels)
            self.W -= rate * (vectors.T @ G + l2 * self.W)
            self.b -= rate * G.sum(axis=0)

    def predict(self, vector: np.ndarray) -> dict[str, float]:
        P = TfidfLogistic._softmax((vector @ self.W + self.b)[None, :])[0]
        return dict(zip(self.labels, P.tolist()))


def training_rows(path: Path = DATA) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            label, text = line.split("\t", 1)
            rows.append((label.strip(), text.strip()))
    return rows


# ── routes and context ────────────────────────────────────────────────────────
@dataclass
class Route:
    intent: str | None
    confidence: float
    domain: str | None = None
    focus: str | None = None
    entities: list[str] = field(default_factory=list)
    place: str | None = None
    time: str | None = None
    capability: str | None = None    # web_search | memory | house | None
    query: str = ""                  # what to search for: the resolved meaning, not the raw words
    meaning: str = ""                # one line for the model answering it
    source: str = "classifier"       # classifier | context | correction
    followup: bool = False
    correction: bool = False
    earlier: str = ""                # the question a follow-up or correction leans on
    asked: str = ""                  # a follow-up's own question, when it asks something new
    clarify: str | None = None       # a short question back, when a pronoun has nobody to point at
    scores: dict = field(default_factory=dict)
    evidence: bool = False           # a domain word, name or place in the message itself
    lookup_phrase: bool = False      # "ipl points table", "price of gold": a lookup with no verb

    @property
    def live_confidence(self) -> float:
        """How sure the router is that this needs current information at all, whichever
        kind: the classes share that one decision, so their probabilities add up."""
        if self.source != "classifier":
            return self.confidence
        return sum(p for label, p in self.scores.items() if label in LIVE)

    @property
    def live(self) -> bool:
        # A split vote with nothing in the words to back it ("how's it going") is no lookup.
        backed = self.confidence >= THRESHOLD or self.evidence
        return self.intent in LIVE and self.live_confidence >= THRESHOLD and backed and not self.clarify

    def public(self) -> dict:
        data = asdict(self)
        data["scores"] = {k: round(v, 3) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])[:3]}
        data["confidence"] = round(self.confidence, 3)
        return data


@dataclass
class Frame:
    """The last information request: what a follow-up leans on."""
    intent: str
    domain: str | None
    focus: str | None
    entities: list[str]
    place: str | None
    time: str | None
    said: str
    turn: int


class IntentRouter:
    """TF-IDF alone at first; once add_embeddings() has run (in the background, never during a
    reply), the average of both models, which generalizes better to new wording."""

    def __init__(self, extra: list[tuple[str, str]] = (), embedder=None) -> None:
        self.rows = training_rows() + list(extra)
        self.model = TfidfLogistic([text for _, text in self.rows], [label for label, _ in self.rows])
        self.version = hashlib.sha1(json.dumps(self.rows).encode()).hexdigest()[:10]
        self.embedder, self.embedding = embedder, None

    def add_embeddings(self) -> bool:
        """Fit the embedding model. Blocking (a second or two): call it from a worker thread."""
        if self.embedder is None or self.embedding is not None:
            return self.embedding is not None
        vectors = self.embedder.embed_sync([text for _, text in self.rows])
        if not vectors:
            return False
        self.embedding = EmbeddingLogistic(np.array(vectors, np.float32), [label for label, _ in self.rows])
        return True

    @property
    def backend(self) -> str:
        return "tfidf+embeddings" if self.embedding is not None else "tfidf"

    def scores(self, text: str) -> dict[str, float]:
        scores = self.model.predict(text)
        if self.embedding is not None:
            vector = self.embedder.embed_sync([text])
            if vector:
                other = self.embedding.predict(np.array(vector[0], np.float32))
                scores = {label: (p + other[label]) / 2 for label, p in scores.items()}
        return scores

    def route(self, text: str, frame: Frame | None = None, turn: int = 0, today: date | None = None) -> Route:
        today = today or date.today()
        # "Don't read any webpages; what's the AQI?": the permission is handled elsewhere,
        # so the clause that asks something is what gets read.
        text = asking_clause(text)
        # "what time kip?", "ah shit already started?": the words that carry meaning.
        core = bare(text)
        slots = read_slots(core)
        scores = self.scores(text)
        intent = max(scores, key=scores.get)
        confidence = scores[intent]
        fresh = frame if frame is not None and turn - frame.turn <= FRAME_TURNS else None
        route = Route(intent=intent, confidence=confidence, scores=scores)
        lowered = normalize(core)
        route.evidence = bool(slots.domains or slots.entities or slots.places)
        route.lookup_phrase = len(lowered.split()) <= 6 and not _VERB.search(lowered)
        if fresh and _CORRECTION.search(lowered):
            route = self._resolve(fresh, slots, scores, "correction")
        elif fresh and self._leans_on(fresh, slots, intent, confidence, lowered, read_intent(text).greeting):
            route = self._resolve(fresh, slots, scores, "context")
        elif _DEFINITION.search(lowered) and not slots.time:
            # "What is a grand prix?", "how does qualifying work?": knowledge, not today's data.
            route.intent, route.confidence, route.source = "information.general", max(confidence, THRESHOLD), "rules"
        else:
            if intent in LIVE and _QUESTION.search(lowered) and is_live_fact(lowered) \
                    and route.live_confidence < THRESHOLD:
                # An information question in the words that always mean live data (AQI, forecast,
                # share price): a rule, not a vote. Small talk ("do you like cricket?") never gets here.
                cue = next((d for d in slots.domains if d in INTENT_OF_DOMAIN), None)
                route.intent = intent if intent in LIVE else INTENT_OF_DOMAIN.get(cue, intent)
                route.confidence, route.source = max(confidence, THRESHOLD), "rules"
            self._fill(route, slots, lowered)
            # "How did he do?" with nobody to point at: ask, never guess who.
            short = slots.follow_cue or len(lowered.split()) <= 4
            if slots.pronoun and not route.entities and (
                    (route.intent in LIVE and route.live_confidence >= THRESHOLD) or (confidence < THRESHOLD and short)):
                route.clarify = "Who you mean, question?"
        route.capability = _capability(route)
        if route.intent in LIVE:
            route.query, route.meaning = _query(route, text, today)
        return route

    def _leans_on(self, frame: Frame, slots: Slots, intent: str, confidence: float, lowered: str,
                  greeting: bool = False) -> bool:
        """A short message that only makes sense with the last request: "and Max?", "how did he
        finish?", "and yesterday?", "what time?". A message about something else stands on its own."""
        own = [d for d in slots.domains if INTENT_OF_DOMAIN.get(d) not in (None, frame.intent)]
        own += [d for _, d in slots.entities if INTENT_OF_DOMAIN.get(d) not in (None, frame.intent)]
        refers = bool(_REFERS_BACK.search(lowered))   # "...in that series?"
        if own or (len(lowered.split()) > 9 and not refers):
            return False
        if slots.follow_cue or refers:
            return True
        # A question too short to have a subject of its own: "what time?", "who won?", "already
        # started?". Never a greeting or a question about something else ("what time is it?").
        asks_aspect = bool(slots.sport_focus or slots.time or _ASPECT.search(lowered))   # never "whats up?"
        elliptic = (len(lowered.split()) <= 4 and _QUESTION.search(lowered) and not _OWN_SUBJECT.search(lowered)
                    and asks_aspect and not (slots.entities or slots.ambiguous or slots.places or slots.domains
                                             or greeting))
        if elliptic and not (intent.startswith("conversation.") and confidence >= THRESHOLD):
            return True
        bare_time = slots.time and len(lowered.split()) <= 3   # "yesterday?"
        if intent not in LIVE and confidence >= THRESHOLD and not (slots.sport_focus or bare_time):
            return False  # "he was rough on me today": about something else, pronoun or not
        if slots.pronoun:  # "how did he finish?", never "he was rough on me"
            return bool(_QUESTION.search(lowered) or slots.sport_focus)
        # A bare time or name: "yesterday?", "Max?"
        return len(lowered.split()) <= 3 and bool(slots.time or slots.ambiguous or slots.entities or slots.places)

    def _resolve(self, frame: Frame, slots: Slots, scores: dict, source: str) -> Route:
        route = Route(intent=frame.intent, confidence=max(0.9, scores.get(frame.intent, 0)), domain=frame.domain,
                      focus=frame.focus, entities=list(frame.entities), place=frame.place, time=frame.time,
                      source=source, followup=source == "context", correction=source == "correction", scores=scores,
                      evidence=True, earlier=frame.said)
        named = sorted((c for c, d in slots.entities + slots.ambiguous
                        if d == frame.domain or (frame.domain in SPORTS and d in SPORTS)),
                       key=lambda c: slots.order.get(c, 0))
        if named:
            route.entities = named
        if frame.intent == "information.weather" and slots.places:
            route.place, route.entities = slots.places[0], [slots.places[0]]
        elif frame.intent == "information.sports" and slots.places:
            route.place = slots.places[0]
        if slots.time:
            route.time = slots.time
        if frame.intent == "information.sports" and slots.sport_focus:
            # "I meant what position" keeps the position the earlier question was about.
            if not (slots.generic_position and frame.focus and "position" in frame.focus):
                route.focus = _sport_focus(slots.sport_focus, route.domain)
        elif source == "context" and not (slots.time or named or slots.places) and len(slots.words) >= 5:
            # A new question about the same subject ("is there a fifth match in that series?"):
            # the subject carries over, the earlier question's aspect does not.
            route.focus, route.asked = None, " ".join(slots.words)
        if frame.intent == "information.weather" and slots.time:
            route.focus = _weather_focus(route.time, None)
        if route.domain in SPORTS:
            route.time = _sport_time(route.time, route.focus, True)
        return route

    def _fill(self, route: Route, slots: Slots, lowered: str) -> None:
        intent = route.intent
        named = slots.entities + [(c, d) for c, d in slots.ambiguous
                                  if d in slots.domains or (intent == "information.sports" and d in SPORTS)
                                  or (intent == "information.finance" and d == "finance")
                                  or any(d == d2 for _, d2 in slots.entities)]
        named.sort(key=lambda item: slots.order.get(item[0], 0))
        if intent == "information.sports":
            sport = [d for d in slots.domains if d in SPORTS] + [d for _, d in named if d in SPORTS]
            # A race, a grid or a driver says Formula 1; a match alone says nothing yet.
            route.domain = _most_common(sport)
            route.entities = [c for c, d in named if d in SPORTS]
            route.place = slots.places[0] if slots.places else None
            route.focus = _sport_focus(slots.sport_focus, route.domain)
            route.time = _sport_time(slots.time or slots.year, route.focus, slots.past)
        elif intent == "information.weather":
            route.domain = "weather"
            route.place = slots.places[0] if slots.places else None
            route.entities = [route.place] if route.place else []
            route.time = slots.time if slots.time not in (None, "latest", "next upcoming") else "now"
            route.focus = _weather_focus(route.time, lowered)
            if route.focus == "forecast" and not slots.time:
                route.time = "upcoming"
        elif intent == "information.finance":
            route.domain = "finance"
            route.entities = [c for c, d in named if d == "finance"]
            route.time = slots.time if slots.time not in (None, "next upcoming") else "now"
            if re.search(r"\b(price|trading|at|worth|how much|close[ds]?|closing|rate|share|stock|cost)\b", lowered):
                route.focus = "price"
        elif intent == "information.news":
            route.domain = "news"
            route.time = slots.time or "today"


def _most_common(items: list[str]) -> str | None:
    return max(set(items), key=items.count) if items else None


def _sport_focus(focus: str | None, domain: str | None) -> str | None:
    if focus != "result":
        return focus
    return "finishing position" if domain in ("f1", "motogp") else "match result"


def _sport_time(time: str | None, focus: str | None, past: bool) -> str | None:
    if focus == "match status":
        return time if time not in (None, "latest", "next upcoming") else "now"
    if focus == "start time":
        return time if time not in (None, "latest", "next upcoming") else "today or next"
    if focus == "schedule":
        return time if time not in (None, "latest") else "next upcoming"
    if time in (None, "latest"):
        return "latest completed" if past or focus else time
    return time


def _weather_focus(time: str | None, lowered: str | None) -> str:
    if time == "yesterday" or (lowered and re.search(r"\bwas\b", lowered)):
        return "past weather"
    if time in ("tomorrow", "this weekend", "this week", "upcoming") or (
            lowered and re.search(r"\b(forecast|will|going to|later|tonight|this evening)\b", lowered)):
        return "forecast"
    return "current conditions"


def _capability(route: Route) -> str | None:
    if route.intent in LIVE:
        return "web_search" if route.live_confidence >= THRESHOLD else None
    if route.confidence < THRESHOLD or not route.intent:
        return None
    return {"memory.recall": "memory", "memory.update": "memory", "action.request": "house"}.get(route.intent)


def _query(route: Route, said: str, today: date) -> tuple[str, str]:
    """(search query, meaning): built from the resolved intent, not the question's wording."""
    domain = DOMAIN_NAMES.get(route.domain or "", "")
    # Two teams or players in one question are a fixture: "India vs New Zealand".
    joiner = " vs " if route.domain in SPORTS - {"f1", "motogp"} and len(route.entities) == 2 else ", "
    who = joiner.join(route.entities)
    day = today.strftime("%d %B %Y").lstrip("0")
    when = {"latest completed": f"the latest completed one ({today.year})", "next upcoming": "the next one",
            "now": f"right now ({day})", "upcoming": "the coming hours",
            "today or next": f"the match on today ({day}), or else the next one"}.get(route.time or "",
                                                                                     route.time or "")
    if route.intent == "information.sports":
        event = f"{route.place} " if route.place else ""
        event += "Grand Prix" if route.domain == "f1" else ("race" if route.domain == "motogp" else "")
        # A new question about the same subject searches its own words, about that subject.
        aspect = route.focus or (f"“{route.asked}”" if route.asked else "result")
        timed = route.focus in ("start time", "match status") or route.time in ("now", "today or next")
        query = " ".join(filter(None, [who, route.focus or route.asked or "result", event.strip(), domain,
                                       day if timed else str(today.year)
                                       if route.time in ("latest completed", "next upcoming", None) else route.time]))
        place_note = " (a position, not a place)" if route.focus and "position" in route.focus else ""
        if route.focus == "start time":
            place_note = " (a clock time and day: just that, not the whole series)"
        meaning = f"{domain or 'Sport'}: {(who + ' — ') if who else ''}{aspect}{place_note}"
        meaning += f" at the {event.strip()}" if event.strip() else ""
        meaning += f", {when}." if when else "."
    elif route.intent == "information.weather":
        aspect = route.focus or "current conditions"
        place = route.place or ""
        query = " ".join(filter(None, ["weather", "forecast" if aspect == "forecast" else "", place,
                                       "yesterday" if aspect == "past weather" else (route.time or "")]))
        meaning = f"Weather: {aspect}{' in ' + place if place else ' (no place given)'}, {when or 'right now'}."
    elif route.intent == "information.finance":
        query = " ".join(filter(None, [who or "", route.focus or "", normalize(said) if not who else "",
                                       route.time if route.time != "now" else "today"]))
        meaning = f"Markets: {who or 'what they asked about'} — {route.focus or 'latest move'}, {when or 'right now'}."
    else:
        query = " ".join(filter(None, [normalize(said), today.strftime("%d %B %Y")]))
        meaning = f"News: {normalize(said)}, {when or 'today'}."
    return " ".join(query.split())[:200], meaning


def frame_of(route: Route, said: str, turn: int) -> Frame | None:
    """What a later follow-up may lean on: only a confident information request."""
    if not route.live:
        return None
    return Frame(intent=route.intent, domain=route.domain, focus=route.focus, entities=list(route.entities),
                 place=route.place, time=route.time, said=said, turn=turn)


def fast_path(route: Route, intent, *, web_allowed: bool, involves_house: bool) -> bool:
    """Whether a live-information question may go straight to the web lookup. Every gate the
    usual path has comes first: stop, permissions, offers, requested tools and the house."""
    if intent.cancel or intent.wants_offer or intent.needs_tools:
        return False
    # Only a permission change or a refusal, with nothing asked: no lookup. "No, I meant the
    # race" is a correction, not a refusal; ruled-out tools are checked by the caller.
    if intent.changes_state_only and not (route.correction and not (intent.granted or intent.revoked)):
        return False
    if not web_allowed or involves_house or intent.about_past:
        return False
    if not (intent.question or intent.request or route.followup or route.correction or route.lookup_phrase):
        return False
    return route.live


_ROUTER: IntentRouter | None = None


def default_router() -> IntentRouter:
    """The TF-IDF router on the shipped examples, trained once per process (a fraction of a
    second). The app builds its own with reviewed corrections and the embedding model."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = IntentRouter()
    return _ROUTER
