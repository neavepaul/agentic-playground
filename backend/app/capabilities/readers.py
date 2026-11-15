"""Feed questions read into each feed's form by classifiers trained on example sentences, the way
the intent router reads intents: a misreading is fixed by adding an example to
reader_data/<domain>.tsv, not by writing a rule.

A form file is tab-separated: a header naming the fields, then one example per line, its label for
each field ("-" when a field does not apply) and last the sentence. Names in a sentence are put in
[brackets]. Names the caller knows by kind (a team, a driver, an index) are given as masks, so one
example teaches the reader about all of them.

Every word an example uses outside brackets is a word of the form. When a question is read, a word
a letter or two away from a form word is that word ("champoinship"), and the words left over are
what it names ("do I have any tata motors"): the feed checks that against its own data (the
holdings, the standings table). Left-over words are not hidden from the classifiers, so an
ordinary word the examples never used ("what's my top holding") cannot pass for a name.

A question with a word the examples never use, beyond the one name it may ask about, is one the
reader has not been taught ("are Russell and Verstappen beefing?"): it is unsure, and the web
lookup answers. Teaching it is adding an example that uses the word.

Nothing here calls a model or the network: a question is read in well under a millisecond.
"""
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).parent / "reader_data"
# Below this on the form's first field the reader has no opinion, and the web lookup answers.
SURE = 0.4  # a reader's default; one with many answers to choose from may need more
_BRACKET = re.compile(r"\[([^\]]+)\]")
_TOKEN = re.compile(r"[a-z0-9&]+(?:'[a-z]+)?|[^\sa-z0-9]")
# Words that ask nothing by themselves: never a sign of an untaught question.
_FILLER = frozenset("a an the please pls can could would you tell me know let us just so hey hi ok okay "
                    "and or of to in on at for i me my it is are was do does did".split())


def _stem(word: str) -> str:
    """A word without its ending: "leads", "leading" and "leaded" alike; "leaving" is not "leading"."""
    word = word.removesuffix("'s")
    for ending in ("ing", "ed", "es", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[:-len(ending)]
    return word


@dataclass
class Reading:
    form: dict[str, str]         # each field's label
    sure: dict[str, float]       # how sure the reader is of each
    main: str                    # the field that decides whether the reading is used
    name: str | None = None      # the first run of left-over words, as said ("tata motors")
    known: list[str] = field(default_factory=list)  # names recognised by kind, as said ("nifty")
    untaught: list[str] = field(default_factory=list)  # words no example uses, beyond the name
    cutoff: float = SURE         # how sure the main field must be for the reading to be used

    @property
    def confident(self) -> bool:
        return self.sure[self.main] >= self.cutoff and not self.untaught


class Reader:
    def __init__(self, domain: str, masks: dict[str, str] | None = None, name_mask: str = "xname",
                 unknown_names: bool = True, sure: float = SURE) -> None:
        """masks: alias -> placeholder for names known by kind. name_mask: what a [bracketed] name
        in an example becomes; the router's own placeholder for the kind of name the domain asks
        about ("xticker"), so an example name and a name the router knows read alike.
        unknown_names: whether a name may be a word no list knows (any company); False when every
        name is in masks (drivers, teams, places), so any left-over word is untaught."""
        self.unknown_names, self.name_mask, self.cutoff = unknown_names, name_mask, sure
        from app.chat.router import TfidfLogistic, bare, normalize  # here: app.chat imports the feeds
        self._normalize, self._bare = normalize, bare
        self.masks = {alias.casefold(): mask for alias, mask in (masks or {}).items()}
        self._mask = re.compile(r"(?<![a-z0-9])(" + "|".join(sorted(map(re.escape, self.masks), key=len, reverse=True))
                                + r")(?![a-z0-9])") if self.masks else None
        lines = [line for line in (DATA / f"{domain}.tsv").read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
        header, *rows = [line.split("\t") for line in lines]
        self.fields = [f.strip() for f in header[:-1]]
        texts = [row[-1].strip() for row in rows]
        self.vocab = {w for text in texts for w in self._tokens(self._masked(_BRACKET.sub(" ", text))[0])}
        self.stems = {_stem(w): w for w in sorted(self.vocab)}
        self.placeholders = set(self.masks.values()) | {name_mask}
        prepared = [self._masked(_BRACKET.sub(name_mask, text))[0] for text in texts]
        self.models = {}
        for i, name in enumerate(self.fields):
            pairs = [(text, row[i].strip()) for text, row in zip(prepared, rows) if row[i].strip() not in ("", "-")]
            self.models[name] = TfidfLogistic([t for t, _ in pairs], [label for _, label in pairs])

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN.findall(text)

    def _masked(self, text: str) -> tuple[str, list[str]]:
        """The text with names known by kind replaced, and those names as said."""
        text, found = self._normalize(text), []
        if self._mask is None:
            return text, found

        def swap(match: re.Match) -> str:
            found.append(match.group(1))
            return self.masks[match.group(1)]
        return self._mask.sub(swap, text), found

    def _known(self, word: str) -> str | None:
        """The form word this is, allowing for another form of it ("leads" for "leading") or a typo
        in a longer word; None for a name. Numbers (a year, an amount) are part of the form."""
        if word in self.vocab or word in self.placeholders or not word[0].isalnum() or word.isdigit():
            return word
        if _stem(word) in self.stems:
            return self.stems[_stem(word)]
        # 0.88 keeps a real word from passing for its neighbour ("leaving" is not "leading").
        close = difflib.get_close_matches(word, self.vocab, n=1, cutoff=0.88) if len(word) >= 5 else []
        return close[0] if close else None

    def _misspelt_name(self, word: str) -> str | None:
        """The one-word name known by kind this is a typo of ("russel" for Russell), if any."""
        if len(word) < 5:
            return None
        close = difflib.get_close_matches(word, [a for a in self.masks if " " not in a], n=1, cutoff=0.88)
        return close[0] if close else None

    def read(self, said: str) -> Reading:
        text, known = self._masked(self._bare(said))  # "hey kip, …" is not a name
        out, masked, run, names = [], [], [], []
        for token in self._tokens(text) + [""]:
            word = self._known(token) if token else ""
            alias = self._misspelt_name(token) if word is None else None
            if alias:
                known.append(alias)
                word = self.masks[alias]
            if word is None:
                run.append(token)
                out.append(token)
                continue
            if run:
                names.append(" ".join(run))
                masked.append(self.name_mask if len(names) == 1 else names[-1])
                run = []
            if token:
                out.append(word)
                masked.append(word)
        form, sure = self._classify(" ".join(out))
        if self.unknown_names and names:
            # The left-over words may be a name ("my tata motors holding") or words the examples
            # never used ("what's my top holding"): read it both ways and keep the surer reading.
            other = self._classify(" ".join(masked))
            if other[1][self.fields[0]] > sure[self.fields[0]]:
                form, sure = other
        beyond = names[1:] if self.unknown_names else names
        untaught = [w for run in beyond for w in run.split() if w not in _FILLER]
        return Reading(form, sure, self.fields[0], names[0] if names else None, known, untaught, self.cutoff)

    def _classify(self, text: str) -> tuple[dict[str, str], dict[str, float]]:
        form, sure = {}, {}
        for name, model in self.models.items():
            scores = model.predict(text)
            form[name] = max(scores, key=scores.get)
            sure[name] = scores[form[name]]
        return form, sure
