"""TrainIR: typed, immutable, content-hashed terms for lifted training programs.

Semantics, judgments, and design rationale: specs/015-lift/IR.md.
Carrier rules: frozen dataclasses, no framework imports, provenance (Loc)
never enters the hash, no binders (Nested introduces the current group;
CurGroup is only legal beneath it).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Union


@dataclass(frozen=True)
class Loc:
    """Provenance: source file/key that elaborated this node. Excluded from hash."""

    src: str = ""
    key: str = ""


NOLOC = Loc()


class Node:
    """Mixin for IR nodes: canonical form, content hash, child traversal."""
    loc: Loc  # every concrete node dataclass declares it (default NOLOC)

    def canon(self) -> dict:
        out: dict = {"_": type(self).__name__}
        for f in fields(self):  # type: ignore[arg-type]
            if f.name == "loc":
                continue
            out[f.name] = _canon_val(getattr(self, f.name))
        return out

    @property
    def hash(self) -> str:
        s = json.dumps(self.canon(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()

    def children(self) -> Iterator[Node]:
        for f in fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            if isinstance(v, Node):
                yield v
            elif isinstance(v, tuple):
                for x in v:
                    if isinstance(x, Node):
                        yield x


def _canon_val(v):
    if isinstance(v, Node):
        return v.canon()
    if isinstance(v, Enum):
        return v.name
    if isinstance(v, tuple):
        return [_canon_val(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    raise TypeError(f"non-canonicalizable value in IR: {type(v).__name__}")


def walk(n: Node) -> Iterator[Node]:
    yield n
    for c in n.children():
        yield from walk(c)


# --- partitions and index sets -------------------------------------------------


@dataclass(frozen=True)
class Part(Node):
    """A partition of the step's tokens. attr != None marks it as defined by an
    execution attribute (micro-batch, dp shard) — the leak judgment keys on this."""

    name: str
    attr: str | None = None
    loc: Loc = NOLOC


@dataclass(frozen=True)
class Tokens(Node):
    """All tokens of the step's batch."""


@dataclass(frozen=True)
class CurGroup(Node):
    """Tokens of the group bound by the enclosing Nested. WF: only under Nested."""


Over = Union[Tokens, CurGroup]


# --- denominators --------------------------------------------------------------


@dataclass(frozen=True)
class Count(Node):
    over: Over


@dataclass(frozen=True)
class WSum(Node):
    over: Over


@dataclass(frozen=True)
class Const(Node):
    value: float


@dataclass(frozen=True)
class CountGroups(Node):
    part: Part


Denom = Union[Count, WSum, Const, CountGroups]


# --- per-token ops (registry-checked) ------------------------------------------

PER_TOK_OPS = frozenset({"ce", "logprob", "dpo", "ratio_clip"})


@dataclass(frozen=True)
class PerTok(Node):
    op: str
    loc: Loc = NOLOC


# --- weights -------------------------------------------------------------------


@dataclass(frozen=True)
class W(Node):
    """Data-supplied per-token weights."""


@dataclass(frozen=True)
class WScaled(Node):
    """base / denom, denom evaluated once over its index set."""

    base: Weight
    denom: Denom


@dataclass(frozen=True)
class WGrouped(Node):
    """base / denom(group(part) containing the token). Data-defined parts only
    survive norm-canon; attr-defined parts here are a leak."""

    base: Weight
    part: Part
    denom_kind: str  # "count" | "wsum"


Weight = Union[W, WScaled, WGrouped]


# --- loss ----------------------------------------------------------------------


class Agg(Enum):
    SUM = "sum"
    MEAN = "mean"


@dataclass(frozen=True)
class Reduce(Node):
    agg: Agg
    denom: Denom | None  # required iff agg is MEAN
    over: Over
    of: PerTok
    weight: Weight
    loc: Loc = NOLOC


@dataclass(frozen=True)
class Nested(Node):
    """agg over the groups of `part` of an inner reduction over CurGroup."""

    agg: Agg
    part: Part
    inner: LossExpr
    loc: Loc = NOLOC


@dataclass(frozen=True)
class Hole(Node):
    """Typed obligation: U unknown / I inexpressible / ENV the RL environment."""

    kind: str  # "U" | "I" | "ENV"
    ty: str = ""
    loc: Loc = NOLOC


LossExpr = Union[Reduce, Nested, Hole]


# --- schedules, optimizer, adapter ---------------------------------------------


@dataclass(frozen=True)
class ConstLR(Node):
    base: float


@dataclass(frozen=True)
class LinearLR(Node):
    base: float
    end_frac: float = 0.0


@dataclass(frozen=True)
class WarmupLR(Node):
    warmup_steps: int
    inner: Sched
    init_frac: float = 0.0  # start of warmup as a fraction of inner's base LR


Sched = Union[ConstLR, LinearLR, WarmupLR]


@dataclass(frozen=True)
class AdamW(Node):
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    wd: float = 0.0
    clip: float | None = None


@dataclass(frozen=True)
class Lora(Node):
    r: int
    alpha: float
    targets: tuple[str, ...] = ()


# --- stream --------------------------------------------------------------------


@dataclass(frozen=True)
class SrcData(Node):
    ref: str
    loc: Loc = NOLOC


@dataclass(frozen=True)
class Render(Node):
    renderer: str
    inner: StreamExpr
    train_on: str = "assistant"  # loss-mask rule: "assistant" | "final" | "all"


@dataclass(frozen=True)
class Tokenize(Node):
    tokenizer: str
    inner: StreamExpr
    add_bos: bool = True
    add_eos: bool = True


@dataclass(frozen=True)
class Truncate(Node):
    max_len: int
    inner: StreamExpr
    # "truncate": cut to max_len. "mask_out": over-length samples are stubbed
    # and removed from the loss (nemo_rl sft_processor: loss_multiplier=0) but
    # still occupy batch slots — a DIFFERENT program from truncation.
    policy: str = "truncate"


@dataclass(frozen=True)
class BatchStream(Node):
    batch_size: int
    epochs: int
    seed: int | None
    inner: StreamExpr


StreamExpr = Union[SrcData, Render, Tokenize, Truncate, BatchStream, Hole]


# --- program -------------------------------------------------------------------


@dataclass(frozen=True)
class Points(Node):
    eval_every: int | None = None
    save_every: int | None = None
    publish: str = "none"  # "none" | "on_optim_step" | ...


@dataclass(frozen=True)
class Program(Node):
    model_ref: str
    adapter: Lora | None
    stream: StreamExpr
    loss: LossExpr
    opt: AdamW
    lr: Sched
    horizon: int | None
    points: Points
    loc: Loc = NOLOC


# --- judgments -----------------------------------------------------------------


def wf(n: Node) -> list[str]:
    """Well-formedness. Returns errors; [] means WF. See IR.md Part II §5."""
    errs: list[str] = []

    def go(x: Node, in_nested: bool):
        if isinstance(x, CurGroup) and not in_nested:
            errs.append("CurGroup outside Nested")
        if isinstance(x, Reduce):
            if x.agg is Agg.MEAN and x.denom is None:
                errs.append("MEAN Reduce without denominator")
            if x.agg is Agg.SUM and x.denom is not None:
                errs.append("SUM Reduce with denominator")
        if isinstance(x, PerTok) and x.op not in PER_TOK_OPS:
            errs.append(f"unregistered per-token op: {x.op}")
        if isinstance(x, Hole) and x.kind not in ("U", "I", "ENV"):
            errs.append(f"bad hole kind: {x.kind}")
        entered = in_nested or isinstance(x, Nested)
        for c in x.children():
            go(c, entered)

    go(n, False)
    return errs


def leaks(n: Node) -> frozenset[str]:
    """Attrs of execution-defined partitions occurring in the term (IR.md §5)."""
    return frozenset(
        x.attr for x in walk(n) if isinstance(x, Part) and x.attr is not None
    )


def holes(n: Node) -> tuple[Hole, ...]:
    return tuple(x for x in walk(n) if isinstance(x, Hole))


def free_attrs(n: Node) -> frozenset[str]:
    """FV over the Attr sort. Today attrs enter terms only via partitions."""
    return leaks(n)


__all__ = [
    "AdamW",
    "Agg",
    "BatchStream",
    "Const",
    "ConstLR",
    "Count",
    "CountGroups",
    "CurGroup",
    "Hole",
    "LinearLR",
    "Loc",
    "Lora",
    "LossExpr",
    "NOLOC",
    "Nested",
    "Node",
    "Part",
    "PerTok",
    "Points",
    "Program",
    "Reduce",
    "Render",
    "Sched",
    "SrcData",
    "StreamExpr",
    "Tokenize",
    "Tokens",
    "Truncate",
    "W",
    "WGrouped",
    "WScaled",
    "WSum",
    "WarmupLR",
    "Weight",
    "free_attrs",
    "holes",
    "leaks",
    "replace",
    "walk",
    "wf",
]
