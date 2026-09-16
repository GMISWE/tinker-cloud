"""LLVM-style pass infrastructure over TrainIR.

Design: specs/015-lift/PASSMANAGER.md. Immutable IR makes analysis caching
memoization by (analysis, term-hash); no invalidation protocol exists.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

from lift import ir


class ELevel(IntEnum):
    E0 = 0
    E1 = 1
    E2 = 2


def meet(a: ELevel, b: ELevel) -> ELevel:
    return max(a, b)


# --- outcomes ------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    reason: str
    witness: dict  # canon form of the offending subterm
    provenance: str = ""


@dataclass(frozen=True)
class Outcome:
    status: str  # "unchanged" | "changed" | "refused" | "error"
    term: ir.Node | None = None
    refusal: Refusal | None = None
    errors: tuple[str, ...] = ()

    @staticmethod
    def unchanged() -> Outcome:
        return Outcome("unchanged")

    @staticmethod
    def changed(term: ir.Node) -> Outcome:
        return Outcome("changed", term=term)

    @staticmethod
    def refused(reason: str, witness: ir.Node) -> Outcome:
        prov = ""
        loc = witness.loc
        if loc.src or loc.key:
            prov = f"{loc.src}:{loc.key}"
        return Outcome("refused", refusal=Refusal(reason, witness.canon(), prov))

    @staticmethod
    def error(*errors: str) -> Outcome:
        return Outcome("error", errors=tuple(errors))


class Pass(Protocol):
    name: str
    level: ELevel

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome: ...


# --- analyses ------------------------------------------------------------------

Analysis = Callable[[ir.Node], object]

ANALYSES: dict[str, Analysis] = {
    "wf": ir.wf,
    "leaks": ir.leaks,
    "free-attrs": ir.free_attrs,
    "holes": ir.holes,
    "hash": lambda n: n.hash,
}


class AnalysisManager:
    def __init__(self, registry: dict[str, Analysis] | None = None):
        self._registry = dict(registry or ANALYSES)
        self._cache: dict[tuple[str, str], object] = {}

    def get(self, name: str, term: ir.Node):
        key = (name, term.hash)
        if key not in self._cache:
            self._cache[key] = self._registry[name](term)
        return self._cache[key]


# --- report --------------------------------------------------------------------


@dataclass
class PassRecord:
    name: str
    status: str
    level: str
    time_ms: float
    refusal: Refusal | None = None
    errors: tuple[str, ...] = ()
    hash_after: str = ""


@dataclass
class CompilationReport:
    pipeline: str
    source: str = ""
    records: list[PassRecord] = field(default_factory=list)
    guarantee: ELevel = ELevel.E0
    stopped: bool = False
    leaks: tuple[str, ...] = ()
    holes: tuple[dict, ...] = ()
    final_hash: str = ""
    analyses: dict = field(default_factory=dict)

    def to_json(self) -> str:
        def default(o):
            if isinstance(o, (Refusal, PassRecord)):
                return o.__dict__
            if isinstance(o, ELevel):
                return o.name
            if isinstance(o, frozenset):
                return sorted(o)
            raise TypeError(type(o).__name__)

        return json.dumps(self.__dict__, default=default, indent=1, sort_keys=True)


# --- pipeline parsing ----------------------------------------------------------


def parse_pipeline(s: str) -> list[tuple[str, dict]]:
    """ "a,b<k=v;k2=v2>,c" -> [(a,{}), (b,{k:v,k2:v2}), (c,{})]."""
    out: list[tuple[str, dict]] = []
    depth, tok = 0, ""
    parts: list[str] = []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(tok)
            tok = ""
        else:
            tok += ch
    if depth != 0:
        raise ValueError(f"unbalanced <> in pipeline: {s}")
    if tok:
        parts.append(tok)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "<" in p:
            name, rest = p.split("<", 1)
            body = rest.rsplit(">", 1)[0]
            opts = {}
            for kv in body.split(";"):
                if not kv:
                    continue
                k, _, v = kv.partition("=")
                opts[k.strip()] = v.strip()
            out.append((name.strip(), opts))
        else:
            out.append((p, {}))
    return out


# --- the manager ---------------------------------------------------------------


class PassManager:
    def __init__(
        self,
        pipeline: str,
        registry: dict[str, Pass],
        verify_each: bool = True,
        analyses: dict[str, Analysis] | None = None,
    ):
        self.pipeline_str = pipeline
        self.stages = parse_pipeline(pipeline)
        self.registry = registry
        self.verify_each = verify_each
        self.am = AnalysisManager(analyses)
        for name, _ in self.stages:
            if name not in registry:
                raise KeyError(f"unknown pass: {name}")

    def run(self, unit, source: str = "") -> tuple[object, CompilationReport]:
        """The unit is usually an ir.Program; the select/emit stages of
        PM[Select] flow non-Node units, for which the Node-only checks
        (wf, hole conservation, leak/hole reporting) are skipped."""
        rpt = CompilationReport(pipeline=self.pipeline_str, source=source)
        cur = unit
        for name, opts in self.stages:
            cur, stop = self._run_one(name, opts, cur, rpt)
            if stop:
                rpt.stopped = True
                break
        if isinstance(cur, ir.Node):
            rpt.leaks = tuple(sorted(self.am.get("leaks", cur)))  # type: ignore[arg-type]
            rpt.holes = tuple(h.canon() for h in self.am.get("holes", cur))  # type: ignore[union-attr]
        rpt.final_hash = cur.hash if isinstance(cur, ir.Node) else ""
        return cur, rpt

    def _run_one(
        self, name: str, opts: dict, cur, rpt: CompilationReport
    ) -> tuple[object, bool]:
        p = self.registry[name]
        holes_before = len(ir.holes(cur)) if isinstance(cur, ir.Node) else None
        t0 = time.perf_counter()
        out = p.run(cur, self.am, opts)
        dt = (time.perf_counter() - t0) * 1e3
        rec = PassRecord(
            name=name, status=out.status, level=p.level.name, time_ms=round(dt, 3)
        )

        if out.status == "error":
            rec.errors = out.errors
            rpt.records.append(rec)
            return cur, True

        if out.status == "refused":
            rec.refusal = out.refusal
            policy = opts.get("policy", "abort")
            rpt.records.append(rec)
            if policy == "abort":
                return cur, True
            if policy == "skip":
                return cur, False
            if policy.startswith("repair:"):
                return self._run_one(policy.split(":", 1)[1], {}, cur, rpt)
            raise ValueError(f"unknown refusal policy: {policy}")

        if out.status == "changed":
            nxt = out.term
            if nxt is None:
                raise ValueError(f"pass {name!r} reported changed without a term")
            rpt.guarantee = meet(rpt.guarantee, p.level)
            if isinstance(nxt, ir.Node):
                # hole conservation: no pass may fabricate or drop obligations
                if (
                    holes_before is not None
                    and len(ir.holes(nxt)) != holes_before
                    and not opts.get("fills_holes")
                ):
                    rec.status = "error"
                    rec.errors = ("hole count changed by pass",)
                    rpt.records.append(rec)
                    return cur, True
                if self.verify_each:
                    errs = self.am.get("wf", nxt)
                    if errs:
                        rec.status = "error"
                        rec.errors = tuple(errs)  # type: ignore[arg-type]
                        rpt.records.append(rec)
                        return cur, True
            rec.hash_after = nxt.hash
            rpt.records.append(rec)
            return nxt, False

        rpt.records.append(rec)
        return cur, False
