"""TrainingGroup._all: a rank that raises while a peer is still running (the peer waits in a collective
the failed rank left) is reported within RANK_GRACE_S, logged when it arrives, and raised as
RanksStranded naming the stranded ranks; the same verdict from every rank is raised as itself."""
import asyncio
import logging

import pytest

from tinkercloud.training.backends.automodel import group as group_mod
from tinkercloud.training.backends.automodel.group import RanksStranded, TrainingGroup


class Verdict(Exception):
    pass


async def _answer(value=None, raises=None, delay=0.0):
    await asyncio.sleep(delay)
    if raises is not None:
        raise raises
    return value


async def _never():
    await asyncio.Event().wait()


def test_answers_come_back_in_rank_order():
    g = TrainingGroup("k")
    assert asyncio.run(g._all([_answer(0, delay=0.02), _answer(1)])) == [0, 1]


def test_the_same_verdict_on_every_rank_is_raised_as_itself():
    g = TrainingGroup("k")
    with pytest.raises(Verdict, match="rank 0"):
        asyncio.run(g._all([_answer(raises=Verdict("rank 0"), delay=0.02), _answer(raises=Verdict("rank 1"))]))


def test_a_rank_raising_beside_a_stranded_peer_is_reported_within_the_grace(monkeypatch, caplog):
    monkeypatch.setattr(group_mod, "RANK_GRACE_S", 0.1)
    g = TrainingGroup("k")
    with caplog.at_level(logging.ERROR, logger=group_mod.__name__):
        with pytest.raises(RanksStranded, match=r"rank 1 raised .*cuDNN.*ranks \[0, 2\]") as ei:
            asyncio.run(asyncio.wait_for(g._all([_never(), _answer(raises=ValueError("cuDNN")), _never()]), 5))
    assert isinstance(ei.value.__cause__, ValueError)
    assert any("rank 1 raised ValueError" in r.getMessage() for r in caplog.records)
