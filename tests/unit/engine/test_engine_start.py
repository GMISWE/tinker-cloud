"""An engine start removes every future the last run left, pending ones included: no operation of the
last run runs after a restart, so a pending future left behind would never settle, and a create-via-load
resend would be answered with it."""
from tinkercloud.training.storage.futures import FuturesStorage

from .support import engine


def test_a_start_removes_the_last_run_s_pending_futures(tmp_path):
    (tmp_path / "meta").mkdir()
    last_run = FuturesStorage(tmp_path / "meta" / "futures.db")
    last_run.save_future("running", "load_weights", {"p": 1}, model_id="s:train:1", seq_id=0)
    last_run.close()
    with engine(tmp_path):
        after = FuturesStorage(tmp_path / "meta" / "futures.db")
        assert after.get_future("running") is None
        assert after.filed("s:train:1", 0, "load_weights", {"p": 1}) is None
        after.close()
