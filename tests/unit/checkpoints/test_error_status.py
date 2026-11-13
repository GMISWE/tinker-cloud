"""The status a store error carries to the application's handler."""
from tinkercloud.training.checkpoints import CheckpointError, CheckpointExists


def test_a_duplicate_save_is_a_checkpoint_error_the_client_caused():
    """It was a bare Exception, outside the errors the application maps to a status. A server test
    cannot hold this: a failed save's future is a 400 whatever the error's class."""
    assert issubclass(CheckpointExists, CheckpointError)
    assert CheckpointExists.status_code == 400
