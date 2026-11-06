"""The conformance suite drives the SDK release the server targets."""
import tinker


def test_the_sdk_is_the_targeted_release():
    assert tinker.__version__ == "0.38.0"
