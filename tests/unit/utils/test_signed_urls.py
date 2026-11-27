"""utils.signed_urls: verify() answers every signature a URL can carry with True or False."""
import pytest

from tinkercloud.training.utils import signed_urls

KEY, MODEL, NAME, REL = "signing-key", "model_e_0123456789abcdef", "export", "model.safetensors"


def test_the_signature_it_made_verifies_until_expiry():
    sig = signed_urls.sign(KEY, MODEL, NAME, REL, 2000)
    assert signed_urls.verify(KEY, MODEL, NAME, REL, 2000, sig, now=1999)
    assert not signed_urls.verify(KEY, MODEL, NAME, REL, 2000, sig, now=2000)


@pytest.mark.parametrize("sig", ["", "00", "é", "\u00ff" * 64, "ü" + "0" * 63])
def test_a_tampered_or_non_ascii_signature_is_refused_not_raised(sig):
    assert signed_urls.verify(KEY, MODEL, NAME, REL, 2000, sig, now=1000) is False
