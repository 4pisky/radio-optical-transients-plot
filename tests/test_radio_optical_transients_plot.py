from radio_optical_transients_plot import __version__
from radio_optical_transients_plot.ro_utils import (
    OpticaltomJy, mJytoOptical, ConvertToABMag
)


def test_version():
    assert __version__ == '0.1.0'


class TestConversions:
    def test_OpticaltomJy_R(self):
        assert 3.631 == OpticaltomJy(15, 'R')

    def test_OpticaltomJy_V(self):
        assert 3.631 == OpticaltomJy(15, 'V')

    def test_mJytoOptical_R(self):
        assert 15.0 == mJytoOptical(3.631, 'R')

    def test_mJytoOptical_V(self):
        assert 15.0 == mJytoOptical(3.631, 'V')

    def test_ConvertToABMag_R(self):
        assert 15.055 == ConvertToABMag(15.0, 'R')

    def test_ConvertToABMag_V(self):
        assert 14.956 == ConvertToABMag(15.0, 'V')
