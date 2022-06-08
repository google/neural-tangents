"""Tests for `examples/imdb.py`."""


from absl.testing import absltest
from jax.config import config
from examples import imdb
from tests import test_utils

config.parse_flags_with_absl()


class ImdbTest(test_utils.NeuralTangentsTestCase):

  def test_imdb(self):
    imdb.main(use_dummy_data=True)


if __name__ == '__main__':
  absltest.main()
