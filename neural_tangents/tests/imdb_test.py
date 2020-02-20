"""Tests for `examples/imdb.py`."""


from jax import test_util as jtu
from jax.config import config
from examples import imdb


config.parse_flags_with_absl()


class ImdbTest(jtu.JaxTestCase):

  def test_imdb(self):
    imdb.main(None)


if __name__ == '__main__':
  jtu.absltest.main()
