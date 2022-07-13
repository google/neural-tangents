from absl.testing import absltest
from absl.testing import parameterized
import functools
from jax import jit
from jax.config import config
import jax.numpy as np
import jax.random as random
from neural_tangents._src.utils import utils
from neural_tangents import stax
from tests import test_utils

import experimental.features as ft

config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')

test_utils.update_test_tolerance()

NUM_DIMS = [128, 256, 512]
WEIGHT_VARIANCES = [0.5, 1.]
BIAS_VARIANCES = [None, 0.1]


def _convert_features_to_matrices(f_, channel_axis=-1):
  if isinstance(f_, ft.Features):
    nngp = _convert_features_to_matrices(f_.nngp_feat, f_.channel_axis)
    ntk = _convert_features_to_matrices(f_.ntk_feat, f_.channel_axis)
    return nngp, ntk
  elif isinstance(f_, np.ndarray):
    channel_dim = f_.shape[channel_axis]
    feat = np.moveaxis(f_, channel_axis, -1).reshape(-1, channel_dim)
    k_mat = feat @ feat.T
    if f_.ndim > 2:
      k_mat = utils.zip_axes(
          k_mat.reshape(
              tuple(f_.shape[i]
                    for i in range(len(f_.shape))
                    if i != channel_axis) * 2))
    return k_mat
  else:
    raise ValueError


def _convert_image_feature_to_kernel(feat):
  return utils.zip_axes(np.einsum("ijkc,xyzc->ijkxyz", feat, feat))


def _get_init_data(rng, shape, normalized_output=False):
  x = random.normal(rng, shape)
  if normalized_output:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)
  else:
    return x


class FeaturesTest(test_utils.NeuralTangentsTestCase):

  @parameterized.product(W_std=WEIGHT_VARIANCES,
                         b_std=BIAS_VARIANCES,
                         n_layers=[1, 2, 3, 4],
                         do_jit=[True, False])
  def test_dense_features(self, W_std, b_std, n_layers, do_jit):
    n, d = 4, 256
    rng = random.PRNGKey(1)
    x = _get_init_data(rng, (n, d))

    dense_args = {'out_dim': 1, 'W_std': W_std, 'b_std': b_std}

    kernel_fn = stax.serial(*[stax.Dense(**dense_args)] * n_layers)[2]
    feature_fn = ft.serial(*[ft.DenseFeatures(**dense_args)] * n_layers)[1]

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x, None)
    f = feature_fn(x, [()] * n_layers)

    self.assertAllClose(k.nngp, f.nngp_feat @ f.nngp_feat.T)
    self.assertAllClose(k.ntk, f.ntk_feat @ f.ntk_feat.T)

  @parameterized.product(
      W_std=WEIGHT_VARIANCES,
      b_std=BIAS_VARIANCES,
      n_layers=[1, 2, 3, 4],
      relu_method=['RANDFEAT', 'POLYSKETCH', 'PSRF', 'POLY', 'EXACT'],
      do_jit=[True, False])
  def test_fc_relu_nngp_ntk(self, W_std, b_std, n_layers, relu_method, do_jit):
    rng = random.PRNGKey(1)
    n, d = 4, 256
    x = _get_init_data(rng, (n, d))

    dense_args = {"out_dim": 1, "W_std": W_std, "b_std": b_std}
    relu_args = {'method': relu_method}
    if relu_method == 'RANDFEAT':
      relu_args['feature_dim0'] = 4096
      relu_args['feature_dim1'] = 4096
      relu_args['sketch_dim'] = 4096
    elif relu_method == 'POLYSKETCH':
      relu_args['poly_degree'] = 4
      relu_args['poly_sketch_dim'] = 4096
      relu_args['sketch_dim'] = 4096
    elif relu_method == 'PSRF':
      relu_args['feature_dim0'] = 4096
      relu_args['poly_degree'] = 4
      relu_args['poly_sketch_dim'] = 4096
      relu_args['sketch_dim'] = 4096
    elif relu_method == 'POLY':
      relu_args['poly_degree'] = 16
    elif relu_method == 'EXACT':
      pass
    else:
      raise ValueError(relu_method)

    kernel_fn = stax.serial(
        *[stax.Dense(**dense_args), stax.Relu()] * n_layers +
        [stax.Dense(**dense_args)])[2]
    init_fn, feature_fn = ft.serial(
        *[ft.DenseFeatures(**dense_args),
          ft.ReluFeatures(**relu_args)] * n_layers +
        [ft.DenseFeatures(**dense_args)])

    rng2 = random.PRNGKey(2)
    _, feat_fn_inputs = init_fn(rng2, x.shape)

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x, None)
    f = feature_fn(x, feat_fn_inputs)

    if np.iscomplexobj(f.nngp_feat) or np.iscomplexobj(f.ntk_feat):
      nngp_feat = np.concatenate((f.nngp_feat.real, f.nngp_feat.imag), axis=-1)
      ntk_feat = np.concatenate((f.ntk_feat.real, f.ntk_feat.imag), axis=-1)
      f = f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)
    k_nngp_approx = f.nngp_feat @ f.nngp_feat.T
    k_ntk_approx = f.ntk_feat @ f.ntk_feat.T

    if relu_method == 'EXACT':
      self.assertAllClose(k.nngp, k_nngp_approx)
      self.assertAllClose(k.ntk, k_ntk_approx)
    else:
      test_utils.assert_close_matrices(self, k.nngp, k_nngp_approx, 0.2, 1.)
      test_utils.assert_close_matrices(self, k.ntk, k_ntk_approx, 0.2, 1.)

  @parameterized.product(W_std=WEIGHT_VARIANCES,
                         b_std=BIAS_VARIANCES,
                         n_layers=[1, 2, 3, 4],
                         do_jit=[True, False])
  def test_conv_features(self, W_std, b_std, n_layers, do_jit):
    n, h, w, c = 3, 4, 5, 2
    rng = random.PRNGKey(1)
    x = _get_init_data(rng, (n, h, w, c))

    conv_args = {
        'out_chan': 1,
        'filter_shape': (3, 3),
        'padding': 'SAME',
        'W_std': W_std,
        'b_std': b_std
    }

    kernel_fn = stax.serial(*[stax.Conv(**conv_args)] * n_layers)[2]
    feature_fn = ft.serial(*[ft.ConvFeatures(**conv_args)] * n_layers)[1]

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x)
    f = feature_fn(x, [()] * n_layers)

    if k.is_reversed:
      nngp_feat = np.moveaxis(f.nngp_feat, 1, 2)
      ntk_feat = np.moveaxis(f.ntk_feat, 1, 2)
      f = f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

    k_nngp_approx = _convert_image_feature_to_kernel(f.nngp_feat)
    k_ntk_approx = _convert_image_feature_to_kernel(f.ntk_feat)

    self.assertAllClose(k.nngp, k_nngp_approx)
    self.assertAllClose(k.ntk, k_ntk_approx)

  @parameterized.product(n_layers=[1, 2, 3, 4], do_jit=[True, False])
  def test_avgpool_features(self, n_layers, do_jit):
    n, h, w, c = 3, 32, 28, 2
    rng = random.PRNGKey(1)
    x = _get_init_data(rng, (n, h, w, c))

    avgpool_args = {
        'window_shape': (2, 2),
        'strides': (2, 2),
        'padding': 'SAME'
    }

    kernel_fn = stax.serial(*[stax.AvgPool(**avgpool_args)] * n_layers +
                            [stax.Flatten()])[2]
    feature_fn = ft.serial(*[ft.AvgPoolFeatures(**avgpool_args)] * n_layers +
                           [ft.FlattenFeatures()])[1]

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x)
    f = feature_fn(x, [()] * (n_layers + 1))

    k_nngp_approx, k_ntk_approx = _convert_features_to_matrices(f)

    self.assertAllClose(k.nngp, k_nngp_approx)
    if k.ntk.ndim > 0:
      self.assertAllClose(k.ntk, k_ntk_approx)

  @parameterized.parameters([{
      'ndim': nd,
      'do_jit': do_jit
  } for nd in [2, 3, 4] for do_jit in [True, False]])
  def test_flatten_features(self, ndim, do_jit):
    key = random.PRNGKey(1)
    n, h, w, c = 4, 8, 6, 5
    width = 1
    W_std = 1.7
    b_std = 0.1
    if ndim == 2:
      input_shape = (n, h * w * c)
    elif ndim == 3:
      input_shape = (n, h * w, c)
    elif ndim == 4:
      input_shape = (n, h, w, c)
    else:
      raise absltest.SkipTest()

    x = random.normal(key, input_shape)

    dense_kernel = stax.Dense(width, W_std=W_std, b_std=b_std)
    dense_feature = ft.DenseFeatures(width, W_std=W_std, b_std=b_std)

    relu_kernel = stax.Relu()
    relu_feature = ft.ReluFeatures(method='EXACT')

    kernel_fc = stax.serial(dense_kernel, relu_kernel, dense_kernel)[2]
    kernel_top = stax.serial(dense_kernel, relu_kernel, dense_kernel,
                             stax.Flatten())[2]
    kernel_mid = stax.serial(dense_kernel, relu_kernel, stax.Flatten(),
                             dense_kernel)[2]
    kernel_bot = stax.serial(stax.Flatten(), dense_kernel, relu_kernel,
                             dense_kernel)[2]

    feature_fc = ft.serial(dense_feature, relu_feature, dense_feature)[1]
    feature_top = ft.serial(dense_feature, relu_feature, dense_feature,
                            ft.FlattenFeatures())[1]
    feature_mid = ft.serial(dense_feature, relu_feature, ft.FlattenFeatures(),
                            dense_feature)[1]
    feature_bot = ft.serial(ft.FlattenFeatures(), dense_feature, relu_feature,
                            dense_feature)[1]

    if do_jit:
      kernel_fc = jit(kernel_fc)
      kernel_top = jit(kernel_top)
      kernel_mid = jit(kernel_mid)
      kernel_bot = jit(kernel_bot)

      feature_fc = jit(feature_fc)
      feature_top = jit(feature_top)
      feature_mid = jit(feature_mid)
      feature_bot = jit(feature_bot)

    k_fc = kernel_fc(x)
    f_fc = feature_fc(x, [()] * 3)
    nngp_fc, ntk_fc = _convert_features_to_matrices(f_fc)
    self.assertAllClose(k_fc.nngp, nngp_fc)
    self.assertAllClose(k_fc.ntk, ntk_fc)

    k_top = kernel_top(x)
    f_top = feature_top(x, [()] * 4)
    nngp_top, ntk_top = _convert_features_to_matrices(f_top)
    self.assertAllClose(k_top.nngp, nngp_top)
    self.assertAllClose(k_top.ntk, ntk_top)

    k_mid = kernel_mid(x)
    f_mid = feature_mid(x, [()] * 4)
    nngp_mid, ntk_mid = _convert_features_to_matrices(f_mid)
    self.assertAllClose(k_mid.nngp, nngp_mid)
    self.assertAllClose(k_mid.ntk, ntk_mid)

    k_bot = kernel_bot(x)
    f_bot = feature_bot(x, [()] * 4)
    nngp_bot, ntk_bot = _convert_features_to_matrices(f_bot)
    self.assertAllClose(k_bot.nngp, nngp_bot)
    self.assertAllClose(k_bot.ntk, ntk_bot)

  @parameterized.product(ndim=[2, 3, 4],
                         channel_axis=[1, 2, 3],
                         n_layers=[1, 2, 3, 4],
                         use_conv=[True, False],
                         use_layernorm=[True, False],
                         do_pool=[True, False],
                         do_jit=[True, False])
  def test_channel_axis(self, ndim, channel_axis, use_conv, n_layers,
                        use_layernorm, do_pool, do_jit):
    n, h, w, c = 4, 8, 6, 5
    W_std = 1.7
    b_std = 0.1
    key = random.PRNGKey(1)

    if ndim == 2:
      if channel_axis != 1:
        raise absltest.SkipTest()
      input_shape = (n, h * w * c)
    elif ndim == 3:
      if channel_axis == 1:
        input_shape = (n, c, h * w)
      elif channel_axis == 2:
        input_shape = (n, h * w, c)
      else:
        raise absltest.SkipTest()
    elif ndim == 4:
      if channel_axis == 1:
        input_shape = (n, c, h, w)
        dn = ('NCAB', 'ABIO', 'NCAB')
      elif channel_axis == 3:
        input_shape = (n, h, w, c)
        dn = ('NABC', 'ABIO', 'NABC')
      else:
        raise absltest.SkipTest()

    x = random.normal(key, input_shape)

    if use_conv:
      if ndim != 4:
        raise absltest.SkipTest()
      else:
        linear = stax.Conv(1, (3, 3), (1, 1),
                           'SAME',
                           W_std=W_std,
                           b_std=b_std,
                           dimension_numbers=dn)
        linear_feat = ft.ConvFeatures(1, (3, 3), (1, 1),
                                      W_std=W_std,
                                      b_std=b_std,
                                      dimension_numbers=dn)
    else:
      linear = stax.Dense(1,
                          W_std=W_std,
                          b_std=b_std,
                          channel_axis=channel_axis)
      linear_feat = ft.DenseFeatures(1,
                                     W_std=W_std,
                                     b_std=b_std,
                                     channel_axis=channel_axis)

    layers = [linear, stax.Relu()] * n_layers
    layers += [linear]
    layers += [stax.LayerNorm(channel_axis, channel_axis=channel_axis)
              ] if use_layernorm else []
    layers += [stax.GlobalAvgPool(
        channel_axis=channel_axis)] if do_pool else [stax.Flatten()]
    kernel_fn = stax.serial(*layers)[2]

    layers = [
        linear_feat,
        ft.ReluFeatures(method='EXACT', channel_axis=channel_axis)
    ] * n_layers
    layers += [linear_feat]
    layers += [ft.LayerNormFeatures(channel_axis, channel_axis=channel_axis)
              ] if use_layernorm else []
    layers += [ft.GlobalAvgPoolFeatures(
        channel_axis=channel_axis)] if do_pool else [ft.FlattenFeatures()]
    feature_fn = ft.serial(*layers)[1]

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x)
    f = feature_fn(x, [()] * len(layers))
    nngp, ntk = _convert_features_to_matrices(f)
    self.assertAllClose(k.nngp, nngp)
    self.assertAllClose(k.ntk, ntk)

  @parameterized.product(
      channel_axis=[1, 3],
      W_std=WEIGHT_VARIANCES,
      b_std=BIAS_VARIANCES,
      relu_method=['RANDFEAT', 'POLYSKETCH', 'PSRF', 'POLY', 'EXACT'],
      depth=[5],
      do_jit=[True, False])
  def test_myrtle_network(self, channel_axis, W_std, b_std, relu_method, depth,
                          do_jit):
    n, h, w, c = 2, 32, 32, 3
    rng = random.PRNGKey(1)
    if channel_axis == 1:
      x = _get_init_data(rng, (n, c, h, w))
      dn = ('NCAB', 'ABIO', 'NCAB')
    elif channel_axis == 3:
      x = _get_init_data(rng, (n, h, w, c))
      dn = ('NABC', 'ABIO', 'NABC')

    layer_factor = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}

    def _get_myrtle_kernel_fn():
      conv = functools.partial(stax.Conv,
                               W_std=W_std,
                               b_std=b_std,
                               padding='SAME',
                               dimension_numbers=dn)
      layers = []
      layers += [conv(1, (3, 3)), stax.Relu()] * layer_factor[depth][0]
      layers += [
          stax.AvgPool((2, 2), strides=(2, 2), channel_axis=channel_axis)
      ]
      layers += [conv(1, (3, 3)), stax.Relu()] * layer_factor[depth][1]
      layers += [
          stax.AvgPool((2, 2), strides=(2, 2), channel_axis=channel_axis)
      ]
      layers += [conv(1, (3, 3)), stax.Relu()] * layer_factor[depth][2]
      layers += [
          stax.AvgPool((2, 2), strides=(2, 2), channel_axis=channel_axis)
      ] * 3
      layers += [stax.Flatten(), stax.Dense(1, W_std=W_std, b_std=b_std)]

      return stax.serial(*layers)

    def _get_myrtle_feature_fn(**relu_args):
      conv = functools.partial(ft.ConvFeatures,
                               W_std=W_std,
                               b_std=b_std,
                               padding='SAME',
                               dimension_numbers=dn)
      layers = []
      layers += [
          conv(1, (3, 3)),
          ft.ReluFeatures(channel_axis=channel_axis, **relu_args)
      ] * layer_factor[depth][0]
      layers += [
          ft.AvgPoolFeatures((2, 2), strides=(2, 2), channel_axis=channel_axis)
      ]
      layers += [
          conv(1, (3, 3)),
          ft.ReluFeatures(channel_axis=channel_axis, **relu_args)
      ] * layer_factor[depth][1]
      layers += [
          ft.AvgPoolFeatures((2, 2), strides=(2, 2), channel_axis=channel_axis)
      ]
      layers += [
          conv(1, (3, 3)),
          ft.ReluFeatures(channel_axis=channel_axis, **relu_args)
      ] * layer_factor[depth][2]
      layers += [
          ft.AvgPoolFeatures((2, 2), strides=(2, 2), channel_axis=channel_axis)
      ] * 3
      layers += [
          ft.FlattenFeatures(),
          ft.DenseFeatures(1, W_std=W_std, b_std=b_std)
      ]

      return ft.serial(*layers)

    kernel_fn = _get_myrtle_kernel_fn()[2]

    relu_args = {'method': relu_method}
    if relu_method == 'RANDFEAT':
      relu_args['feature_dim0'] = 2048
      relu_args['feature_dim1'] = 2048
      relu_args['sketch_dim'] = 2048
    elif relu_method == 'POLYSKETCH':
      relu_args['poly_degree'] = 4
      relu_args['poly_sketch_dim'] = 2048
      relu_args['sketch_dim'] = 2048
    elif relu_method == 'PSRF':
      relu_args['feature_dim0'] = 2048
      relu_args['poly_degree'] = 4
      relu_args['poly_sketch_dim'] = 2048
      relu_args['sketch_dim'] = 2048
    elif relu_method == 'POLY':
      relu_args['poly_degree'] = 16
    elif relu_method == 'EXACT':
      pass
    else:
      raise ValueError(relu_method)

    init_fn, feature_fn = _get_myrtle_feature_fn(**relu_args)

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x)
    k_nngp = k.nngp
    k_ntk = k.ntk

    _, feat_fn_inputs = init_fn(rng, x.shape)
    f = feature_fn(x, feat_fn_inputs)
    if np.iscomplexobj(f.nngp_feat) or np.iscomplexobj(f.ntk_feat):
      nngp_feat = np.concatenate((f.nngp_feat.real, f.nngp_feat.imag), axis=-1)
      ntk_feat = np.concatenate((f.ntk_feat.real, f.ntk_feat.imag), axis=-1)
      f = f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

    k_nngp_approx = f.nngp_feat @ f.nngp_feat.T
    k_ntk_approx = f.ntk_feat @ f.ntk_feat.T

    if relu_method == 'EXACT':
      self.assertAllClose(k_nngp, k_nngp_approx)
      self.assertAllClose(k_ntk, k_ntk_approx)
    else:
      test_utils.assert_close_matrices(self, k_nngp, k_nngp_approx, 0.2, 1.)
      test_utils.assert_close_matrices(self, k_ntk, k_ntk_approx, 0.2, 1.)

  def test_aggregate_features(self):
    rng = random.PRNGKey(1)
    rng1, rng2 = random.split(rng, 2)

    batch_size = 4
    num_channels = 3
    shape = (5,)
    width = 1

    x = random.normal(rng1, (batch_size,) + shape + (num_channels,))
    pattern = random.uniform(rng2, (batch_size,) + shape * 2)

    kernel_fn = stax.serial(stax.Dense(width, W_std=2**0.5), stax.Relu(),
                            stax.Aggregate(), stax.GlobalAvgPool(),
                            stax.Dense(width))[2]

    k = jit(kernel_fn)(x, None, pattern=(pattern, pattern))

    feature_fn = ft.serial(ft.DenseFeatures(width, W_std=2**0.5),
                           ft.ReluFeatures(method='EXACT'),
                           ft.AggregateFeatures(), ft.GlobalAvgPoolFeatures(),
                           ft.DenseFeatures(width))[1]

    f = feature_fn(x, [()] * 5, **{'pattern': pattern})
    self.assertAllClose(k.nngp, f.nngp_feat @ f.nngp_feat.T)
    self.assertAllClose(k.ntk, f.ntk_feat @ f.ntk_feat.T)


if __name__ == "__main__":
  absltest.main()
