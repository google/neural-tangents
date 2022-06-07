from absl.testing import absltest
from absl.testing import parameterized
import functools

from jax import jit
from jax.config import config
import jax.numpy as np
import jax.random as random
from jax import test_util as jtu
from neural_tangents._src.utils import utils as ntutils
from neural_tangents import stax
from tests import test_utils

from experimental.features import DenseFeatures, ReluFeatures, ConvFeatures, AvgPoolFeatures, FlattenFeatures, serial

config.update("jax_enable_x64", True)
config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')

NUM_DIMS = [64, 128, 256, 512]
WEIGHT_VARIANCES = [0.001, 0.01, 0.1, 1.]
BIAS_VARIANCES = [None, 0.001, 0.01, 0.1]
test_utils.update_test_tolerance()


class FeaturesTest(jtu.JaxTestCase):

  @classmethod
  def _get_init_data(cls, rng, shape, normalized_output=False):
    x = random.normal(rng, shape)
    if normalized_output:
      return x / np.linalg.norm(x, axis=-1, keepdims=True)
    else:
      return x

  @classmethod
  def _convert_image_feature_to_kernel(cls, f_):
    return ntutils.zip_axes(np.einsum("ijkc,xyzc->ijkxyz", f_, f_))

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              ' [Wstd{}_bstd{}_{}layers_{}] '.format(W_std, b_std, n_layers,
                                                     'jit' if do_jit else ''),
          'W_std':
              W_std,
          'b_std':
              b_std,
          'n_layers':
              n_layers,
          'do_jit':
              do_jit,
      } for W_std in WEIGHT_VARIANCES for b_std in BIAS_VARIANCES
                          for n_layers in [1, 2, 3, 4]
                          for do_jit in [True, False]))
  def testDenseFeatures(self, W_std, b_std, n_layers, do_jit):
    n, d = 4, 256
    rng = random.PRNGKey(1)
    x = self._get_init_data(rng, (n, d))

    dense_args = {'out_dim': 1, 'W_std': W_std, 'b_std': b_std}

    kernel_fn = stax.serial(*[stax.Dense(**dense_args)] * n_layers)[2]
    feature_fn = serial(*[DenseFeatures(**dense_args)] * n_layers)[1]

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x, None)
    f = feature_fn(x, [()] * n_layers)

    self.assertAllClose(k.nngp, f.nngp_feat @ f.nngp_feat.T)
    self.assertAllClose(k.ntk, f.ntk_feat @ f.ntk_feat.T)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              ' [Wstd{}_bstd{}_numlayers{}_{}_{}] '.format(
                  W_std, b_std, n_layers, relu_method, 'jit' if do_jit else ''),
          'W_std':
              W_std,
          'b_std':
              b_std,
          'n_layers':
              n_layers,
          'relu_method':
              relu_method,
          'do_jit':
              do_jit,
      } for W_std in WEIGHT_VARIANCES for b_std in BIAS_VARIANCES
                          for relu_method in
                          ['RANDFEAT', 'POLYSKETCH', 'PSRF', 'POLY', 'EXACT']
                          for n_layers in [1, 2, 3, 4]
                          for do_jit in [True, False]))
  def test_fc_relu_nngp_ntk(self, W_std, b_std, n_layers, relu_method, do_jit):
    rng = random.PRNGKey(1)
    n, d = 4, 256
    x = self._get_init_data(rng, (n, d))

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
    elif relu_method in ['EXACT', 'POLY']:
      pass
    else:
      raise ValueError(relu_method)

    _, _, kernel_fn = stax.serial(
        *[stax.Dense(**dense_args), stax.Relu()] * n_layers +
        [stax.Dense(**dense_args)])
    init_fn, feature_fn = serial(
        *[DenseFeatures(**dense_args),
          ReluFeatures(**relu_args)] * n_layers + [DenseFeatures(**dense_args)])

    rng2 = random.PRNGKey(2)
    _, feat_fn_inputs = init_fn(rng2, x.shape)

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x, None)
    k_nngp = k.nngp
    k_ntk = k.ntk

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
      test_utils.assert_close_matrices(self, k_nngp, k_nngp_approx, 0.1, 1.)
      test_utils.assert_close_matrices(self, k_ntk, k_ntk_approx, 0.1, 1.)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              ' [Wstd{}_bstd{}_numlayers{}_{}] '.format(
                  W_std, b_std, n_layers, 'jit' if do_jit else ''),
          'W_std':
              W_std,
          'b_std':
              b_std,
          'n_layers':
              n_layers,
          'do_jit':
              do_jit,
      } for W_std in WEIGHT_VARIANCES for b_std in BIAS_VARIANCES
                          for n_layers in [1, 2, 3, 4]
                          for do_jit in [True, False]))
  def test_conv_features(self, W_std, b_std, n_layers, do_jit):
    n, h, w, c = 3, 4, 5, 2
    rng = random.PRNGKey(1)
    x = self._get_init_data(rng, (n, h, w, c))

    conv_args = {
        'out_chan': 1,
        'filter_shape': (3, 3),
        'padding': 'SAME',
        'W_std': W_std,
        'b_std': b_std
    }

    kernel_fn = stax.serial(*[stax.Conv(**conv_args)] * n_layers)[2]
    feature_fn = serial(*[ConvFeatures(**conv_args)] * n_layers)[1]

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x)
    f = feature_fn(x, [()] * n_layers)

    k_nngp_approx = self._convert_image_feature_to_kernel(f.nngp_feat)
    k_ntk_approx = self._convert_image_feature_to_kernel(f.ntk_feat)

    self.assertAllClose(k.nngp, k_nngp_approx)
    self.assertAllClose(k.ntk, k_ntk_approx)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              ' [nlayers{}_{}] '.format(n_layers, 'jit' if do_jit else ''),
          'n_layers':
              n_layers,
          'do_jit':
              do_jit
      } for n_layers in [1, 2, 3, 4] for do_jit in [True, False]))
  def test_avgpool_features(self, n_layers, do_jit):
    n, h, w, c = 3, 32, 28, 2
    rng = random.PRNGKey(1)
    x = self._get_init_data(rng, (n, h, w, c))

    avgpool_args = {
        'window_shape': (2, 2),
        'strides': (2, 2),
        'padding': 'SAME'
    }

    kernel_fn = stax.serial(*[stax.AvgPool(**avgpool_args)] * n_layers)[2]
    feature_fn = serial(*[AvgPoolFeatures(**avgpool_args)] * n_layers)[1]

    if do_jit:
      kernel_fn = jit(kernel_fn)
      feature_fn = jit(feature_fn)

    k = kernel_fn(x)
    f = feature_fn(x, [()] * n_layers)

    k_nngp_approx = self._convert_image_feature_to_kernel(f.nngp_feat)

    self.assertAllClose(k.nngp, k_nngp_approx)

  def test_flatten_features(self):
    n, h, w, c = 3, 32, 28, 2
    n_layers = 1
    rng = random.PRNGKey(1)
    x = self._get_init_data(rng, (n, h, w, c))

    kernel_fn = stax.serial(*[stax.Flatten()] * n_layers)[2]

    k = kernel_fn(x)

    feature_fn = serial(*[FlattenFeatures()] * n_layers)[1]

    f = feature_fn(x, [()] * n_layers)

    self.assertAllClose(k.nngp, f.nngp_feat @ f.nngp_feat.T)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              ' [Wstd{}_bstd{}_depth{}_{}_{}] '.format(
                  W_std, b_std, depth, relu_method, 'jit' if do_jit else ''),
          'W_std':
              W_std,
          'b_std':
              b_std,
          'depth':
              depth,
          'relu_method':
              relu_method,
          'do_jit':
              do_jit,
      } for W_std in WEIGHT_VARIANCES for b_std in BIAS_VARIANCES
                          for relu_method in ['PSRF'] for depth in [5]
                          for do_jit in [False]))
  def test_myrtle_network(self, W_std, b_std, relu_method, depth, do_jit):
    if relu_method in ['RANDFEAT', 'POLYSKETCH', 'PSRF']:
      import os
      os.environ['CUDA_VISIBLE_DEVICES'] = ''

    n, h, w, c = 2, 32, 32, 3
    rng = random.PRNGKey(1)
    x = self._get_init_data(rng, (n, h, w, c))

    layer_factor = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}

    def _get_myrtle_kernel_fn():
      conv = functools.partial(stax.Conv,
                               W_std=W_std,
                               b_std=b_std,
                               padding='SAME')

      layers = []
      layers += [conv(1, (3, 3)), stax.Relu()] * layer_factor[depth][0]
      layers += [stax.AvgPool((2, 2), strides=(2, 2))]
      layers += [conv(1, (3, 3)), stax.Relu()] * layer_factor[depth][1]
      layers += [stax.AvgPool((2, 2), strides=(2, 2))]
      layers += [conv(1, (3, 3)), stax.Relu()] * layer_factor[depth][2]
      layers += [stax.AvgPool((2, 2), strides=(2, 2))] * 3
      layers += [stax.Flatten(), stax.Dense(1, W_std=W_std, b_std=b_std)]

      return stax.serial(*layers)

    def _get_myrtle_feature_fn(**relu_args):
      conv = functools.partial(ConvFeatures, W_std=W_std, b_std=b_std)
      layers = []
      layers += [conv(1, (3, 3)), ReluFeatures(**relu_args)
                ] * layer_factor[depth][0]
      layers += [AvgPoolFeatures((2, 2), strides=(2, 2))]
      layers += [conv(1, (3, 3)), ReluFeatures(**relu_args)
                ] * layer_factor[depth][1]
      layers += [AvgPoolFeatures((2, 2), strides=(2, 2))]
      layers += [conv(1, (3, 3)), ReluFeatures(**relu_args)
                ] * layer_factor[depth][2]
      layers += [AvgPoolFeatures((2, 2), strides=(2, 2))] * 3
      layers += [FlattenFeatures(), DenseFeatures(1, W_std=W_std, b_std=b_std)]

      return serial(*layers)

    _, _, kernel_fn = _get_myrtle_kernel_fn()

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
    elif relu_method in ['EXACT', 'POLY']:
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
      test_utils.assert_close_matrices(self, k_nngp, k_nngp_approx, 0.15, 1.)
      test_utils.assert_close_matrices(self, k_ntk, k_ntk_approx, 0.15, 1.)


if __name__ == "__main__":
  absltest.main()
