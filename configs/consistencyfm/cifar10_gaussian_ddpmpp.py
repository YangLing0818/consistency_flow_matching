# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training ConsistencyFM on CIFAR-10 with DDPM++."""
import ml_collections
from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.batch_size = 512
  training.log_freq = 2500
  training.eval_freq = 20000
  training.sde = 'consistencyfm'
  training.continuous = False
  training.n_iters = 100001
  training.snapshot_freq = 20000
  training.reduce_mean = False
  training.snapshot_freq_for_preemption = 20000

  # sampling
  sampling = config.sampling
  sampling.method = 'rectified_flow'
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'euler' # rk45 or euler
  sampling.ode_tol = 1e-5 # only used for rk45
  sampling.sample_N = 2

  # data
  data = config.data
  data.centered = True
  
  # consistencyfm
  config.consistencyfm = consistencyfm = ml_collections.ConfigDict()
  consistencyfm.delta_t = 1e-3
  consistencyfm.num_segments = 2
  consistencyfm.boundary = 0.9
  consistencyfm.alpha = 1e-5

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.999999
  model.dropout = 0.0
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  return config
