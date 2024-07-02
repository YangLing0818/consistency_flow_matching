import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 64
  training.n_iters = 2400001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075

  sampling.sigma_variance = 0.0 # NOTE: XC: sigma variance for turning ODe to SDE
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'euler'
  sampling.ode_tol = 1e-5
  sampling.sample_N = 1000

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  evaluate.batch_size = 512
  evaluate.enable_sampling = False
  evaluate.enable_figures_only = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  # predefined_z
  config.eval.predefined_z_path = "None" # when sampling uncurated images using the same noise
  
  # evaluation: rectflow_fid or cleanfid
  eval = config.eval
  # rectflow_fid
  eval.rectflow_fid = False
  # cleanfid
  eval.clean_fid = ml_collections.ConfigDict()
  eval.clean_fid.enabled = False
  eval.clean_fid.png_base_dir = "./cache_images"
  eval.clean_fid.custom_stat = ml_collections.ConfigDict()
  eval.clean_fid.custom_stat.dataset_name1 = "None"
  eval.clean_fid.custom_stat.dataset_name2 = "None"
  eval.clean_fid.custom_stat.dataset_name3 = "None"
  eval.clean_fid.custom_stat.dataset_name4 = "None"

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'LSUN'
  data.image_size = 256
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 3
  data.root_path = 'YOUR_ROOT_PATH'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  return config
