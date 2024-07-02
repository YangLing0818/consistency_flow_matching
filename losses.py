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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import ConsistencyFM


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_rectified_flow_loss_fn(sde, train, reduce_mean=True, eps=1e-3):
  """Create a loss function for training with rectified flow.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A velocity model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    if sde.reflow_flag:
        z0 = batch[0]
        data = batch[1]
        batch = data.detach().clone()

    else:
        z0 = sde.get_z0(batch).to(batch.device)
    
    if sde.reflow_flag:
        if sde.reflow_t_schedule=='t0': ### distill for t = 0 (k=1)
            t = torch.zeros(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        elif sde.reflow_t_schedule=='t1': ### reverse distill for t=1 (fast embedding)
            t = torch.ones(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        elif sde.reflow_t_schedule=='uniform': ### train new rectified flow with reflow
            t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        elif type(sde.reflow_t_schedule)==int: ### k > 1 distillation
            t = torch.randint(0, sde.reflow_t_schedule, (batch.shape[0], ), device=batch.device) * (sde.T - eps) / sde.reflow_t_schedule + eps
        else:
            assert False, 'Not implemented'
    else:
        ### standard rectified flow loss
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps

    t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
    perturbed_data = t_expand * batch + (1.-t_expand) * z0
    target = batch - z0 
    
    model_fn = mutils.get_model_fn(model, train=train)
    score = model_fn(perturbed_data, t*999) ### Copy from models/utils.py 

    if sde.reflow_flag:
        ### we found LPIPS loss is the best for distillation when k=1; but good to have a try
        if sde.reflow_loss=='l2':
            ### train new rectified flow with reflow or distillation with L2 loss
            losses = torch.square(score - target)
        elif sde.reflow_loss=='lpips':
            assert sde.reflow_t_schedule=='t0'
            losses = sde.lpips_model(z0 + score, batch)
        elif sde.reflow_loss=='lpips+l2':
            assert sde.reflow_t_schedule=='t0'
            lpips_losses = sde.lpips_model(z0 + score, batch).view(batch.shape[0], 1)
            l2_losses = torch.square(score - target).view(batch.shape[0], -1).mean(dim=1, keepdim=True)
            losses = lpips_losses + l2_losses
        else:
            assert False, 'Not implemented'
    else:
        losses = torch.square(score - target)
    
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_consistency_flow_matching_loss_fn(sde, train, reduce_mean=True, eps=1e-3):
  """Create a loss function for training with rectified flow.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  hyperparameter = sde.consistencyfm_hyperparameters

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A velocity model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    assert sde.reflow_flag == False, "not implemented"

    z0 = sde.get_z0(batch).to(batch.device)
    
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    r = torch.clamp(t + hyperparameter["delta"], max=1.0)

    t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
    r_expand = r.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
    xt = t_expand * batch + (1.-t_expand) * z0
    xr = r_expand * batch + (1.-r_expand) * z0
    
    segments = torch.linspace(0, 1, hyperparameter["num_segments"] + 1, device=batch.device)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1) # .clamp(min=1) prevents the inclusion of 0 in indices.
    segment_ends = segments[seg_indices]
    
    segment_ends_expand = segment_ends.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
    x_at_segment_ends = segment_ends_expand * batch + (1.-segment_ends_expand) * z0
    
    def f_euler(t_expand, segment_ends_expand, xt, vt):
      return xt + (segment_ends_expand - t_expand) * vt
    def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
      if (threshold, int) and threshold == 0:
        return x_at_segment_ends
      
      less_than_threshold = t_expand < threshold
      
      res = (
        less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt)
        + (~less_than_threshold) * x_at_segment_ends
        )
      return res
    
    model.train(train)
    model_fn = model
    
    rng_state = torch.cuda.get_rng_state()
    vt = model_fn(xt, t*999)
    
    torch.cuda.set_rng_state(rng_state) # Shared Dropout Mask
    with torch.no_grad():
      if (isinstance(hyperparameter["boundary"], int) 
          and hyperparameter["boundary"] == 0): # when hyperparameter["boundary"] == 0, vr is not needed
        vr = None
      else:
        vr = model_fn(xr, r*999)
        vr = torch.nan_to_num(vr)
      
    
    ft = f_euler(t_expand, segment_ends_expand, xt, vt)
    fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, hyperparameter["boundary"], x_at_segment_ends)

    ##### loss #####
    losses_f = torch.square(ft - fr)
    losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)
    
    
    def masked_losses_v(vt, vr, threshold, segment_ends, t):
      if (threshold, int) and threshold == 0:
        return 0
    
      less_than_threshold = t_expand < threshold
      
      far_from_segment_ends = (segment_ends - t) > 1.01 * hyperparameter["delta"]
      far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
      
      losses_v = torch.square(vt - vr)
      losses_v = less_than_threshold * far_from_segment_ends * losses_v
      losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
      
      return losses_v
    
    losses_v = masked_losses_v(vt, vr, hyperparameter["boundary"], segment_ends, t)

    loss = torch.mean(
      losses_f + hyperparameter["alpha"] * losses_v
    )
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, ConsistencyFM):
      loss_fn = get_consistency_flow_matching_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
