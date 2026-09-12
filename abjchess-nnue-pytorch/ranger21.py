"""Ranger21 optimizer used by the NNUE trainer."""

import collections
import math

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def normalize_gradient(x, use_channels=False, epsilon=1e-8):
  """Normalize gradients by standard deviation."""
  size = x.dim()
  if size > 1 and use_channels:
    s = x.std(dim=tuple(range(1, size)), keepdim=True) + epsilon
    x.div_(s)
  elif torch.numel(x) > 2:
    s = x.std() + epsilon
    x.div_(s)
  return x


def centralize_gradient(x, gc_conv_only=False):
  """Gradient centralization."""
  size = x.dim()
  if gc_conv_only:
    if size > 3:
      x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
  else:
    if size > 1:
      x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
  return x


class Ranger21(Optimizer):
  def __init__(
      self,
      params,
      lr,
      lookahead_active=True,
      lookahead_mergetime=5,
      lookahead_blending_alpha=0.5,
      lookahead_load_at_validation=False,
      use_madgrad=False,
      use_adabelief=False,
      softplus=True,
      beta_softplus=50,
      using_gc=True,
      using_normgc=True,
      gc_conv_only=False,
      normloss_active=True,
      normloss_factor=1e-4,
      use_adaptive_gradient_clipping=True,
      agc_clipping_value=1e-2,
      agc_eps=1e-3,
      betas=(0.9, 0.999),
      momentum_type="pnm",
      pnm_momentum_factor=1.0,
      momentum=0.9,
      eps=1e-8,
      num_batches_per_epoch=None,
      num_epochs=None,
      use_warmup=True,
      num_warmup_iterations=None,
      warmdown_active=True,
      warmdown_start_pct=0.72,
      warmdown_min_lr=3e-5,
      weight_decay=1e-4,
      decay_type="stable",
      warmup_type="linear",
      warmup_pct_default=0.22,
      logging_active=True):
    if use_madgrad:
      raise NotImplementedError("This local Ranger21 port implements the AdamW/PNM path only.")
    if lr <= 0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if eps <= 0:
      raise ValueError("Invalid eps: {}".format(eps))
    if not 0.0 <= lookahead_blending_alpha <= 1.0:
      raise ValueError("Invalid lookahead blending alpha: {}".format(lookahead_blending_alpha))
    if lookahead_mergetime < 1:
      raise ValueError("Invalid lookahead merge time: {}".format(lookahead_mergetime))

    defaults = dict(
      lr=lr,
      momentum=momentum,
      betas=betas,
      eps=eps,
      weight_decay=weight_decay,
    )
    super().__init__(params, defaults)

    self.logging = logging_active
    self.use_madgrad = False
    self.core_engine = "AdamW"
    self.use_adabelief = use_adabelief
    self.eps = eps

    self.softplus = softplus
    self.beta_softplus = beta_softplus

    self.normloss_active = normloss_active
    self.normloss_factor = normloss_factor

    self.lookahead_active = lookahead_active
    self.lookahead_mergetime = lookahead_mergetime
    self.lookahead_step = 0
    self.lookahead_alpha = lookahead_blending_alpha
    self.lookahead_validation_load = lookahead_load_at_validation

    self.agc_active = use_adaptive_gradient_clipping
    self.agc_clip_val = agc_clipping_value
    self.agc_eps = agc_eps

    if num_batches_per_epoch is None or num_epochs is None:
      raise ValueError("Ranger21 needs num_batches_per_epoch and num_epochs")
    self.num_batches_per_epoch = int(num_batches_per_epoch)
    self.num_epochs = int(num_epochs)
    if self.num_batches_per_epoch <= 0 or self.num_epochs <= 0:
      raise ValueError("Ranger21 schedule dimensions must be positive")
    self.total_iterations = self.num_epochs * self.num_batches_per_epoch

    self.starting_lr = lr
    self.current_lr = lr

    self.use_warmup = use_warmup
    self.warmup_complete = False
    self.warmup_type = warmup_type
    self.warmup_pct_default = warmup_pct_default
    if num_warmup_iterations is None:
      beta_warmup_iters = math.ceil(2 / (1 - betas[1]))
      beta_pct = beta_warmup_iters / self.total_iterations
      if beta_pct > 0.45:
        self.num_warmup_iters = int(self.warmup_pct_default * self.total_iterations)
      else:
        self.num_warmup_iters = beta_warmup_iters
    else:
      self.num_warmup_iters = int(num_warmup_iterations)
    if self.num_warmup_iters < 1:
      self.num_warmup_iters = 1

    self.min_lr = warmdown_min_lr
    self.warmdown_lr_delta = self.starting_lr - self.min_lr
    self.warmdown_active = warmdown_active
    if self.warmdown_active:
      self.warm_down_start_pct = warmdown_start_pct
      self.start_warm_down = int(self.warm_down_start_pct * self.total_iterations)
      self.warmdown_total_iterations = self.total_iterations - self.start_warm_down
      if self.warmdown_total_iterations < 1:
        self.warmdown_total_iterations = 1
      self.warmdown_displayed = False
    self.warmup_curr_pct = 0.01

    self.current_epoch = 0
    self.current_iter = 0
    self.use_gc = using_gc
    self.use_gcnorm = using_normgc
    self.gc_conv_only = gc_conv_only
    self.epoch_count = 0
    self.momentum_pnm = momentum_type == "pnm"
    self.pnm_momentum = pnm_momentum_factor
    self.decay = weight_decay
    self.decay_type = decay_type
    self.param_size = 0
    self.tracking_lr = []
    if self.logging:
      self.tracking_variance_sum = []
      self.tracking_variance_normalized = []

    self.show_settings()

  def __setstate__(self, state):
    super().__setstate__(state)

  def show_settings(self):
    print("Ranger21 optimizer ready with following settings:\n")
    print("Core optimizer = {}".format(self.core_engine))
    print("Learning rate of {}\n".format(self.starting_lr))
    print(
      "Important - num_epochs of training = ** {} epochs **\n"
      "please confirm this is correct or warmup and warmdown will be off\n".format(self.num_epochs)
    )
    if self.use_adabelief:
      print("using AdaBelief for variance computation")
    if self.use_warmup:
      print("{} warmup over {} iterations\n".format(self.warmup_type, self.num_warmup_iters))
    if self.lookahead_active:
      print(
        "Lookahead active, merging every {} steps, with blend factor of {}".format(
          self.lookahead_mergetime, self.lookahead_alpha
        )
      )
    if self.normloss_active:
      print("Norm Loss active, factor = {}".format(self.normloss_factor))
    if self.decay:
      print("Stable weight decay of {}".format(self.decay))
    print("Gradient Centralization = {}\n".format("On" if self.use_gc else "Off"))
    print("Adaptive Gradient Clipping = {}".format(self.agc_active))
    if self.agc_active:
      print("\tclipping value of {}".format(self.agc_clip_val))
      print("\tsteps for clipping = {}".format(self.agc_eps))
    if self.warmdown_active:
      print(
        "\nWarm-down: Linear warmdown, starting at {}%, iteration {} of {}".format(
          self.warm_down_start_pct * 100, self.start_warm_down, self.total_iterations
        )
      )
      print("warm down will decay until {} lr".format(self.min_lr))

  def clear_cache(self):
    print("clearing lookahead cache...")
    for group in self.param_groups:
      for p in group["params"]:
        param_state = self.state[p]
        if "lookahead_params" not in param_state:
          print("no lookahead cache present.")
          return
        param_state["lookahead_params"] = torch.zeros_like(p.data)
    print("lookahead cache cleared")

  def clear_and_load_backup(self):
    for group in self.param_groups:
      for p in group["params"]:
        param_state = self.state[p]
        p.data.copy_(param_state["backup_params"])
        del param_state["backup_params"]

  def backup_and_load_cache(self):
    for group in self.param_groups:
      for p in group["params"]:
        param_state = self.state[p]
        param_state["backup_params"] = torch.zeros_like(p.data)
        param_state["backup_params"].copy_(p.data)
        p.data.copy_(param_state["lookahead_params"])

  def unit_norm(self, x):
    keepdim = True
    dim = None
    xlen = len(x.shape)
    if xlen <= 1:
      keepdim = False
    elif xlen in (2, 3):
      dim = 1
    elif xlen == 4:
      dim = (1, 2, 3)
    else:
      dim = tuple(range(1, xlen))
    return x.norm(dim=dim, keepdim=keepdim, p=2.0)

  def agc(self, p):
    p_norm = self.unit_norm(p).clamp_(self.agc_eps)
    g_norm = self.unit_norm(p.grad)
    max_norm = p_norm * self.agc_clip_val
    clipped_grad = p.grad * (max_norm / g_norm.clamp(min=1e-6))
    p.grad.detach().copy_(torch.where(g_norm > max_norm, clipped_grad, p.grad))

  def warmup_dampening(self, lr, step):
    if self.warmup_type is None:
      return lr
    if step > self.num_warmup_iters:
      if not self.warmup_complete:
        if self.warmup_curr_pct != 1.0:
          print("Error - lr did not achieve full set point from warmup, currently {}".format(self.warmup_curr_pct))
        self.warmup_complete = True
        print("\n** Ranger21 update = Warmup complete - lr set to {}\n".format(lr))
      return lr
    if self.warmup_type == "linear":
      self.warmup_curr_pct = min(1.0, step / self.num_warmup_iters)
      new_lr = lr * self.warmup_curr_pct
      self.current_lr = new_lr
      return new_lr
    raise ValueError("warmup type {} not implemented.".format(self.warmup_type))

  def get_warm_down(self, lr, iteration):
    if iteration < self.start_warm_down:
      return lr
    if not self.warmdown_displayed:
      print("\n** Ranger21 update: Warmdown starting now. Current iteration = {}....\n".format(iteration))
      self.warmdown_displayed = True
    warmdown_iteration = (iteration + 1) - self.start_warm_down
    if warmdown_iteration < 1:
      warmdown_iteration = 1
    warmdown_pct = warmdown_iteration / (self.warmdown_total_iterations + 1)
    if warmdown_pct > 1.0:
      warmdown_pct = 1.0
    new_lr = self.starting_lr - self.warmdown_lr_delta * warmdown_pct
    if new_lr < self.min_lr:
      new_lr = self.min_lr
    self.current_lr = new_lr
    return new_lr

  def track_epochs(self, iteration):
    self.current_iter += 1
    if self.current_iter % self.num_batches_per_epoch == 0:
      self.current_iter = 0
      self.epoch_count += 1
      self.tracking_lr.append(self.current_lr)
      if self.lookahead_active and self.lookahead_validation_load:
        self.backup_and_load_cache()

  def update_lr(self, lr, step):
    if self.use_warmup and not self.warmup_complete:
      lr = self.warmup_dampening(lr, step)
    if self.warmdown_active:
      lr = self.get_warm_down(lr, step)
      assert lr > 0, "lr went negative"
    return lr

  def apply_weightdecay_normloss_updates(self, p, variance_normalized, lr, decay):
    if decay:
      if self.decay_type != "stable":
        raise ValueError("Only stable weight decay is implemented.")
      p.data.mul_(1 - decay * lr / variance_normalized)
    if self.normloss_active:
      unorm = self.unit_norm(p.data)
      correction = 2 * self.normloss_factor * (1 - torch.div(1, unorm + self.eps))
      p.mul_(1 - lr * correction)

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None and isinstance(closure, collections.abc.Callable):
      with torch.enable_grad():
        loss = closure()

    param_size = 0
    variance_ma_sum = None

    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        param_size += p.numel()
        if self.agc_active:
          self.agc(p)

        grad = p.grad
        if grad.is_sparse:
          raise RuntimeError("Ranger21 does not support sparse gradients")

        state = self.state[p]
        if len(state) == 0:
          state["step"] = 0
          state["grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          state["variance_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          if self.lookahead_active:
            state["lookahead_params"] = torch.zeros_like(p.data)
            state["lookahead_params"].copy_(p.data)
          if self.use_adabelief:
            state["variance_ma_belief"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          if self.momentum_pnm:
            state["neg_grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        if self.use_gc:
          grad = centralize_gradient(grad, gc_conv_only=self.gc_conv_only)
        if self.use_gcnorm:
          grad = normalize_gradient(grad)

        state["step"] += 1
        beta1, beta2 = group["betas"]
        grad_ma = state["grad_ma"]
        variance_ma = state["variance_ma"]

        if self.use_adabelief:
          grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)
          grad_residual = grad - grad_ma
          state["variance_ma_belief"].mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

        variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction2 = 1 - beta2 ** state["step"]
        variance_ma_debiased_sum = (variance_ma / bias_correction2).sum()
        variance_ma_sum = variance_ma_debiased_sum if variance_ma_sum is None else variance_ma_sum + variance_ma_debiased_sum

    if not self.param_size:
      self.param_size = param_size
      print("params size saved")
      print("total param groups = {}".format(len(self.param_groups)))
    if not self.param_size:
      raise ValueError("failed to set param size")

    variance_normalized = torch.sqrt(variance_ma_sum / param_size)
    if torch.isnan(variance_normalized).item():
      raise RuntimeError("hit nan for variance_normalized")

    if self.logging:
      self.tracking_variance_sum.append(variance_ma_sum.detach().item())
      self.tracking_variance_normalized.append(variance_normalized.detach().item())

    last_step = 0
    for group in self.param_groups:
      first_p_with_grad = next((p for p in group["params"] if p.grad is not None), None)
      if first_p_with_grad is None:
        continue
      group_step = self.state[first_p_with_grad].get("step", 0)
      lr = self.update_lr(group["lr"], group_step)
      decay = group["weight_decay"]
      eps = group["eps"]
      beta1, beta2 = group["betas"]

      for p in group["params"]:
        if p.grad is None:
          continue

        state = self.state[p]
        step = state["step"]
        last_step = step
        self.apply_weightdecay_normloss_updates(p, variance_normalized, lr, decay)

        grad = p.grad
        if self.momentum_pnm:
          if step % 2 == 1:
            grad_ma = state["grad_ma"]
            neg_grad_ma = state["neg_grad_ma"]
          else:
            grad_ma = state["neg_grad_ma"]
            neg_grad_ma = state["grad_ma"]
        else:
          grad_ma = state["grad_ma"]
          neg_grad_ma = None

        if self.use_gc:
          grad = centralize_gradient(grad, gc_conv_only=self.gc_conv_only)
        if self.use_gcnorm:
          grad = normalize_gradient(grad)

        if not self.use_adabelief:
          grad_ma.mul_(beta1 ** 2).add_(grad, alpha=1 - beta1 ** 2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        variance_ma = state["variance_ma"]
        if self.use_adabelief:
          variance_ma = state["variance_ma_belief"]
        denom = (variance_ma.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        if self.softplus:
          denom = F.softplus(denom, beta=self.beta_softplus)

        step_size = lr / bias_correction1
        if self.momentum_pnm:
          noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
          pnmomentum = (
            grad_ma.mul(1 + self.pnm_momentum)
            .add(neg_grad_ma, alpha=-self.pnm_momentum)
            .mul(1 / noise_norm)
          )
          p.addcdiv_(pnmomentum, denom, value=-step_size)
        else:
          p.addcdiv_(grad_ma, denom, value=-step_size)

    if self.lookahead_active:
      self.lookahead_process_step()
    self.track_epochs(last_step)
    return loss

  def lookahead_process_step(self):
    if not self.lookahead_active:
      return
    self.lookahead_step += 1
    if self.lookahead_step < self.lookahead_mergetime:
      return
    self.lookahead_step = 0
    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        param_state = self.state[p]
        p.data.mul_(self.lookahead_alpha).add_(
          param_state["lookahead_params"],
          alpha=1.0 - self.lookahead_alpha,
        )
        param_state["lookahead_params"].copy_(p.data)
