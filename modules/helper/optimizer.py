# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import math
import warnings
import weakref
from collections import OrderedDict
from collections import abc as container_abcs
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, Optional, Tuple

import paddle

def cast(x, dtype):
    if not isinstance(dtype, paddle.framework.dtype.dtype):
        dtype = paddle.framework.convert_np_dtype_to_dtype_(dtype)
    if x.dtype == dtype:
        return x
    return paddle.cast(x, dtype)


# copy_ 和 set_value 不支持不同类型的copy

paddle.Tensor.cast = cast

__all__ = [
    "Optimizer",
    "register_optimizer_step_pre_hook",
    "register_optimizer_step_post_hook",
    "Lion",
    "RemovableHandle",
]

# patch lerp_, lerp_ donot support float16


def lerp_(x, y, weight, name=None):
    if x.dtype == paddle.float16 or y.dtype == paddle.float16:
        return x.add_(weight * (y - x))
    else:
        return paddle.tensor.math.lerp_(x, y, weight, name=name)


paddle.Tensor.lerp_ = lerp_

def lerp(x, y, weight):
    return x + weight * (y - x)


class RemovableHandle(object):
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (dict): An additional dictionary whose keys will be deleted
            when the same keys are removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref = weakref.ref(extra_dict) if extra_dict is not None else None

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        if self.extra_dict_ref is not None:
            extra_dict = self.extra_dict_ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        return (
            (self.hooks_dict_ref(), self.id)
            if self.extra_dict_ref is None
            else (self.hooks_dict_ref(), self.id, self.extra_dict_ref())
        )

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        self.extra_dict_ref = None if len(state) < 3 else weakref.ref(OrderedDict() if state[2] is None else state[2])

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


_global_optimizer_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_optimizer_post_hooks: Dict[int, Callable] = OrderedDict()


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def register_optimizer_step_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a pre hook common to all optimizers. The hook should have the following
    signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`utils.hooks.RemoveableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle


def register_optimizer_step_post_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a post hook common to all optimizers. The hook should have the following
    signature::

        hook(optimizer, args, kwargs) -> None

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`utils.hooks.RemoveableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_optimizer_post_hooks)
    _global_optimizer_post_hooks[handle.id] = hook
    return handle


class Optimizer(object):
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        self.defaults = defaults
        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._optimizer_step_post_hooks: Dict[int, Callable] = OrderedDict()

        if isinstance(params, paddle.Tensor):
            raise TypeError(
                "params argument given to the optimizer should be "
                "an iterable of Tensors or dicts, but got " + type(params)
            )

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        self._warned_capturable_if_run_uncaptured = True

    def __getstate__(self):
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_optimizer_step_pre_hooks" not in self.__dict__:
            self._optimizer_step_pre_hooks = OrderedDict()
        if "_optimizer_step_post_hooks" not in self.__dict__:
            self._optimizer_step_post_hooks = OrderedDict()
        self.defaults.setdefault("differentiable", False)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += "Parameter Group {0}\n".format(i)
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += "    {0}: {1}\n".format(key, group[key])
        format_string += ")"
        return format_string

    def _optimizer_step_code(self):
        """Entry point for `profile.profiler`.

        When python tracing is enabled the profiler will hook into this
        function at the CPython level to inspect the optimizer's parameters and
        param groups. It is called it after `step()` since many optimizers
        lazily initialize state.

        This is a workaround due to lack of a proper step hook on the optimizer,
        and will be removed if it exists.
        """
        pass

    def register_step_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        r"""Register an optimizer step pre hook which will be called before
        optimizer step. It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._optimizer_step_pre_hooks)
        self._optimizer_step_pre_hooks[handle.id] = hook
        return handle

    def register_step_post_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        r"""Register an optimizer step post hook which will be called after optimizer step.
        It should have the following signature::

            hook(optimizer, args, kwargs) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._optimizer_step_post_hooks)
        self._optimizer_step_post_hooks[handle.id] = hook
        return handle

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {id(p): i for i, p in enumerate(group["params"], start_index) if id(p) not in param_mappings}
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, paddle.Tensor) else k): v for k, v in self.state.items()
        }
        return {
            "state": packed_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of " "parameter groups")
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group " "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g["params"] for g in saved_groups)),
                chain.from_iterable((g["params"] for g in groups)),
            )
        }

        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, paddle.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
                if key != "step":
                    if param.is_floating_point():
                        value = value.cast(param.dtype)
                    if value.place != param.place:
                        value = value._to(param.place)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        foreach = self.defaults.get("foreach", False)

        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if hasattr(p.grad, "grad_fn") and p.grad.grad_fn is not None:
                            p.grad = p.grad.detach()
                        else:
                            p.grad.stop_gradient = True
                        if not foreach or p.grad.is_sparse():
                            p.grad.zero_()
                        else:
                            per_device_and_dtype_grads[p.grad.place][p.grad.dtype].append(p.grad)
        if foreach:
            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    for grad in grads:
                        grad.zero_()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group["params"]
        if isinstance(params, paddle.Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, paddle.Tensor):
                raise TypeError("optimizer can only optimize Tensors, " "but one of the params is " + type(param))
            if not self.defaults.get("differentiable", None) and not (param.is_leaf or param.retains_grad):
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " + name)
            else:
                param_group.setdefault(name, default)

        params = param_group["params"]
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; "
                "see github.com/pytorch/pytorch/issues/40967 for more information",
                stacklevel=3,
            )

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    clear_grad = zero_grad


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

    @paddle.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            with paddle.set_grad_enabled():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group["params"]):

                grad, lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                if grad.is_sparse():
                    raise RuntimeError("Lion does not support sparse gradients, please consider SparseLion instead")

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = paddle.zeros_like(p)

                exp_avg = state["exp_avg"]

                p.copy_(p * (1 - lr * wd), True)
                update = lerp(exp_avg, grad, 1 - beta1)
                p.copy_(p - lr * paddle.sign(update), True)
                exp_avg.copy_(lerp(exp_avg, grad, 1 - beta2), True)

        return loss


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @paddle.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with paddle.set_grad_enabled():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse():
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = paddle.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = paddle.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.copy_(
                    paddle.lerp(grad, exp_avg, beta1), True
                )
                exp_avg_sq.copy_(
                    paddle.lerp(grad**2, exp_avg_sq, beta2), True
                )              
                # exp_avg.scale_(beta1).add_(grad * (1.0 - beta1))
                # exp_avg_sq.scale_(beta2).add_(grad**2 * (1.0 - beta2))
                
                denom = exp_avg_sq.sqrt() + group["eps"]

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.copy_(
                    p - exp_avg / denom * step_size, True
                )
                # p.add_(-exp_avg / denom * step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.copy_(
                        (1 - group["lr"] * group["weight_decay"]) * p, True
                    )
                    # p.add_(p * (-group["lr"] * group["weight_decay"]))

        return loss


class Adafactor(Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://arxiv.org/abs/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
    `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (`Iterable[Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*):
            The external learning rate.
        eps (`Tuple[float, float]`, *optional*, defaults to (1e-30, 1e-3)):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
        weight_decay (`float`, *optional*, defaults to 0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

        - Training without LR warmup or clip_threshold is not recommended.

           - use scheduled LR warm-up to fixed LR
           - use clip_threshold=1.0 (https://arxiv.org/abs/1804.04235)
        - Disable relative updates
        - Use scale_parameter=False
        - Additional optimizer operations like gradient clipping should not be used alongside Adafactor

    Example:

    ```python
    Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    ```

    Others reported the following combination to work well:

    ```python
    Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    ```

    When using `lr=None` with [`Trainer`] you will most likely need to use [`~optimization.AdafactorSchedule`]
    scheduler as following:

    ```python
    from transformers.optimization import Adafactor, AdafactorSchedule

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)
    trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
    ```

    Usage:

    ```python
    # replace AdamW with Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    ```
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(axis=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return r_factor * c_factor

    @paddle.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with paddle.set_grad_enabled():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse():
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                if grad.dtype in {paddle.float16, paddle.bfloat16}:
                    grad = grad.cast(paddle.float32)

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = paddle.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = paddle.zeros(grad_shape[:-1], dtype=grad.dtype)
                        state["exp_avg_sq_col"] = paddle.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype)
                    else:
                        state["exp_avg_sq"] = paddle.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].cast(grad.dtype)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].cast(grad.dtype)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].cast(grad.dtype)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].cast(grad.dtype)

                p_data_fp32 = p
                if p.dtype in {paddle.float16, paddle.bfloat16}:
                    p_data_fp32 = p_data_fp32.cast(paddle.float32)

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.copy_(
                        lerp(update.mean(-1), exp_avg_sq_row, beta2t), True
                    )
                    # exp_avg_sq_row.scale_(beta2t).add_(update.mean(-1) * (1.0 - beta2t))
                    exp_avg_sq_col.copy_(
                        lerp(update.mean(-2), exp_avg_sq_col, beta2t), True
                    )                
                    # exp_avg_sq_col.scale_(beta2t).add_(update.mean(-2) * (1.0 - beta2t))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.copy_(update * grad, True)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.copy_(
                        lerp(update, exp_avg_sq, beta2t), True
                    )      
                    # exp_avg_sq.scale_(beta2t).add_(update * (1.0 - beta2t))
                    update = exp_avg_sq.rsqrt() * grad
                    
                update.copy_(
                    lr * update / (self._rms(update) / group["clip_threshold"]).clip(min=1.0), True
                )

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.copy_(
                        lerp(update, exp_avg, group["beta1"]), True
                    )
                    # exp_avg.scale_(group["beta1"]).add_(update * (1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.copy_(
                        p_data_fp32 * (1 - lr * group["weight_decay"]), True
                    )
                    # p_data_fp32.add_(p_data_fp32 * (-group["weight_decay"] * lr))

                p_data_fp32.copy_(
                    p_data_fp32 - update, True
                )
                # p_data_fp32.add_(-update)

                if p.dtype in {paddle.float16, paddle.bfloat16}:
                    p.copy_(p_data_fp32.cast(p.dtype), True)

        return loss
