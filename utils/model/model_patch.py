import inspect
from types import MethodType
from typing import Optional

import torch
from torch import nn


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        original_dtype = input_tensor.dtype
        return (input_tensor >= 0).to(original_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _filter_supported_kwargs(module, kwargs):
    signature = inspect.signature(module.forward)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs

    supported_kwargs = {}
    for key, value in kwargs.items():
        if key in signature.parameters:
            supported_kwargs[key] = value
    return supported_kwargs


def _call_self_attention(
    layer,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    extra_kwargs=None,
):
    attn_kwargs = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_value": past_key_value,
        "output_attentions": output_attentions,
        "use_cache": use_cache,
    }
    if extra_kwargs:
        attn_kwargs.update(extra_kwargs)

    return layer.self_attn(**_filter_supported_kwargs(layer.self_attn, attn_kwargs))


def _parse_granularity(granularity):
    granularity = str(granularity or "attn_sequence").lower().replace("-", "_")
    route_level = "token" if "token" in granularity else "sequence"

    if "block" in granularity:
        route_target = "block"
    elif "mlp" in granularity:
        route_target = "mlp"
    else:
        route_target = "attn"

    return route_target, route_level


def _compute_routing_state(layer, hidden_states, training):
    if layer.route_level == "token":
        routing_logits = layer.router(hidden_states)
    else:
        routing_logits = layer.router(hidden_states.mean(dim=1, keepdim=True))

    routing_scores = torch.sigmoid(routing_logits)
    if training:
        routing_mask = STEFunction.apply(routing_scores - layer.threshold)
    else:
        routing_mask = (routing_scores - layer.threshold >= 0).to(routing_scores.dtype)

    current_capacity = routing_mask.float().mean()
    mod_capacity = current_capacity.detach().cpu().item()
    if not training or layer.target_mod_capacity is None:
        mod_loss = current_capacity.new_zeros(()) if training else None
    else:
        target_capacity = current_capacity.new_tensor(float(layer.target_mod_capacity))
        mod_loss = torch.relu(current_capacity - target_capacity) * layer.gradient_scale

    return routing_mask, mod_capacity, mod_loss


def _apply_routing_mask(residual, update, routing_mask):
    if routing_mask.dim() == 2:
        routing_mask = routing_mask.unsqueeze(-1)
    return residual + update * routing_mask.to(update.dtype)


def _patched_decoder_layer_forward(
    self,
    hidden_states,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
):
    self._last_mod_capacity = None
    self._last_mod_loss = None

    block_residual = hidden_states
    attn_inputs = self.input_layernorm(hidden_states)

    routing_mask = None
    if self.is_mod:
        routing_mask, mod_capacity, mod_loss = _compute_routing_state(self, attn_inputs, training=self.training)
        self._last_mod_capacity = mod_capacity
        self._last_mod_loss = mod_loss

    attn_outputs = _call_self_attention(
        self,
        attn_inputs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        extra_kwargs=kwargs,
    )
    hidden_states, self_attn_weights, present_key_value = attn_outputs

    if self.is_mod and self.route_target == "attn":
        hidden_states = _apply_routing_mask(block_residual, hidden_states, routing_mask)
    else:
        hidden_states = block_residual + hidden_states

    mlp_residual = hidden_states
    mlp_inputs = self.post_attention_layernorm(hidden_states)
    mlp_outputs = self.mlp(mlp_inputs)

    if self.is_mod and self.route_target == "mlp":
        hidden_states = _apply_routing_mask(mlp_residual, mlp_outputs, routing_mask)
    else:
        hidden_states = mlp_residual + mlp_outputs

    if self.is_mod and self.route_target == "block":
        hidden_states = _apply_routing_mask(block_residual, hidden_states - block_residual, routing_mask)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def _reset_router_state(layer):
    layer._last_mod_capacity = None
    layer._last_mod_loss = None


def _patch_single_layer(layer, config, layer_idx):
    if getattr(layer, "_router_tuning_patched", False):
        return

    layer.config = config
    layer.layer_idx = layer_idx
    layer.is_mod = bool(config.is_mod[layer_idx])
    layer.granularity = getattr(config, "granularity", "attn_sequence")
    layer.route_target, layer.route_level = _parse_granularity(layer.granularity)

    if layer.is_mod and not hasattr(layer, "router"):
        layer.router = nn.Linear(config.hidden_size, 1, bias=False)
        layer.gradient_scale = getattr(config, "gradient_scale", 0.0) or 0.0
        target_capacity = getattr(config, "mod_capacity", None)
        if isinstance(target_capacity, (list, tuple)):
            target_capacity = target_capacity[layer_idx]
        layer.target_mod_capacity = target_capacity
        layer.threshold = getattr(config, "threshold", 0.5)
    elif layer.is_mod:
        layer.gradient_scale = getattr(config, "gradient_scale", getattr(layer, "gradient_scale", 0.0)) or 0.0
        target_capacity = getattr(config, "mod_capacity", getattr(layer, "target_mod_capacity", None))
        if isinstance(target_capacity, (list, tuple)):
            target_capacity = target_capacity[layer_idx]
        layer.target_mod_capacity = target_capacity
        layer.threshold = getattr(config, "threshold", getattr(layer, "threshold", 0.5))

    _reset_router_state(layer)
    layer.forward = MethodType(_patched_decoder_layer_forward, layer)
    layer._router_tuning_patched = True


def _get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Unable to locate decoder layers for Router-Tuning patching.")


def _looks_like_supported_decoder_layer(layer):
    required_attributes = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
    return all(hasattr(layer, attribute) for attribute in required_attributes)


def supports_router_tuning_patch(model):
    try:
        layers = _get_decoder_layers(model)
    except ValueError:
        return False

    if len(layers) == 0:
        return False
    return _looks_like_supported_decoder_layer(layers[0])


def _patched_model_forward(self, *args, **kwargs):
    bound_arguments = inspect.signature(self._router_tuning_original_forward).bind_partial(*args, **kwargs)
    past_key_values = bound_arguments.arguments.get("past_key_values")

    if past_key_values is None:
        for layer in self._router_tuning_layers:
            _reset_router_state(layer)

    outputs = self._router_tuning_original_forward(*args, **kwargs)

    mod_capacities = []
    mod_loss = None
    for layer in self._router_tuning_layers:
        if getattr(layer, "_last_mod_capacity", None) is not None:
            mod_capacities.append(layer._last_mod_capacity)
        if self.training and getattr(layer, "_last_mod_loss", None) is not None:
            mod_loss = layer._last_mod_loss if mod_loss is None else mod_loss + layer._last_mod_loss

    averaged_capacity = None
    if mod_capacities:
        averaged_capacity = sum(mod_capacities) / len(mod_capacities)

    if hasattr(outputs, "mod_capacity"):
        outputs.mod_capacity = averaged_capacity
    elif not isinstance(outputs, tuple):
        outputs.mod_capacity = averaged_capacity

    if self.training and mod_loss is not None:
        if hasattr(outputs, "loss") and outputs.loss is not None:
            outputs.loss = outputs.loss + mod_loss
        if hasattr(outputs, "mod_losses"):
            outputs.mod_losses = mod_loss
        elif not isinstance(outputs, tuple):
            outputs.mod_losses = mod_loss

    return outputs


def apply_router_tuning_patch(model):
    if getattr(model, "_router_tuning_runtime_patched", False):
        return model

    layers = _get_decoder_layers(model)
    if len(model.config.is_mod) != len(layers):
        raise ValueError(
            f"`config.is_mod` length ({len(model.config.is_mod)}) does not match decoder layers ({len(layers)})."
        )
    if not _looks_like_supported_decoder_layer(layers[0]):
        raise ValueError(
            f"Unsupported decoder layer type for Router-Tuning patch: {layers[0].__class__.__name__}."
        )

    for layer_idx, layer in enumerate(layers):
        _patch_single_layer(layer, model.config, layer_idx)

    model._router_tuning_layers = layers
    model._router_tuning_original_forward = model.forward
    model.forward = MethodType(_patched_model_forward, model)
    model._router_tuning_runtime_patched = True
    return model
