"""LLaMA-MindSkip model configuration"""

from typing import Union
from transformers import LlamaConfig
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)


class LlamaMindSkipConfig(PretrainedConfig):
    model_type = "llama_mindskip"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            # 🔍
            is_mindskip: list = None,
            granularity: str = "sequence", 
            gradient_scale: float = 5e-3, 
            threshold: float = 0.5, 
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # 🔍
        self.is_mindskip = [False for _ in range(num_hidden_layers)] if is_mindskip is None else is_mindskip
        self.granularity = granularity
        self.gradient_scale = gradient_scale
        self.threshold = threshold
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`, " f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    # 🔍
    def from_llama_config(
            config: LlamaConfig,
            is_mindskip: list = None,
            mindskip_capacity: Union[float, list] = 0.9,
            mindskip_loss_coefficient: float = 0.1,
            mindskip_loss_type: str = "self",
            rescale_hidden_states: bool = True,
            scale_factor: float = 1.0,
            scale_gap: float = 1.0,
            gate_init_method: str = "zero",
            **kwargs
    ):
        return LlamaMindSkipConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            rms_norm_eps=config.rms_norm_eps,
            use_cache=config.use_cache,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pretraining_tp=config.pretraining_tp,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            # 🔍
            is_mindskip=is_mindskip,
            mindskip_capacity=mindskip_capacity,
            mindskip_loss_coefficient=mindskip_loss_coefficient,
            mindskip_loss_type=mindskip_loss_type,
            rescale_hidden_states=rescale_hidden_states,
            scale_factor=scale_factor,
            scale_gap=scale_gap,
            gate_init_method=gate_init_method,
            **kwargs
        )

