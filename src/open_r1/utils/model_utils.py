import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
)

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig, ScriptArguments


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_processor(
    model_args: ModelConfig,
    training_args: SFTConfig | GRPOConfig,
    script_args: ScriptArguments,
) -> AutoProcessor:
    """Get the processor for VLM models."""

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set the image processor resize size
    if script_args.image_resize is not None and "resolution_max_side" in script_args.image_resize:
        processor.image_processor.size = {
            "longest_edge": script_args.image_resize["resolution_max_side"]
        }
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.truncation_side = "right"
        processor.tokenizer.padding_side = "right"

    if training_args.chat_template is not None:
        processor.chat_template = training_args.chat_template

    return processor


def get_model(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig
) -> AutoModelForCausalLM | AutoModelForImageTextToText:
    """Get the model - supports both text-only and vision-language models"""
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Check if this is a VLM model using the explicit flag
    if hasattr(training_args, "vision_model") and training_args.vision_model:
        # Load as vision-language model
        model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
    else:
        # Load as text-only model
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

    return model
