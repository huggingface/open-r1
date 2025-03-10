# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Union
from datasets import enable_progress_bars, disable_progress_bars
import torch
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from torch.utils.data import DataLoader, RandomSampler, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_liger_kernel_available, is_peft_available

import deepspeed
import trl
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from open_r1.trainers.job_launcher import SGLangSlurmJobLauncher
from open_r1.trainers.remote_model import RemoteModel
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_rich_available
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import pad, print_prompt_completions_sample, selective_log_softmax


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


if is_wandb_available():
    import wandb


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


# TODO: add the shared options with a mixin to reduce code duplication


class RepeatBatchRandomSampler(RandomSampler):
    def __init__(
        self,
        *args,
        mini_repeat_count: int = 1,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        return super().__len__() * self.mini_repeat_count

    def __iter__(self) -> Iterator[int]:
        batch_indices = []
        for idx in super().__iter__():
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                batch_indices = batch_indices * self.mini_repeat_count
                yield from batch_indices
                batch_indices = []


@dataclass
class RemoteGRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    checkpoint_dir: Optional[str] = field(
        default="/fsx/h4/tmp/", metadata={"help": "The directory to save temporary checkpoints to."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    remote_gen_model_url: Optional[str] = field(
        default=None,
    )
    remote_gen_model_port: str = field(
        default="30010",
    )
    remote_gen_model_n_gpus: str = field(
        default=8,
    )
    use_liger: bool = field(
        default=True,
        metadata={"help": "Whether to use Liger kernel for training."},
    )


class RemoteGRPOTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[RemoteGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        self.args = args
        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions

        # Training arguments
        self.num_generations = args.num_generations  # = G in the GRPO paper

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if args.use_liger:
                if not is_liger_kernel_available():
                    raise ImportError("Please install Liger-kernel for use_liger=True")
                model = AutoLigerKernelForCausalLM.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        self.batch_buffer = []
        self.grad_acc_scalar = exact_div(self.args.gradient_accumulation_steps, self.num_generations)

        super().__init__(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            data_collator=data_collator,
        )

        ip_address = self.args.remote_gen_model_url

        if self.args.remote_gen_model_url is None and self.accelerator.is_main_process:
            # we launch a job from here, get the ip on main process and broadcast to others
            # it would be better to move this to the start so the server warms up which the local model is being loaded
            model_revision = args.model_init_kwargs.get("revision", "main")
            self.sglang_job_launcher = SGLangSlurmJobLauncher(
                model_id,
                model_revision,
                num_gpus=self.args.remote_gen_model_n_gpus,
                sglang_port=self.args.remote_gen_model_port,
            )
            ip_address = self.sglang_job_launcher.launch()

        # get the ip from main process and broadcast to others
        gather_ip_address = broadcast_object_list([ip_address], 0)
        self.args.remote_gen_model_url = gather_ip_address[0]

        self.remote_model = RemoteModel(
            self.args.remote_gen_model_url, self.args.remote_gen_model_port, self.processing_class.eos_token_id
        )
        self.remote_model.wait_for_server()

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

    def _get_train_sampler(self) -> Sampler:
        """
        Return the train sampler.

        Returns:
            Sampler: The train sampler.
        """
        if self.args.dataloader_num_workers != 0:
            raise ValueError("dataloader_num_workers should not be greater than 0 for remote training")
        return RepeatBatchRandomSampler(
            data_source=self.train_dataset,
            batch_size=self._train_batch_size * self.grad_acc_scalar,
            mini_repeat_count=self.num_generations,
            replacement=False,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: RemoteGRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        prompts_to_log = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(prompts_text)

        prompt_ids = prompt_inputs["input_ids"]
        # sync weights here?
        self._sync_weights()
        with profiling_context(self, "remote_generate"):
            all_outputs = self.remote_model.generate(
                prompt_ids,
                max_new_tokens=self.args.max_completion_length,
                temperature=self.args.temperature,
                num_generations=self.num_generations,
            )
        completion_ids = [example["completion_ids"] for example in all_outputs]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        repeated_prompts = []
        for prompt in prompts_to_log:
            repeated_prompts.extend([prompt] * self.num_generations)

        repeated_prompt_texts = []
        for prompt in prompts_text:
            repeated_prompt_texts.extend([prompt] * self.num_generations)

        if is_conversational(inputs[0]):
            completions_to_log = []
            for prompt, completion in zip(repeated_prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions_to_log.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions_to_log = completions_text

        rewards = torch.zeros(len(repeated_prompts), len(self.reward_funcs))
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = defaultdict(list)
            for example in inputs:
                for key in keys:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=repeated_prompts, completions=completions_to_log, **reward_kwargs)
            rewards[:, i] = torch.tensor(output_reward_func, dtype=torch.float32) * self.reward_weights[i]

        # calculate the advantages, the prompt is all on the same device to no need to gather here
        grouped_rewards = rewards.sum(-1).view(len(prompts_to_log), self.num_generations)
        EPS = 1e-4
        grouped_advantages = (grouped_rewards - grouped_rewards.mean(-1, keepdim=True)) / (
            grouped_rewards.std(-1, keepdim=True) + EPS
        )
        advantages = grouped_advantages.flatten().tolist()

        examples = []
        for i, prompt in enumerate(repeated_prompt_texts):
            example = {
                "prompt": prompt,
                "prompt_ids": prompt_ids[i // self.num_generations],
                "completion": completions_text[i],
                "completion_ids": completion_ids[i],
                "advantages": advantages[i],
                "rewards": rewards[i],
            }
            examples.append(example)

        # Instead of logging metrics here, collect them
        mode = "eval" if getattr(self, "control", None) and self.control.should_evaluate else "train"
        device = self.accelerator.device

        # Collect completion length metrics
        completion_lengths = [len(example["completion_ids"]) for example in examples]
        gathered_completion_lengths = self.accelerator.gather_for_metrics(torch.Tensor(completion_lengths).to(device))
        self._metrics[mode]["mean_completion_lengths"].append(gathered_completion_lengths.mean().item())
        self._metrics[mode]["max_completion_lengths"].append(gathered_completion_lengths.max().item())
        self._metrics[mode]["min_completion_lengths"].append(gathered_completion_lengths.min().item())

        # Collect reward metrics
        rewards = torch.stack(
            [
                example["rewards"].to(device)
                if isinstance(example["rewards"], torch.Tensor)
                else torch.tensor(example["rewards"], device=device)
                for example in examples
            ]
        )
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        reward_per_func = gathered_rewards.mean(0)

        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(reward_per_func.sum().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object([example["prompt"] for example in examples])
            completions_to_log = gather_object([example["completion"] for example in examples])
            if self.accelerator.is_main_process:
                if is_rich_available():
                    # TODO: enable num_samples in TRL to avoid clogging logs
                    print_prompt_completions_sample(
                        prompts_to_log[:5],
                        completions_to_log[:5],
                        gathered_rewards.sum(1).tolist()[:5],
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(prompts_to_log),
                        "prompts": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": gathered_rewards.sum(1).tolist(),
                    }
                    df = pd.DataFrame(table)

                    if wandb.run is not None and self.accelerator.is_main_process:
                        wandb.log({"completions": wandb.Table(dataframe=df)})

        return examples

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        if len(self.batch_buffer) > 0:
            return self.batch_buffer.pop(0)
        inputs = self._generate_and_score_completions(inputs)
        gen_dataset = Dataset.from_list(inputs)
        exact_div(len(gen_dataset), self.args.per_device_train_batch_size, "len(gen_dataset) is not divisible by batch size")

        def get_ref_logprobs(example):
            # dict of lists to list of dicts
            examples = [dict(zip(example.keys(), values)) for values in zip(*example.values())]
            input_ids, attention_mask, completion_mask, completion_ids = self._get_padded_inputs_and_attn_mask(examples)
            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            with torch.no_grad():
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)

            lengths = [len(example["completion_ids"]) for example in examples]
            # STRIP OFF THE COMPLETION MASK
            ref_per_token_logps = ref_per_token_logps.to("cpu").tolist()
            ref_per_token_logps = [logps[:length] for logps, length in zip(ref_per_token_logps, lengths)]
            example["ref_per_token_logps"] = ref_per_token_logps
            return example
        
        disable_progress_bars()
        gen_dataset = gen_dataset.map(get_ref_logprobs, batched=True, batch_size=self.args.per_device_train_batch_size)
        enable_progress_bars()

        def mini_batch_collator(mini_batch):
            return mini_batch

        mini_batch_dataloader = DataLoader(
            gen_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,  # we technically don't need to shuffle due to grad acc, but we may move to clipped loss later
            drop_last=True,
            collate_fn=mini_batch_collator,
        )

        for mini_batch in mini_batch_dataloader:
            self.batch_buffer.append(mini_batch)

        return self.batch_buffer.pop(0)



    @profiling_decorator
    def _sync_weights(self):
        self.accelerator.wait_for_everyone()
        # if self.accelerator.is_main_process:
        start = time.time()
        # would be better if this was a ram disk + separate thread for writing
        # TODO, we need multi-process tmp dir here

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if is_deepspeed_zero3_enabled():
            state_dict = {}
            for name, param in unwrapped_model.named_parameters():
                with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                    state_dict[name] = param.cpu().detach().clone()
        else:
            state_dict = unwrapped_model.state_dict()

        if self.accelerator.is_main_process:
            with tempfile.TemporaryDirectory(dir=self.args.checkpoint_dir) as temp_dir_path:
                self._save(temp_dir_path, state_dict=state_dict)
                self.remote_model.load_weights_from_path(temp_dir_path)

            print(f"Weight sync took: {time.time() - start:.2f}s")
        self.accelerator.wait_for_everyone()

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens


    def _get_padded_inputs_and_attn_mask(self, inputs):
        device = self.accelerator.device
        prompt_ids = [torch.LongTensor(example["prompt_ids"]) for example in inputs]
        completion_ids = [torch.LongTensor(example["completion_ids"]) for example in inputs]
        # ref_per_token_logps = [torch.Tensor(example["ref_per_token_logps"]) for example in inputs]

        # for logps, completion_id in zip(ref_per_token_logps, completion_ids):
        #     assert len(logps) == len(completion_id), f"len(logps)={len(logps)} != len(completion_id)={len(completion_id)}"

        pad_token_id = self.processing_class.pad_token_id

        prompt_ids = pad(prompt_ids, padding_value=pad_token_id, padding_side="left")
        completion_ids = pad(completion_ids, padding_value=pad_token_id, padding_side="right")
        # padd_ref_per_token_logps = pad(ref_per_token_logps, padding_value=0.0, padding_side="right")

        if self.args.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.args.max_prompt_length :]

        # compute the masks
        prompt_mask = (prompt_ids != pad_token_id).long()

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(device)
        completion_mask = completion_mask.to(device)
        
        return input_ids, attention_mask, completion_mask, completion_ids

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = self.accelerator.device
        advs = torch.Tensor([example["advantages"] for example in inputs]).to(device)
        input_ids, attention_mask, completion_mask, completion_ids = self._get_padded_inputs_and_attn_mask(inputs)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        
        ref_per_token_logps = [torch.Tensor(example["ref_per_token_logps"]) for example in inputs]

        # Sanity check: ensure that the number of log probabilities matches the number of tokens in the completion, can be removed later
        for logps, inp in zip(ref_per_token_logps, inputs):
            assert len(logps) == len(inp["completion_ids"]), f"len(logps)={len(logps)} != len(completion_id)={len(inp['completion_ids'])}"
        pad_token_id = self.processing_class.pad_token_id
        
        pad_ref_per_token_logps = pad(ref_per_token_logps, padding_value=pad_token_id, padding_side="right").to(device)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        per_token_kl = torch.exp(pad_ref_per_token_logps - per_token_logps) - (pad_ref_per_token_logps - per_token_logps) - 1

        # TODO: convert to clipped loss so we can run multiple GRPO epochs
        per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advs.unsqueeze(1)
        per_token_loss = per_token_loss + self.args.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()
