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

# great reference: https://github.com/vllm-project/vllm/issues/11400

import contextlib
import functools
import gc
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import reduction
from typing import Callable, Optional, Union
from unittest.mock import patch
from transformers.utils import is_liger_kernel_available
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

import trl
from accelerate import Accelerator
from accelerate.utils import gather_object
from open_r1.trainers.job_launcher import SGLangSlurmJobLauncher
from open_r1.trainers.remote_model import RemoteModel
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.utils import pad, selective_log_softmax
from vllm import LLM, SamplingParams


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


if is_wandb_available():
    import wandb


@contextlib.contextmanager
def profiling_context(instance, name):
    """
    A context manager function for profiling a block of code.
    Can also be used as a decorator.
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    if "wandb" in instance.args.report_to and wandb.run is not None and instance.accelerator.is_main_process:
        wandb.log({f"profiling/Time taken: {instance.__class__.__name__}.{name}": duration})


def profiling_decorator(func):
    """
    Decorator to profile a function and log execution time using profiling_context.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


from accelerate import Accelerator


if is_wandb_available():
    import wandb


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class FastGRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
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
    remote_gen_model_url: str = field(
        default="26.0.165.24",
    )
    remote_gen_model_port: str = field(
        default="30010",
    )
    remote_gen_model_n_gpus: str = field(
        default=8,
    )


class FastGRPOTrainer(Trainer):
    _tag_names = ["trl", "fast_grpo"]

    def __init__(
        self,
        model: str,  # only accept str for now
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: FastGRPOConfig,
        train_dataset: Dataset,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
    ) -> None:
        self.args = args
        self.reward_funcs = reward_funcs
        # Reward weights (move this logic to post_init of config?)
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = args.reward_weights
        else:
            self.reward_weights = ([1.0] * len(reward_funcs),)

        # start the remote model so it has time to warmup while we load the local model(s)
        if self.args.remote_gen_model_url is None:
            self.sglang_job_launcher = SGLangSlurmJobLauncher(
                model, num_gpus=self.args.remote_gen_model_n_gpus, sglang_port=self.args.remote_gen_model_port
            )
            self.sglang_job_launcher.submit_job()

        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
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
            model_str = model
            model = AutoModelForCausalLM.from_pretrained(model_str, **model_init_kwargs)
            # offload to cpu
            ref_model = AutoModelForCausalLM.from_pretrained(model_str, **model_init_kwargs) #.to("cpu")

        self.model = model
        self.ref_model = ref_model
        if self.args.use_liger_kernel:
            if is_liger_kernel_available():
                from liger_kernel.transformers import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)
                _apply_liger_kernel_to_instance(model=self.ref_model)
            else:
                raise ImportError(
                    "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
                    "Please install it with `pip install liger-kernel`"
                )
        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        self.processing_class = processing_class

        self.train_dataset = train_dataset

        if data_collator is not None:
            raise ValueError("")

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        self.data_collator = data_collator

        local_dataloader_batch_size = exact_div(
            args.per_device_train_batch_size * args.gradient_accumulation_steps,
            args.num_generations,
            "per_device_train_batch_size * gradient_accumulation_steps must >= num_generations to remain on policy",
        )
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

        self.train_dataset_len = len(self.train_dataset)
        num_total_samples = int(self.args.num_train_epochs * self.train_dataset_len)
        self.total_steps_per_device = num_total_samples // (
            local_dataloader_batch_size * self.accelerator.num_processes
        )
        self.create_optimizer_and_scheduler(num_training_steps=self.total_steps_per_device)
        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        torch.manual_seed(args.seed)
                # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            self.model = self._enable_gradient_checkpointing(self.model, self.args)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        # connect to a remote sglang model
        if self.args.remote_gen_model_url is None:
            self.sglang_job_launcher.wait_for_server()
            self.args.remote_gen_model_url = self.sglang_job_launcher.get_remote_ip()
        self.remote_model = RemoteModel(
            self.args.remote_gen_model_url, self.args.remote_gen_model_port, self.processing_class.eos_token_id
        )
    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: FastGRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )
        if use_reentrant:
            model.enable_input_require_grads()

        return model
    def print_gpu_memory_usage(self):
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            print(f"GPU memory allocated: {gpu_memory_allocated / (1024**3):.2f} GB")
            print(f"GPU memory reserved: {gpu_memory_reserved / (1024**3):.2f} GB")
        else:
            print("CUDA is not available.")

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

    @torch.no_grad()
    @profiling_decorator
    def _prepare_batch(self, batch):
        """
        This will:
        - generate k samples for each problem
        - using internal reward model(s) to get rewards
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in batch]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch]
        prompt_inputs = self.processing_class(prompts_text)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # add cuda clear cache here and a sleep

        all_outputs = self.remote_model.generate(
            prompt_ids,
            max_new_tokens=self.args.max_completion_length,
            temperature=self.args.temperature,
            num_generations=self.args.num_generations,
        )

        # all_outputs = self.gen_vllm.generate(prompts_text, sampling_params=self.sampling_params, use_tqdm=True)

        completion_ids = [example["completion_ids"] for example in all_outputs]

        # completion_ids = []
        # for outputs in all_outputs:
        #     for output in outputs.outputs:
        #         completion_ids.append(output.token_ids)

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * self.args.num_generations)

        repeated_prompt_texts = []
        for prompt in prompts_text:
            repeated_prompt_texts.extend([prompt] * self.args.num_generations)

        if is_conversational(batch[0]):
            completions = []
            for prompt, completion in zip(repeated_prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards = torch.zeros(len(repeated_prompts), len(self.reward_funcs))
        for (
            i,
            reward_func,
        ) in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in batch[0] if key not in ["prompt", "completion"]]
            reward_kwargs = defaultdict(list)
            for example in batch:
                for key in keys:
                    reward_kwargs[key].extend([example[key]] * self.args.num_generations)
            output_reward_func = reward_func(prompts=repeated_prompts, completions=completions, **reward_kwargs)
            rewards[:, i] = torch.tensor(output_reward_func, dtype=torch.float32) * self.reward_weights[i]

        # calculate the advantages, the prompt is all on the same device to no need to gather here
        grouped_rewards = rewards.sum(-1).view(len(prompts), self.args.num_generations)
        EPS = 1e-4
        grouped_advantages = (grouped_rewards - grouped_rewards.mean(-1, keepdim=True)) / (
            grouped_rewards.std(-1, keepdim=True) + EPS
        )
        advantages = grouped_advantages.flatten().tolist()

        # build batch as list of dicts
        examples = []
        for i, prompt in enumerate(repeated_prompt_texts):
            example = {
                "prompt": prompt,
                "prompt_ids": prompt_ids[i // self.args.num_generations],
                "completion": completions_text[i],
                "completion_ids": completion_ids[i],
                "advantages": advantages[i],
                "rewards": rewards[i],
            }
            examples.append(example)

        return examples

    @profiling_decorator
    def _sync_weights(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            start = time.time()
            with tempfile.TemporaryDirectory(dir="/fsx/edward/work/open-r1/data/") as temp_dir_path:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(temp_dir_path)
                self.remote_model.load_weights_from_path(temp_dir_path)
            print("weight sync took: ", time.time() - start)
        self.accelerator.wait_for_everyone()

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
    ):
        start_step = 1  # todo, set this when we resume + load model, opt state etc

        if self.args.logging_steps is not None:
            if self.args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * self.args.logging_steps)
            else:
                self.state.logging_steps = self.args.logging_steps

        if self.args.save_steps is not None:
            if self.args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * self.args.save_steps)
            else:
                self.state.save_steps = self.args.save_steps

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        self.state.global_step = 0
        self.state.max_steps = self.total_steps_per_device
        self.state.num_train_epochs = self.args.num_train_epochs

        def repeat_generator():
            while True:
                yield from self.dataloader

        iter_dataloader = iter(repeat_generator())

        self.model.train()

        @torch.no_grad()
        def mini_batch_collator(examples):
            device = self.accelerator.device

            prompt_ids = [torch.LongTensor(example["prompt_ids"]) for example in examples]
            completion_ids = [torch.LongTensor(example["completion_ids"]) for example in examples]
            ref_per_token_logps = [torch.Tensor(example["ref_per_token_logps"]) for example in examples]
            
            for logps, completion_id in zip(ref_per_token_logps, completion_ids):
                assert len(logps) == len(completion_id), f"len(logps)={len(logps)} != len(completion_id)={len(completion_id)}"
            
            pad_token_id = self.processing_class.pad_token_id
            
            padded_prompt_ids = pad(prompt_ids, padding_value=pad_token_id, padding_side="left")
            padded_completion_ids = pad(completion_ids, padding_value=pad_token_id, padding_side="right")
            padd_ref_per_token_logps = pad(ref_per_token_logps, padding_value=0.0, padding_side="right")
            
            if self.args.max_prompt_length is not None:
                padded_prompt_ids = padded_prompt_ids[:, -self.args.max_prompt_length :]

            # compute the masks
            prompt_mask = (padded_prompt_ids != pad_token_id).long()

            # Mask everything after the first EOS token
            is_eos = padded_completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            advantages = torch.Tensor([example["advantages"] for example in examples])
            

            return {
                "prompt_ids": padded_prompt_ids.to(device),
                "prompt_mask": prompt_mask.to(device),
                "completion_ids": padded_completion_ids.to(device),
                "completion_mask": completion_mask.to(device),
                "advantages": advantages.to(device),
                "ref_per_token_logps": padd_ref_per_token_logps.to(device),
            }

        device = self.accelerator.device
        for step in range(start_step, self.total_steps_per_device + 1):
            batch = next(iter_dataloader)
            batch = self._prepare_batch(batch)

            # TODO: log completions, rewards, etc
            gen_dataset = Dataset.from_list(batch)
            
            @torch.no_grad()
            def compute_ref_logps(examples):
                device = self.accelerator.device
                prompt_ids = [torch.LongTensor(prompt_id) for prompt_id in examples["prompt_ids"]]
                completion_ids = [torch.LongTensor(completion_id) for completion_id in examples["completion_ids"]]
                completion_lengths = [len(c) for c in completion_ids]
                pad_token_id = self.processing_class.pad_token_id
                padded_prompt_ids = pad(prompt_ids, padding_value=pad_token_id, padding_side="left")
                padded_completion_ids = pad(completion_ids, padding_value=pad_token_id, padding_side="right")
                
                
                input_ids = torch.cat([padded_prompt_ids, padded_completion_ids], dim=1)
                attention_mask = torch.cat([padded_prompt_ids != pad_token_id, padded_completion_ids != pad_token_id], dim=1)
                logits_to_keep = torch.tensor(completion_lengths).to(device)
                logits_to_keep = padded_completion_ids.size(1)
                with torch.inference_mode():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids.to(device), attention_mask.to(device), logits_to_keep
                    )
                ref_per_token_logps = ref_per_token_logps.to("cpu")    
                examples["ref_per_token_logps"] = [logprobs[:length] for logprobs, length in zip(ref_per_token_logps, completion_lengths)]
                
                return examples
            
            
            self.ref_model = self.ref_model.to(device)
            # precompute the ref logprobs and offload the model to cpu
            gen_dataset = gen_dataset.map(compute_ref_logps, batched=True, batch_size=self.args.per_device_train_batch_size)
            self.ref_model = self.ref_model.to("cpu")
            
            # we could add some optimizations here like sorting the dataset by length to improve throughput, but we will keep it simple for now
            mini_batch_dataloader = DataLoader(
                gen_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,  # we technically don#t need to shuffle due to grad acc, but we may move to clipped loss later
                drop_last=True,
                collate_fn=mini_batch_collator,
            )
            # optimization
            # stats for logging
            losses = []
            kls = []

            with profiling_context(self, "train_step"):
                for mini_batch in mini_batch_dataloader:
                    loss_metric, kl_metric = self._optimization_step(mini_batch)
                    losses.append(loss_metric)
                    kls.append(kl_metric)

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.state.epoch = step / self.total_steps_per_device  #  TODO, this is not correct

            # logging stats
            metrics = {}
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["loss"] = self.accelerator.gather_for_metrics(torch.Tensor(losses).to(device)).mean().item()
            metrics["kl"] = self.accelerator.gather_for_metrics(torch.Tensor(kls).to(device)).mean().item()

            # completions stats
            completion_lengths = [len(c) for c in gen_dataset["completion_ids"]]
            gathered_completion_lengths = self.accelerator.gather_for_metrics(
                torch.Tensor(completion_lengths).to(device)
            )
            metrics["mean_completion_lengths"] = gathered_completion_lengths.mean().item()
            metrics["max_completion_lengths"] = gathered_completion_lengths.max().item()
            metrics["min_completion_lengths"] = gathered_completion_lengths.min().item()

            # reward stats
            rewards = gen_dataset["rewards"]
            gathered_rewards = self.accelerator.gather_for_metrics(torch.Tensor(rewards).to(device))
            reward_per_func = gathered_rewards.mean(0)
            for i, reward_func in enumerate(self.reward_funcs):
                reward_func_name = reward_func.__name__
                metrics[f"rewards/{reward_func_name}"] = reward_per_func[i].item()

            metrics["reward"] = reward_per_func.sum().item()

            self.log(metrics)
            if self.args.log_completions and "wandb" in self.args.report_to:
                import pandas as pd

                prompts = gather_object(gen_dataset["prompt"])
                completions = gather_object(gen_dataset["completion"])
                # For logging
                table = {
                    "step": [str(self.state.global_step)] * len(prompts),
                    "prompts": prompts,
                    "completion": completions,
                    "reward": gathered_rewards.sum(1).tolist(),
                }
                df = pd.DataFrame(table)

                if wandb.run is not None and self.accelerator.is_main_process:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            # sync weights to remote server
            self._sync_weights()

            self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(self.model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(self.model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def _optimization_step(self, mini_batch)-> tuple[float, float]:
        prompt_ids, prompt_mask = mini_batch["prompt_ids"], mini_batch["prompt_mask"]
        completion_ids, completion_mask = mini_batch["completion_ids"], mini_batch["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        ref_per_token_logps = mini_batch["ref_per_token_logps"]

        with self.accelerator.accumulate(self.model):
            per_token_logps = self._get_per_token_logps(
                self.model, input_ids, attention_mask, logits_to_keep
            )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

            advantages = mini_batch["advantages"]
            # TODO: convert to clipped loss so we can multiple GRPO epochs
            per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(
                1
            )
            per_token_loss = per_token_loss + self.args.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        del per_token_logps, per_token_kl, per_token_loss, loss
        
        # force garbage collection and empty cache
        gc.collect()
        torch.cuda.empty_cache()
        
        return loss.detach().item(), per_token_kl.mean().item()