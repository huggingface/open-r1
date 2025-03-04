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


import contextlib
import functools
import gc
import math
import os
import tempfile
import time
import accelerate
from torch.utils.data import RandomSampler
from typing import Iterator
from transformers import AutoModelForCausalLM
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import reduction
from typing import Callable, Optional, Union
from unittest.mock import patch
import deepspeed
from confection import ARGS_FIELD_ALIAS
from transformers.utils import is_liger_kernel_available
import torch
from copy import deepcopy
from datasets import Dataset
from accelerate.utils import broadcast_object_list
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
from torch.utils.data import Sampler
from typing import Any
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from open_r1.trainers.job_launcher import SGLangSlurmJobLauncher

import trl
from accelerate import Accelerator
from accelerate.utils import gather_object
from open_r1.trainers.job_launcher import SGLangSlurmJobLauncher
from open_r1.trainers.remote_model import RemoteModel
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.utils import pad, selective_log_softmax
from accelerate import Accelerator
from trl import SFTTrainer
from open_r1.trainers.special_dataloader import RemoteGRPODataloader
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState
if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
from typing import Optional, Iterator, Sized

if is_wandb_available():
    import wandb
def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q

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
# TODO: add the shared options with a mixin to reduce code duplication


class RepeatBatchRandomSampler(RandomSampler):
    def __init__(self, 
                 *args,
                num_generations: int = 1, 
                batch_size: int = 3,  
                 **kwargs,
                 )-> None:
        self.num_generations = num_generations
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
    
    def __len__(self) -> int:
        return super().__len__() * self.num_generations

    def __iter__(self) -> Iterator[int]:
        batch_indices = []
        for idx in super().__iter__():
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                batch_indices = batch_indices * self.num_generations
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
    def __init__(self, model, 
                 reward_funcs: Union[RewardFunc, list[RewardFunc]], 
                 args: RemoteGRPOConfig, 
                 train_dataset, 
                 processing_class, 
                 callbacks):
        self.args = args
        


        
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
        self.model_name_or_path = model
        if isinstance(model, str):
            model_path = model
            model = self._create_model_from_path(model_path, args)
            self.ref_model = self._create_model_from_path(model_path, args)
            
            
        def data_collator(features):  # No data collation is needed in GRPO
            return features
        
        self.batch_buffer = []
        self.grad_acc_scalar = exact_div(self.args.gradient_accumulation_steps, self.args.num_generations)
            
        super().__init__(model, args, train_dataset=train_dataset, processing_class=processing_class, callbacks=callbacks, data_collator=data_collator)
        
        ip_address = self.args.remote_gen_model_url
        
        if self.args.remote_gen_model_url is None and self.accelerator.is_main_process:
            # we launch a job from here, get the ip on main process and broadcast to others
            # it would be better to move this to the start so the server warms up which the local model is being loaded
            model_revision = args.model_init_kwargs.get("revision", "main")
            self.sglang_job_launcher = SGLangSlurmJobLauncher(
                model_path, model_revision, num_gpus=self.args.remote_gen_model_n_gpus, sglang_port=self.args.remote_gen_model_port
            )
            ip_address = self.sglang_job_launcher.launch()

        # get the ip from main process and broadcast to others
        gather_ip_address = broadcast_object_list([ip_address], 0)
        self.args.remote_gen_model_url = gather_ip_address[0]
            
        self.remote_model = RemoteModel(
            self.args.remote_gen_model_url, self.args.remote_gen_model_port, self.processing_class.eos_token_id
        )
        self.remote_model.wait_for_server()
        
    def _create_model_from_path(self, model_path: str, args) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        if args.gradient_checkpointing:
            model_init_kwargs["use_cache"] = False

        # Create model
        if args.use_liger:
            if not is_liger_kernel_available():
                raise ImportError("Please install Liger-kernel for use_liger=True")
            model = AutoLigerKernelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
            
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
            
        return model
    
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
            batch_size=self._train_batch_size *self.grad_acc_scalar,
            num_generations=self.args.num_generations,
            replacement=False,
        )
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        if len(self.batch_buffer) > 0:
            return self.batch_buffer.pop(0)
        
        prompts = [x["prompt"] for x in inputs]
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
                num_generations=self.args.num_generations,
            )
        completion_ids = [example["completion_ids"] for example in all_outputs]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * self.args.num_generations)

        repeated_prompt_texts = []
        for prompt in prompts_text:
            repeated_prompt_texts.extend([prompt] * self.args.num_generations)

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(repeated_prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards = torch.zeros(len(repeated_prompts), len(self.reward_funcs))
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = defaultdict(list)
            for example in inputs:
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

        gen_dataset = Dataset.from_list(examples)
        
        # logging to wandb
        device = self.accelerator.device
        metrics = {}
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
        
        
        def mini_batch_collator(mini_batch):
            return mini_batch
        
        mini_batch_dataloader = DataLoader(
            gen_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,  # we technically don#t need to shuffle due to grad acc, but we may move to clipped loss later
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
        state_dict = {}
        for name, param in unwrapped_model.named_parameters():
            with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                state_dict[name] = param.cpu().detach().clone()
        
        
        # state_dict = self.accelerator.get_state_dict(self.deepspeed)
        
        if self.accelerator.is_main_process:
            with tempfile.TemporaryDirectory(dir="/fsx/h4/tmp/") as temp_dir_path:
                self._save(temp_dir_path, state_dict=state_dict)
                self.remote_model.load_weights_from_path(temp_dir_path)
            
            print("weight sync took: ", time.time() - start)
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
    
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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

        advs = torch.Tensor([example["advantages"] for example in inputs])    
        
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(device)
        completion_mask =completion_mask.to(device)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        # TODO: this could be precomputed at the generation stage with larger batches so the ref model can be unloaded
        with torch.inference_mode():
            ref_per_token_logps = self._get_per_token_logps(
                self.model, input_ids, attention_mask, logits_to_keep
            )

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        # TODO: add the reference model
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )
        advs = torch.Tensor([example["advantages"] for example in inputs]).to(device)
        # TODO: convert to clipped loss so we can multiple GRPO epochs
        per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advs.unsqueeze(1)
        per_token_loss = per_token_loss + self.args.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        
        return loss