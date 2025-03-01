from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from open_r1.configs import GRPOConfig
from open_r1.trainers.remote_model import RemoteModel
from transformers import AutoTokenizer

from collections import defaultdict
from trl.data_utils import is_conversational, maybe_apply_chat_template



class RemoteGRPODataloader(DataLoader):
    def __init__(self, *args, config: GRPOConfig, remote_model=None, processing_class=None, reward_funcs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.remote_model = remote_model
        self.processing_class = processing_class
        self.reward_funcs = reward_funcs
        self.reward_weights = [1.0] * len(reward_funcs) # TODO: make this configurable
        
    def __len__(self):
        return super().__len__() * self.config.num_generations    
    
    def __iter__(self):
        for batch in super().__iter__():
            batch = self._prepare_batch(batch)
            gen_dataset = Dataset.from_list(batch)
            mini_batch_dataloader = DataLoader(
                gen_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,  # we technically don#t need to shuffle due to grad acc, but we may move to clipped loss later
                drop_last=True,
                collate_fn=self.collate_fn,
            )
            for mini_batch in mini_batch_dataloader:
                yield mini_batch
                
                
    def _prepare_batch(self, batch):
        prompts = [x["prompt"] for x in batch]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch]
        prompt_inputs = self.processing_class(prompts_text)

        prompt_ids = prompt_inputs["input_ids"]
        # add cuda clear cache here and a sleep

        all_outputs = self.remote_model.generate(
            prompt_ids,
            max_new_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            num_generations=self.config.num_generations,
        )

        completion_ids = [example["completion_ids"] for example in all_outputs]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * self.config.num_generations)

        repeated_prompt_texts = []
        for prompt in prompts_text:
            repeated_prompt_texts.extend([prompt] * self.config.num_generations)

        if is_conversational(batch[0]):
            completions = []
            for prompt, completion in zip(repeated_prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards = torch.zeros(len(repeated_prompts), len(self.reward_funcs))
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in batch[0] if key not in ["prompt", "completion"]]
            reward_kwargs = defaultdict(list)
            for example in batch:
                for key in keys:
                    reward_kwargs[key].extend([example[key]] * self.config.num_generations)
            output_reward_func = reward_func(prompts=repeated_prompts, completions=completions, **reward_kwargs)
            rewards[:, i] = torch.tensor(output_reward_func, dtype=torch.float32) * self.reward_weights[i]

        grouped_rewards = rewards.sum(-1).view(len(prompts), self.config.num_generations)
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
                "prompt_ids": prompt_ids[i // self.config.num_generations],
                "completion": completions_text[i],
                "completion_ids": completion_ids[i],
                "advantages": advantages[i],
                "rewards": rewards[i],
            }
            examples.append(example)

        return examples


if __name__ == "__main__":

    dataset = load_dataset("open-r1/OpenR1-Math-cn_k12-86k", split="train").select(range(32))
    def make_conversation(example):
        prompt = []

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    def collate_fn(batch):
        return batch
    dataset = dataset.remove_columns("messages")
    def reward_func(prompts, completions, **kwargs):
        return [0.5]*len(prompts)

    reward_funcs = [reward_func, reward_func]

    MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
    processing_class = AutoTokenizer.from_pretrained(MODEL)
    remote_model = RemoteModel("0.0.0.0", 30010, processing_class.eos_token_id)
    config = GRPOConfig()
    data_loader = RemoteGRPODataloader(dataset, 
                                remote_model=remote_model,
                                processing_class=processing_class,
                                reward_funcs=reward_funcs,
                                batch_size=2,
                                num_workers=0,
                                collate_fn=collate_fn,
                                config=config
                                )
    print(len(data_loader))

    for i, batch in enumerate(data_loader):
        print(i, len(batch))
        print(batch)