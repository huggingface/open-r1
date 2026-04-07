from PIL import Image
from scripts.agents.function_parser import parse_function_call

from qwen_vl_utils import smart_resize

def resize_images_in_messages(batch_messages, script_args) -> list[Image.Image]:

    min_pixels = script_args.image_resize["min_pixels"]
    max_pixels = script_args.image_resize["max_pixels"]
    factor = script_args.image_resize["factor"]

    all_image_inputs = []
    for messages in batch_messages:

        old_image = messages[1]["content"][0]["image"]
        resized_height, resized_width = smart_resize(
            old_image.height,
            old_image.width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        new_image = old_image.resize((resized_width, resized_height))
        messages[1]["content"][0]["image"] = new_image

        function_calls = parse_function_call(messages[2]["content"])
        old_function_call_strings = [
            function_call.to_string() for function_call in function_calls
        ]
        for function_call, old_function_call_string in zip(function_calls, old_function_call_strings):
            if function_call.function_name in [
                "click",
                "long_press",
                "double_click",
                "move_mouse",
            ]:
                function_call.parameters["arg_0"] = (
                    int(function_call.parameters["arg_0"]
                    / old_image.width
                    * new_image.width)
                )
                function_call.parameters["arg_1"] = (
                    int(function_call.parameters["arg_1"]
                    / old_image.height
                    * new_image.height)
                )
            elif function_call.function_name in ["swipe", "drag"]:
                function_call.parameters["arg_0"] = (
                    int(function_call.parameters["arg_0"][0]
                    / old_image.width
                    * new_image.width),
                    int(function_call.parameters["arg_0"][1]
                    / old_image.height
                    * new_image.height)
                )
                function_call.parameters["arg_1"] = (
                    int(function_call.parameters["arg_1"][0]
                    / old_image.width
                    * new_image.width),
                    int(function_call.parameters["arg_1"][1]
                    / old_image.height
                    * new_image.height)
                )
            messages[2]["content"] = messages[2]["content"].replace(old_function_call_string, function_call.to_string())

        all_image_inputs.append([new_image])
    return all_image_inputs

def create_vlm_collate_fn(processor, script_args):
    """Optimized collate function for VLM training that masks system prompt tokens."""

    def collate_fn(examples: list[dict[str, str | Image.Image]]):
        batch_messages = []
        system_prompts = []
        user_prompts = []
        for example in examples:
            system = example["system"]
            user = example["user"]
            assistant = example["assistant"]
            image = example["image"]

            system_prompts.append(system)
            user_prompts.append(user)
            batch_messages.append(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": user},
                        ],
                    },
                    {"role": "assistant", "content": assistant},
                ]
            )

        all_image_inputs = []
        if script_args.image_resize is not None:
            all_image_inputs = resize_images_in_messages(batch_messages, script_args)


        texts = [
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in batch_messages
        ]

        batch = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt",
            max_length=4096,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        if hasattr(processor, "image_token"):
            image_token_id = processor.tokenizer.convert_tokens_to_ids(
                processor.image_token
            )
            if image_token_id is not None:
                labels[labels == image_token_id] = -100
        else:
            raise ValueError("Processor does not have image_token")

        system_encodings = processor.tokenizer(
            system_prompts, add_special_tokens=False, padding=False
        )["input_ids"]

        user_encodings = processor.tokenizer(
            user_prompts, add_special_tokens=False, padding=False
        )["input_ids"]

        for encodings in [system_encodings, user_encodings]:
            for i, system_ids in enumerate(encodings):
                if input_ids[i, : len(system_ids)].tolist() == system_ids:
                    labels[i, : len(system_ids)] = -100
                else:
                    seq = input_ids[i].tolist()
                    for j in range(len(seq) - len(system_ids) + 1):
                        if seq[j : j + len(system_ids)] == system_ids:
                            labels[i, j : j + len(system_ids)] = -100
                            break  # early exit

        batch["labels"] = labels
        return batch

    return collate_fn
