from PIL import Image
from scripts.agents.function_parser import parse_function_call
import numpy as np
from transformers.models.smolvlm.image_processing_smolvlm import (
    get_resize_output_image_size,
)
from transformers.image_utils import ChannelDimension


def transform_messages(
    batch_messages,
    image_resize: dict[str, int | bool],
) -> list[list[Image.Image]]:

    resolution_max_side = image_resize["resolution_max_side"] if "resolution_max_side" in image_resize else None
    to_pixel_coordinates = image_resize["to_pixel_coordinates"] if "to_pixel_coordinates" in image_resize else False

    if not to_pixel_coordinates and resolution_max_side is None:
        return batch_messages

    all_image_inputs: list[list[Image.Image]] = []
    for messages in batch_messages:
        new_image = None
        for i in range(len(messages)):
            if "image" in messages[i]["content"][0]:
                old_image = messages[i]["content"][0]["image"]

                if resolution_max_side is not None:
                    resized_height, resized_width = get_resize_output_image_size(
                        np.array(old_image),
                        resolution_max_side=resolution_max_side,
                        input_data_format=ChannelDimension.LAST,
                    )
                    new_image = old_image.resize((resized_width, resized_height))
                else:
                    resized_height, resized_width = old_image.height, old_image.width
                    new_image = old_image

                messages[i]["content"][0]["image"] = new_image
                all_image_inputs.append([new_image])

            if messages[i]["role"] == "assistant" and to_pixel_coordinates:
                assert new_image is not None, "new_image is None"

                function_calls = parse_function_call(messages[i]["content"][0]["text"])
                old_function_call_strings = [
                    function_call.to_string() for function_call in function_calls
                ]
                for function_call, old_function_call_string in zip(
                    function_calls, old_function_call_strings
                ):
                    if function_call.function_name in [
                        "click",
                        "long_press",
                        "double_click",
                        "move_mouse",
                    ]:
                        function_call.parameters["x"] = int(
                            function_call.parameters["x"] * new_image.width
                        )
                        function_call.parameters["y"] = int(
                            function_call.parameters["y"] * new_image.height
                        )
                    elif function_call.function_name in ["swipe", "drag"]:
                        function_call.parameters["x"] = (
                            int(function_call.parameters["x"][0] * new_image.width),
                            int(
                                function_call.parameters["x"][1] * new_image.height
                            ),
                        )
                        function_call.parameters["y"] = (
                            int(function_call.parameters["y"][0] * new_image.width),
                            int(
                                function_call.parameters["y"][1] * new_image.height
                            ),
                        )
                    messages[i]["content"][0]["text"] = messages[i]["content"][0][
                        "text"
                    ].replace(old_function_call_string, function_call.to_string())

    return all_image_inputs


def create_vlm_collate_fn(processor, script_args):
    """Optimized collate function for VLM training that masks system prompt tokens."""

    def collate_fn(examples: list[dict[str, list | str | Image.Image]]):
        batch_messages: list[list[dict[str, list | str | Image.Image]]] = []
        assistant_messages: list[list[str]] = []
        all_image_inputs: list[list[Image.Image]] = []
        for example in examples:
            images: list[Image.Image] = example["images"]
            is_first_user = True
            sample: list[dict[str, list | str | Image.Image]] = []
            assistant: list[str] = []
            for text in example["texts"]:
                if "system" in text.keys():
                    sample.append(
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": text["system"]}],
                        }
                    )

                if is_first_user:
                    sample.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": images[0]},
                                {"type": "text", "text": text["user"]},
                            ],
                        }
                    )
                    is_first_user = False
                else:
                    sample.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text["user"]},
                            ],
                        }
                    )

                sample.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "\n" + text["assistant"]}],
                    }
                )
                assistant.append(text["assistant"] + "<end_of_utterance>")

            batch_messages.append(sample)
            assistant_messages.append(assistant)
            all_image_inputs.append(images)

        if script_args.image_resize is not None and "to_pixel_coordinates" in script_args.image_resize and script_args.image_resize["to_pixel_coordinates"]:
            all_image_inputs = transform_messages(
                batch_messages,
                image_resize=script_args.image_resize,
            )


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
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        assistant_encodings = [
            processor.tokenizer(
                assistant_message, add_special_tokens=False, padding=False
            )["input_ids"]
            for assistant_message in assistant_messages
        ]

        # Mask out all except the assistant messages
        for i, assistant_ids_list in enumerate(assistant_encodings):
            seq = input_ids[i].tolist()
            assistant_positions: list[int] = []
            for ids in assistant_ids_list:
                start_pos = 0
                while start_pos < len(seq) - len(ids) + 1:
                    found = False
                    for j in range(start_pos, len(seq) - len(ids) + 1):
                        if seq[j : j + len(ids)] == ids:
                            assistant_positions.extend(range(j, j + len(ids)))
                            start_pos = j + len(ids)
                            found = True
                            break
                    if not found:
                        break

            for pos in range(len(seq)):
                if pos not in assistant_positions:
                    labels[i, pos] = -100


        batch["labels"] = labels
        return batch

    return collate_fn


if __name__ == "__main__":
    from transformers import AutoProcessor
    from datasets import load_dataset

    class ScriptArguments:
        image_resize = None

    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    )
    processor.image_processor.size = {"longest_edge": 384}
    collate_fn = create_vlm_collate_fn(processor, script_args=ScriptArguments)
    max_length = 0
    for dataset_name in ['omniact', 'ricoig16k', 'webui350k', 'widget_captioning', 'seeclick', 'ui_refexp', 'ricosca', 'guienv']:
        data = load_dataset("smolagents/aguvis-stage-1", dataset_name, split="train")
        print("processing", dataset_name)
        for example in data:
            batch = collate_fn([example])
            max_length = max(max_length, batch["input_ids"].shape[1])
    print("max_length", max_length)
    open("max_length_384_phase_1.txt", "w").write(str(max_length))
