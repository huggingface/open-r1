import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.smolvlm.image_processing_smolvlm import SmolVLMImageProcessor


class TransformersModel:
    def __init__(self, model_id: str, to_device: str = "cuda"):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        self.processor.image_processor.size = {"longest_edge": 3 * 384}
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(to_device)

    def generate(self, messages: list[dict], **kwargs):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs, **kwargs)
        return self.processor.batch_decode(
                generated_ids[:, len(inputs["input_ids"][0]) :], skip_special_tokens=True
            )[0]


if __name__ == "__main__":
    from PIL import Image

    model = TransformersModel(
        model_id="/fsx/amir_mahla/smolagents-SmolVLM2-2.2B-Instruct-Agentic-GUI-phase-1-max-size-1152/checkpoint-800",
        to_device="cuda:0",
    )

    image = Image.open("/admin/home/amir_mahla/screensuite/examples/sample_image.png")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "Given the screenshot, and the instruction, output a click that completes the instruction or targets the given element (always target the center of the element).\n\nOutput the click position as follows:\n\n<think>(thought process)</think><code>click(x, y)</code>\nWith x the number of pixels from the left edge and y the number of pixels from the top edge.\n\nNow write the click needed to complete the instruction:\nInstruction: view more information about bomber\n",
                },
            ],
        },
    ]


    print(model.generate(messages, max_new_tokens=128))