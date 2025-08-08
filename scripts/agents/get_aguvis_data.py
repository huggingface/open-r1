#!/usr/bin/env python3
"""
Script to download, process, and upload the aguvis-stage2 dataset.
Downloads from huggingface.co/datasets/xlangai/aguvis-stage2 and uploads to smolagents/aguvis-stage-2
"""

import re
import gc
import sys
import json
import os
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Generator, Callable, Literal
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, snapshot_download
from collections import defaultdict
from PIL import Image
import tarfile
from itertools import islice
import multiprocessing as mp
from multiprocessing import Pool, Manager
from prompts import OS_SYSTEM_PROMPT, MOBILE_SYSTEM_PROMPT
from models import ConversationDataList, ConversationData, ChatMessage, DataRow
from function_parser import parse_function_call
from action_conversion import action_conversion
from pydantic import BaseModel
from config import config_dict_stage_1, config_dict_stage_2, MOBILE_FILE


api = HfApi()


def authenticate_huggingface():
    """Authenticate with HuggingFace Hub using token."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Authenticating with HuggingFace Hub using token...")
        login(token=hf_token)
    else:
        raise ValueError("HF_TOKEN environment variable not set.")


def discover_dataset_config(dataset_path: str, config_dict: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Discover dataset configuration by scanning the data directory."""
    dataset_dir = Path(dataset_path)
    train_dir = dataset_dir

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    configs = []
    processed_splits = set()

    # Find all JSON files in the train directory
    for config in config_dict:
        subset_name = (
            config["json_path"]
            .replace(".json", "")
            .replace("-l1", "")
            .replace("-l2", "")
            .replace("-l3", "")
        )

        # Skip if we already processed this split
        if subset_name in processed_splits:
            continue

        config["subset_name"] = subset_name
        configs.append(config)
        processed_splits.add(subset_name)
        print(
            f"Discovered config: {config['subset_name']} -> {config['images_folder']}"
        )

    return configs


def download_dataset(
    repo_id: str = "xlangai/aguvis-stage2", local_dir: str = "./aguvis_raw"
) -> str:
    """Download the dataset using snapshot_download."""
    print(f"Downloading dataset from {repo_id}...")
    local_path = snapshot_download(
        repo_id=repo_id, local_dir=local_dir, repo_type="dataset"
    )
    print(f"Dataset downloaded to: {local_path}")
    return local_path


def extract_zip_files(dataset_path: str):
    """Extract all zip files found in the dataset directory, but only if not already extracted."""
    print("Extracting zip files...")
    dataset_dir = Path(dataset_path)

    for zip_file in dataset_dir.rglob("*.zip"):
        extract_dir = zip_file.parent / zip_file.stem
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(
                f"Skipping extraction for {zip_file} (already extracted at {extract_dir})"
            )
            continue

        print(f"Extracting: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")


def extract_tar_parts_grouped(dataset_path: str):
    """
    Finds all .tar.gz.part_* groups, merges them, and extracts them into directories
    named after their common prefix.
    """
    dataset_dir = Path(dataset_path)
    part_files = list(dataset_dir.glob("*.tar.gz.part_*"))

    if not part_files:
        print("No split .tar.gz.part_* files found.")
        return

    # Group part files by prefix
    groups = defaultdict(list)
    for part in part_files:
        prefix = part.name.split(".tar.gz.part_")[0]
        groups[prefix].append(part)

    for prefix, parts in groups.items():
        parts = sorted(parts)  # Ensure correct order
        merged_tar_path = dataset_dir / f"{prefix}.tar.gz"
        extract_dir = dataset_dir / prefix

        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(
                f"Skipping extraction for '{prefix}' (already extracted at {extract_dir})"
            )
            continue

        # Merge parts
        CHUNK_SIZE = 1024 * 1024
        print(f"Merging parts for '{prefix}'...")
        with open(merged_tar_path, "wb") as outfile:
            for part in parts:
                print(f"  Adding: {part.name}")
                with open(part, "rb") as infile:
                    while chunk := infile.read(CHUNK_SIZE):
                        outfile.write(chunk)

        print(f"Merged to: {merged_tar_path}")

        # Extract
        print(f"Extracting to: {extract_dir}")
        with tarfile.open(merged_tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"Done extracting '{prefix}'\n")


def check_subset_exists(repo_id: str, subset_name: str) -> bool:
    """Check if a subset already exists in the remote dataset."""
    try:
        # Try to get dataset info with specific subset
        from datasets import get_dataset_config_names

        config_names = get_dataset_config_names(repo_id)
        return subset_name in config_names
    except Exception as e:
        print(f"Could not check if subset exists: {e}")
        return False


def load_image_from_folder(images_folder: Path, img_path: str) -> Image.Image:
    """Load images from the specified folder."""
    full_path = images_folder / img_path
    img = Image.open(full_path)
    new_img = img.copy()
    img.close()
    return new_img


def convert_to_code_agent_format(messages: list[ChatMessage], json_path: str, reasoning: bool):
    for i, message in enumerate(messages):
        content = message.content

        if message.role == "system":
            if json_path in MOBILE_FILE:
                content = MOBILE_SYSTEM_PROMPT
            else:
                content = OS_SYSTEM_PROMPT

        if message.role == "user":
            content = content.replace("<image>\n", "").replace("<image>", "")

        elif message.role == "assistant":
            content = (
                content.replace("Action: ", "")
                .replace("Observation: ", "")
                .replace("Thought: ", "")
            )
            if reasoning and i == len(messages) - 1:
                content = (
                    "<code>\n" + content.strip() + "\n</code>"
                )
            elif reasoning:
                # TODO: Check if there is always only 2 assistants
                content = (
                    "<think>\n"
                    + content.strip()
                    + "\n</think>\n"
                )
            else:
                content = content.strip()

        messages[i].content = content

        # Fuse subsequent messages have the same role, merge it
        if i > 0 and messages[i].role == messages[i - 1].role:
            # Need to fuse both messages
            if reasoning:
                messages[i - 1].content += messages[i].content
            else:
                messages[i - 1].content += "\n" + messages[i].content
            messages.pop(i)

    return messages


def convert_to_chat_format(
    data: ConversationData, json_path: str, reasoning: bool
) -> list[ChatMessage]:
    """Convert data item to chat template format."""
    # This is a placeholder - you'll need to adapt this based on the actual data structure
    # The exact conversion depends on how the original data is structured
    chat_messages = data.to_chat_messages()
    # mobile = json_path in open("mobile_files.txt", "r").read()
    # os = json_path in open("os_files.txt", "r").read()
    # if not mobile and not os:
    #     for message in chat_messages:
    #         if mobile and os:
    #             break
    #         if message.role == "assistant":
    #             if not mobile and "mobile" in message.content:
    #                 with open("mobile_files.txt", "a") as mobile_files:
    #                     mobile_files.write(json_path + "\n")
    #                 mobile = True
    #             if not os and "pyautogui" in message.content:
    #                 with open("os_files.txt", "a") as os_files:
    #                     os_files.write(json_path + "\n")
    #                 os = True
    # Aguvis stage 1     
    chat_messages = convert_to_code_agent_format(chat_messages, json_path, reasoning)
    return chat_messages


def convert_to_new_action_space(
    messages: list[ChatMessage], resolution: tuple[int, int], code_format: bool = True
) -> list[ChatMessage]:
    regex_match: re.Match | str | None = None
    index = -1
    regex = r"<code>\n(.*?)\n</code>"
    assistant_msg = [(i, message) for i, message in enumerate(messages) if message.role == "assistant"]
    if assistant_msg:
        for index, msg in assistant_msg:

            if code_format:
                regex_match = re.search(regex, msg.content, re.DOTALL)
            else:
                regex_match = msg.content

            if regex_match is not None:
                function_calls = parse_function_call(
                    regex_match.group(1) if isinstance(regex_match, re.Match) else regex_match,
                    pattern_to_match=["pyautogui", "mobile", "terminate", "answer"],
                )


                if len(function_calls) > 0:

                    for i, function_call in enumerate(deepcopy(function_calls)):

                        if function_call.function_name == "pyautogui.dragTo" and not isinstance(list(function_calls[i].parameters.values())[0], (list, tuple)):
                            x1, y1 = islice(function_calls[i-1].parameters.values(), 2)
                            x2, y2 = islice(function_calls[i].parameters.values(), 2)
                            function_calls[i].parameters = {"from_coord": (x1, y1), "to_coord": (x2, y2)}
                            function_calls[i].original_string = function_calls[i].to_string()
                            function_calls.pop(i-1)

                    function_calls = action_conversion(function_calls, resolution=resolution)

                    new_action_string = "\n".join(
                        [function_call.to_string() for function_call in function_calls]
                    )
                    messages[index].content = messages[index].content.replace(
                        regex_match.group(1) if isinstance(regex_match, re.Match) else regex_match, new_action_string
                    )


    return messages


def process_subset(
    config: Dict[str, Any],
    dataset_path: str,
) -> tuple[ConversationDataList, Path]:
    """Process a single dataset subset."""
    subset_name = config["subset_name"]

    print(f"Processing split: {subset_name}")

    dataset_dir = Path(dataset_path)
    images_folder = dataset_dir / config["subset_name"] / config["images_folder"]

    if not images_folder.exists():
        print(f"Images folder not found: {images_folder}")
    else:
        print(f"Images folder: {images_folder}")

    json_config_path = dataset_dir / config["json_path"]
    with open(json_config_path, "r") as f:
        data = ConversationDataList.model_validate_json(f.read())
        # data = f.read()
        print(f"Added '{json_config_path}'")

    return data, images_folder


def row_generator(
    data: ConversationDataList, images_folder: Path, json_path: str, reasoning: bool
) -> Generator[Dict[str, Any], None, None]:
    conversations: list[ConversationData] = data.root
    for item in tqdm(conversations):
        # Extract image paths from the data item
        try:
            # Load images
            image = load_image_from_folder(images_folder, item.image)
            chat_message = convert_to_chat_format(item, json_path, reasoning)
            chat_message = convert_to_new_action_space(chat_message, image.size, code_format=reasoning)
            if len(chat_message) == 0:
                continue

            row = DataRow.from_chat_messages(chat_message, image, source=json_path.split("/")[-1].split(".")[0])
            yield row.model_dump(exclude_none=True)
            del image
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing item: {e}", item)
            continue


class DatasetConfig(BaseModel):
    huggingface_repo_id: str
    local_path: str
    config_dict: List[Dict[str, Any]]
    smolagents_repo_id: str
    reasoning: bool


def process_single_config(config: Dict[str, Any], dataset_path: str, smolagents_repo_id: str, reasoning: bool) -> bool:
    """Process a single config in a separate process."""
    try:
        # Authenticate in this process
        authenticate_huggingface()
        
        print(f"\n{'=' * 50}")
        print(f"Processing config: {config}")

        # Check if the subset already exists in the remote dataset
        subset_name = config["subset_name"]
        # if check_subset_exists(smolagents_repo_id, subset_name):
        #     print(
        #         f"Subset '{subset_name}' already exists in {smolagents_repo_id}, skipping processing."
        #     )
        #     return True

        json_path = config["json_path"]
        data, image_folder = process_subset(config, dataset_path)

        # Collect all rows first
        rows = []
        datasets = []
        for row in row_generator(data, image_folder, json_path, reasoning):
            rows.append(row)
            if len(rows) > 20000:
                print("Creating batch dataset")
                dataset = Dataset.from_list(rows)
                datasets.append(dataset)
                rows = []
                gc.collect()
        
        if len(rows) > 0:
            # Create dataset from collected data
            dataset = Dataset.from_list(rows)
            datasets.append(dataset)
            rows = []

        dataset_to_push = concatenate_datasets(datasets)
        
        # Push to hub
        dataset_to_push.push_to_hub(
            smolagents_repo_id,
            # config_name=subset_name,  # This sets the subset name
            split="train",  # This should be "train" not the subset name
        )

        print(f"Processed and uploaded subset: {config['subset_name']}")

        # Force garbage collection to manage memory
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error processing config {config.get('subset_name', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return False


def make_dataset_from_original_data(dataset_config: DatasetConfig, max_processes: int | None = None):
    """Main function to orchestrate the entire process."""
    load_dotenv(override=True)

    print(f"Starting {dataset_config.smolagents_repo_id} dataset processing...")

    # Step 0: Authenticate with HuggingFace Hub
    authenticate_huggingface()

    dataset_path = download_dataset(
        dataset_config.huggingface_repo_id, dataset_config.local_path
    )

    # extract_zip_files(dataset_path)
    # extract_tar_parts_grouped(dataset_path)

    dataset_configs = discover_dataset_config(dataset_path, dataset_config.config_dict)
    converted_repo_id = dataset_config.smolagents_repo_id
    reasoning = dataset_config.reasoning
    
    # Use multiprocessing to process configs in parallel
    available_cpus = mp.cpu_count()
    if max_processes is None:
        max_processes = available_cpus
    num_processes = min(max_processes, len(dataset_configs))
    print(f"Using {num_processes} processes (out of {available_cpus} available CPUs) to process {len(dataset_configs)} configs")
    
    # Prepare arguments for multiprocessing
    process_args = [
        (config, dataset_path, converted_repo_id, reasoning) 
        for config in dataset_configs if config["subset_name"] if config["subset_name"] in ["guiact-web-single"]
    ]
    
    # Process configs in parallel with progress tracking
    print(f"Starting parallel processing of {len(dataset_configs)} configs...")
    try:
        with Pool(processes=num_processes) as pool:
            results = []
            for i, result in enumerate(pool.starmap(process_single_config, process_args)):
                results.append(result)
                print(f"Completed {i+1}/{len(dataset_configs)} configs")
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to sequential processing...")
        results = []
        for i, args in enumerate(process_args):
            result = process_single_config(*args)
            results.append(result)
            print(f"Completed {i+1}/{len(dataset_configs)} configs (sequential)")
    
    # Check results
    successful = sum(results)
    total = len(dataset_configs)
    print(f"\nProcessing complete: {successful}/{total} configs processed successfully")
    
    if successful < total:
        failed_count = total - successful
        print(f"Warning: {failed_count} configs failed to process. Check the logs above for details.")
    else:
        print("All configs processed successfully!")

#     # Cleanup
#     print("\nCleaning up temporary files...")
#     # shutil.rmtree(dataset_path, ignore_errors=True)
#
#     # api.upload_large_folder(folder_path=converted_folder, repo_id="smolagents/aguvis-stage-2", repo_type="dataset")
#
#     shutil.rmtree(converted_folder, ignore_errors=True)
#
#     print("All done!")


if __name__ == "__main__":
    # dataset_config_1 = DatasetConfig(
    #     huggingface_repo_id="xlangai/aguvis-stage1",
    #     local_path="/fsx/amir_mahla/aguvis_raw_stage_1",
    #     config_dict=config_dict_stage_1,
    #     smolagents_repo_id="smolagents/aguvis-stage-1",
    #     reasoning=False,
    # )
    # dataset_config_2 = DatasetConfig(
    #     huggingface_repo_id="xlangai/aguvis-stage2",
    #     local_path="/fsx/amir_mahla/aguvis_raw_stage_2",
    #     config_dict=config_dict_stage_2,
    #     smolagents_repo_id="smolagents/aguvis-stage-2",
    #     reasoning=True,
    # )
    dataset_config_3 = DatasetConfig(
        huggingface_repo_id="xlangai/aguvis-stage2",
        local_path="/fsx/amir_mahla/aguvis_raw_stage_2",
        config_dict=config_dict_stage_2,
        smolagents_repo_id="smolagents/guiact-web-single",
        reasoning=True,
    )
    # You can specify max_processes to limit the number of parallel processes
    # make_dataset_from_original_data(dataset_config_1, max_processes=4)
    make_dataset_from_original_data(dataset_config_3, 1)
