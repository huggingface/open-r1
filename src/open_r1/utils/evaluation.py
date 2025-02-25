import subprocess
from typing import TYPE_CHECKING, Dict, Union

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id


if TYPE_CHECKING:
    from trl import GRPOConfig, SFTConfig, ModelConfig

import os


# We need a special environment setup to launch vLLM from within Slurm training jobs.
# - Reference code: https://github.com/huggingface/brrr/blob/c55ba3505686d690de24c7ace6487a5c1426c0fd/brrr/lighteval/one_job_runner.py#L105
# - Slack thread: https://huggingface.slack.com/archives/C043JTYE1MJ/p1726566494958269
user_home_directory = os.path.expanduser("~")
VLLM_SLURM_PREFIX = [
    "env",
    "-i",
    "bash",
    "-c",
    f"for f in /etc/profile.d/*.sh; do source $f; done; export HOME={user_home_directory}; sbatch ",
]


def register_lighteval_task(
    configs: Dict[str, str], eval_suite: str, task_name: str, task_list: str, num_fewshot: int = 0
):
    """Registers a LightEval task configuration.

    - Core tasks can be added from this table: https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl
    - Custom tasks that require their own metrics / scripts, should be stored in scripts/evaluation/extended_lighteval_tasks

    Args:
        configs (Dict[str, str]): The dictionary to store the task configuration.
        eval_suite (str, optional): The evaluation suite.
        task_name (str): The name of the task.
        task_list (str): The comma-separated list of tasks in the format "extended|{task_name}|{num_fewshot}|0" or "lighteval|{task_name}|{num_fewshot}|0".
        num_fewshot (int, optional): The number of few-shot examples. Defaults to 0.
        is_custom_task (bool, optional): Whether the task is a custom task. Defaults to False.
    """
    # Format task list in lighteval format
    task_list = ",".join(f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(","))
    configs[task_name] = task_list


LIGHTEVAL_TASKS = {}

register_lighteval_task(LIGHTEVAL_TASKS, "custom", "math_500", "math_500", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "aime24", "aime24", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "aime25", "aime25", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "gpqa", "gpqa:diamond", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)


def get_lighteval_tasks():
    return list(LIGHTEVAL_TASKS.keys())


SUPPORTED_BENCHMARKS = get_lighteval_tasks()


def run_lighteval_job(
    benchmark: str, training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig"
) -> None:
    task_list = LIGHTEVAL_TASKS[benchmark]
    model_name = training_args.hub_model_id
    model_revision = training_args.hub_model_revision
    # For large models >= 30b params or those running the MATH benchmark, we need to shard them across the GPUs to avoid OOM
    num_gpus = get_gpu_count_for_vllm(model_name, model_revision)
    if get_param_count_from_repo_id(model_name) >= 30_000_000_000:
        tensor_parallel = True
    else:
        num_gpus = 8
        tensor_parallel = False

    cmd = VLLM_SLURM_PREFIX.copy()
    cmd_args = [
        f"--gres=gpu:{num_gpus}",
        f"--job-name=or1_{benchmark}_{model_name.split('/')[-1]}_{model_revision}",
        "slurm/evaluate.slurm",
        benchmark,
        f'"{task_list}"',
        model_name,
        model_revision,
        f"{tensor_parallel}",
        f"{model_args.trust_remote_code}",
    ]
    if training_args.system_prompt is not None:
        cmd_args.append(f"--system_prompt={training_args.system_prompt}")
    cmd[-1] += " " + " ".join(cmd_args)
    subprocess.run(cmd, check=True)


def run_benchmark_jobs(training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig") -> None:
    benchmarks = training_args.benchmarks
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # Evaluate on all supported benchmarks. Later we may want to include a `chat` option
        # that just evaluates on `ifeval` and `mt_bench` etc.

    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")
