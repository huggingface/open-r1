[部分機翻版本]
# Open R1

*A fully open reproduction of DeepSeek-R1. This repo is a work in progress, let's build it together!*
*DeepSeek-R1的全開源復刻版本. 這個專案正在建置中，讓我們一起建立它吧!*

**目錄**

1. [概述](#overview)
2. [行動計畫](#plan-of-attack)
3. [安裝](#installation)
4. [模型訓練](#training-models)
   - [SFT](#sft)
   - [GRPO](#grpo)
5. [模型評估](#evaluating-models)
6. [重現 Deepseek 的評估結果](#reproducing-deepseeks-evaluation-results)
7. [資料生成](#data-generation)
   - [從小型蒸餾 R1 模型生成資料](#generate-data-from-a-smol-distilled-r1-model)
   - [從 DeepSeek-R1 生成資料](#generate-data-from-deepseek-r1)
8. [貢獻](#contributing)

## 概述

專案目標是建立 R1 流程中缺失的部分，以便所有人都能夠重現並在其基礎上進行建立。這個專案設計簡單，主要包含：

- `src/open_r1`:  包含用於訓練、評估模型以及生成合成資料的腳本：
    - `grpo.py`:  在給定資料集上使用 GRPO 訓練模型。
    - `sft.py`:  在資料集上對模型執行簡單的 SFT。
    - `evaluate.py`:  在 R1 基準測試上評估模型。
    - `generate.py`:  使用 [Distilabel](https://github.com/argilla-io/distilabel) 從模型生成合成資料。
- `Makefile`:  包含 R1 流程中每個步驟的簡易指令，並利用上述腳本。

### 行動計劃

我們將使用 DeepSeek-R1 的 [技術報告](https://github.com/deepseek-ai/DeepSeek-R1) 作為指南，該報告大致可以分為三個主要步驟：

* 步驟 1：透過從 DeepSeek-R1 蒸餾高品質語料庫，複製 R1-Distill 模型。
* 步驟 2：複製 DeepSeek 用於創造 R1-Zero 的純 RL 流程。這可能需要為數學、推理和程式碼，規劃新的大型資料集。
* 步驟 3：展示我們可以透過多階段訓練，從基礎模型轉變為經 RL 優化的模型。

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>


## 安裝

> [!CAUTION]
> 套件相依 CUDA 12.4。 如果您看到與分段錯誤相關的錯誤，請使用 `nvcc --version` 再次檢查您系統正在執行的版本。

要執行此專案中的程式碼，首先，建立 Python 虛擬環境，例如 `uv`。
要安裝 `uv`，請依照 [UV 安裝指南](https://docs.astral.sh/uv/getting-started/installation/)。


```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> 給 Hugging Face Cluster使用者：將 `export UV_LINK_MODE=copy` 添加到您的 `.bashrc` 檔案中，以防止 `uv` 發出快取警告。

下一步, 安裝 vLLM 及 FlashAttention:

```shell
uv pip install vllm==0.7.2
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
```

這也會安裝 PyTorch `v2.5.1` 版本，**務必** 使用此版本，因為 vLLM 二進制檔案是為此版本編譯的。 然後您可以透過 `pip install -e .[LIST OF MODES]` 安裝針對您特定使用案例的剩餘套件。 對於大多數貢獻者，我們推薦：

```shell
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

下一步, 請依照下列步驟登入你的 Hugging Face 和 Weights & Biases 帳號:

```shell
huggingface-cli login
wandb login
```

最後，請確認您已安裝 Git LFS，以便您可以載入和推送模型/資料集到 Hugging Face Hub：

```shell
git-lfs --version
```

若還沒安裝, 請執行:

```shell
sudo apt-get install git-lfs
```

## 模型訓練

我們支援使用 DDP 或 DeepSpeed (ZeRO-2 和 ZeRO-3) 訓練模型。 例如，若要使用從 DeepSeek-R1 蒸餾並包含推理軌跡的資料集 (例如 [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)) 執行 SFT，請執行：

```shell
# Train via command line
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 16384 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill

# Train via YAML config
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

目前支援以下任務：

* 監督式微調 (Supervised Fine-Tuning) `sft`
* 群組相對策略最佳化 (Group Relative Policy Optimization) `grpo`

> [!TIP]
> 如果你增加/減少 GPU 的數量，我們建議同時增加每裝置batch大小或梯度累加步數，以維持全域batch大小不變。

預設情況下，這些腳本會將每個模型推送到您的 Hugging Face Hub 使用者名稱下，即 `{username}/{model_name}-{task}`。 您可以透過在指令中附加參數來覆寫每個 YAML 設定中的參數，如下所示：

```shell
# Change batch size, number of epochs etc
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
    --per_device_train_batch_size=1 --num_train_epochs=5
```

如果您也希望覆寫 Weights & Biases 的預設設定，您可以依照以下方式操作：

```shell
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
    --wandb_entity huggingface --wandb_project open-r1 --run_name Qwen2.5-1.5B-GRPO
```

> [!NOTE]
> 下方的訓練指令是針對具有 8 個 H100 (80GB) GPU 的節點所設定的。 對於不同的硬體和拓撲結構，你可能需要調整batch大小和梯度累加步數。

### SFT

若要使用從 DeepSeek-R1 蒸餾並包含推理軌跡的資料集 (例如 [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)) 執行 SFT：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

### GRPO

為了透過 GRPO 訓練器進行訓練，我們使用一個 GPU 運行 vLLM 以加速生成，並使用剩餘的 GPU 進行訓練。 例如，在具有 8 個 GPU 的節點上，設定 `--num_processes` 以覆寫 `accelerate` 設定中的預設值：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml
```

> [!WARNING]
> 蒸餾版 DeepSeek 模型中使用的聊天模板省略了 `<think>` 和 `</think>` 標籤內的推理區塊內容。 它還使用 `<think>` 預填充了助理回應，這會干擾格式獎勵函數。 為了處理這個問題，務必依照範例覆寫聊天模板，如 [recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml](./recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml)。


我們提供了一個使用 GRPO 進行數學推理的最小可重現實驗，參考了 [SimpleRL-Reason](https://hkust-nlp.notion.site/simplerl-reason) 中的方法，該方法使用在 8K 範例上訓練的 7B 模型。 在 8 個 H100 80G GPU 上執行此實驗大約需要 3 小時：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml
```

我們最終的[模型](https://huggingface.co/Dongwei/Qwen-2.5-7B_Base_Math_smalllr)雖然使用了不同的learning rates、loss functions 和 reward structures，但在 MATH-500 基準測試上達到了 69.4% 的準確度，展現出比基礎模型提升了 17% 以上的效能。

#### 👨‍💻 使用程式碼直譯器進行訓練

我們提供了一個 `code` 獎勵函數，用於在訓練期間執行策略生成的程式碼。目前，此獎勵函數的目標是像 [Codeforces](https://codeforces.com) 這樣的程式碼競賽，在這些競賽中，解決方案會針對一組測試案例執行，並將總體成功率作為最終獎勵回傳。為了確保安全執行，我們使用 [E2B](https://e2b.dev) 沙箱，它們執行速度快且成本低廉。若要使用此獎勵函數，請先安裝必要的套件：

```shell
uv pip install -e '.[code]'
```

接著，建立一個 `.env` 檔案並將 E2B 的 API token 放入其中：

```
E2B_API_KEY="e2b_xxx"
```

接著，請確保你的資料集包含一個 `verification_info` 欄位，具有以下 schema (採用自 PrimeIntellect 出色的可驗證問題集[資料集](https://huggingface.co/collections/PrimeIntellect/synthetic-1-67a2c399cfdd6c9f7fae0c37))：

```python
{
    "language": "python",
    "test_cases": [
        {
            "input": "4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n",
            "output": "1\n3 \n-1\n0\n\n2\n1 2 \n",
            "type": "stdin_stdout",
        }
    ],
}
```

例如，若要訓練一個小型模型來解決 Python 問題，請執行：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml
```

#### 資料去汙染

根據 [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) (Simple test-time scaling) 中的方法，可以使用以下腳本對資料進行去汙染處理：[scripts/decontaminate.py](./scripts/decontaminate.py)。此腳本使用 8-gram 進行去汙染，並對資料進行去重覆。執行範例：

```shell
python scripts/decontaminate.py \
    --dataset "open-r1/verifiable-coding-problems-python" \
    --problem_column problem \
    --cleanup
```


它將針對基準測試資料集進行去汙染，並在之後移除受汙染的樣本。如果沒有提供 `--new_dataset_name` 參數，則會重複使用相同的資料集，並添加 `_decontaminated` 後綴。它針對提示 (對於此資料集，提示是 `problem` 欄位) 執行，但也可以提供不同的欄位。

腳本的參數：

```shell
usage: decontaminate.py [-h] --dataset DATASET [--split SPLIT] [--ngram_size NGRAM_SIZE] [--problem_column PROBLEM_COLUMN] [--cleanup] [--new_dataset_name NEW_DATASET_NAME]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of the dataset to check for contamination.
  --split SPLIT         Split to check for contamination, defaults to `train`.
  --ngram_size NGRAM_SIZE
                        Size of n-grams to build, defaults to 8.
  --problem_column PROBLEM_COLUMN
                        Name of the column containing the problem (prompt).
  --cleanup           Whether to remove the contaminated rows before pushing the dataset.
  --new_dataset_name NEW_DATASET_NAME
                        New name for the dataset. If not provided, will reuse the name and add a `_decontaminated` to the name.
```

### 在 Slurm 集群上啟動任務

如果你有權限存取 Slurm 集群，我們提供了一個 `slurm/train.slurm` 腳本，它將自動為你排隊訓練任務。 以下是如何使用它：

```shell
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm {model_name} {task} {config_suffix} {accelerator}
```

這裡，`{model_name}` 和 `{task}` 的定義如上所述，而 `{config_suffix}` 指的是特定的配置，`{accelerator}` 指的是 `recipes/accelerate_configs` 中 🤗 Accelerate 配置的選擇。 如果你希望覆寫預設的配置參數，你可以透過附加一個以空格分隔的字串來提供它們，例如 `'--arg1=value1 --arg2=value2'`。 以下是在 1 個節點 (8 個 GPU) 上運行 SFT 的具體範例：

```shell
# Launch on Slurm and override default hyperparameters
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm Qwen2.5-1.5B-Instruct sft demo zero3 '--per_device_train_batch_size=1 --num_train_epochs=5'
```

你可以透過增加 `--nodes` 標誌來擴展節點數量。

> [!NOTE]
> `slurm/train.slurm` 中的配置已針對 Hugging Face Compute Cluster 進行最佳化，可能需要調整才能適用於你自己的計算節點。

## 模型評估

我們使用 `lighteval` 來評估模型，並在 `src/open_r1/evaluate.py` 中定義了自訂任務。對於可以在單個 GPU 上運行的模型，請執行：

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

> [!IMPORTANT]
> 你務必在 `vllm` 命令中設定 `max_model_length=32768`，以對齊我們在每次評估中定義的 `max_new_tokens`。如果沒有這樣做，`lighteval` 將會拋出錯誤。

為了提高跨多個 GPU 的吞吐量，請使用_data parallel_，如下所示：

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

對於需要跨 GPU 分片的大型模型，請使用_張量並行_並執行：

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

你也可以使用 `make evaluate` 啟動評估，並指定模型、任務，以及可選的並行技術和 GPU 數量。

若要使用單個 GPU 進行評估：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

若要使用 Data Parallelism：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

若要使用 Tensor Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## 重現 Deepseek 的評估結果

> [!NOTE]
> DeepSeek-R1 論文使用每個查詢取樣 64 個回應來估計 `pass@1`。 在下方，我們報告了每個查詢取樣 1 個回應的結果，這可能解釋了我們的結果與他們的結果之間 1-3σ 的小幅差異。

### AIME 2024

我們能夠在約 1-3 個標準差範圍內重現 Deepseek 在 AIME 2024 基準測試上報告的結果：

| Model                         | AIME 2024 (🤗 LightEval) | AIME 2024 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          26.7           |             28.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          56.6           |             55.5             |
| DeepSeek-R1-Distill-Qwen-14B  |          60.0           |             69.7             |
| DeepSeek-R1-Distill-Qwen-32B  |          73.2           |             72.6             |
| DeepSeek-R1-Distill-Llama-8B  |          43.3           |             50.4             |
| DeepSeek-R1-Distill-Llama-70B |          73.3           |             70.0             |

若要重現這些結果，請使用以下命令：

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|aime24|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

或者，你也可以依照以下方式啟動 Slurm 任務：

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks aime24
```

### MATH-500

我們能夠在約 1-3 個標準差範圍內重現 Deepseek 在 MATH-500 基準測試上報告的結果：

| Model                         | MATH-500 (🤗 LightEval) | MATH-500 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          84.6           |             83.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          93.0           |             92.8             |
| DeepSeek-R1-Distill-Qwen-14B  |          95.0           |             93.9             |
| DeepSeek-R1-Distill-Qwen-32B  |          96.6           |             94.3             |
| DeepSeek-R1-Distill-Llama-8B  |          88.6           |             89.1             |
| DeepSeek-R1-Distill-Llama-70B |          96.4           |             94.5             |

若要重現這些結果，請使用以下命令：

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

或者，你也可以依照以下方式啟動 Slurm 任務：

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks math_500
```

### GPQA Diamond

我們能夠在約 1-3 個標準差範圍內重現 Deepseek 在 GPQA Diamond 基準測試上報告的結果：

| Model                         | GPQA Diamond (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:---------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |            34.3             |               33.8               |
| DeepSeek-R1-Distill-Qwen-7B   |            50.5             |               49.1               |
| DeepSeek-R1-Distill-Qwen-14B  |            59.6             |               59.1               |
| DeepSeek-R1-Distill-Qwen-32B  |            63.6             |               62.1               |
| DeepSeek-R1-Distill-Llama-8B  |            52.0             |               49.0               |
| DeepSeek-R1-Distill-Llama-70B |            67.2             |               65.2               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|gpqa:diamond|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks gpqa
```

### LiveCodeBench

我們能夠在約 1-3 個標準差範圍內重現 Deepseek 在 LiveCodeBench 程式碼生成基準測試上報告的結果：

| Model                         | LiveCodeBench (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:----------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |             16.3             |               16.9               |
| DeepSeek-R1-Distill-Qwen-7B   |             36.6             |               37.6               |
| DeepSeek-R1-Distill-Qwen-14B  |             51.5             |               53.1               |
| DeepSeek-R1-Distill-Qwen-32B  |             56.6             |               57.2               |
| DeepSeek-R1-Distill-Llama-8B  |             37.0             |               39.6               |
| DeepSeek-R1-Distill-Llama-70B |             54.5             |               57.5               |

若要重現這些結果，請使用以下命令：

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models, or data_parallel_size=8 with the smaller models for speed
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks lcb
```

## 資料生成

### 從小型蒸餾 R1 模型生成資料

以下範例可以在 1xH100 上運行。
首先，安裝以下依賴項：

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

現在，將以下程式碼片段儲存到名為 `pipeline.py` 的檔案中，並使用 `python pipeline.py` 運行它。 它將為 10 個範例中的每一個生成 4 個輸出 (請將儲存庫的使用者名稱更改為你的組織/使用者名稱)：

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

不妨查看一下 [HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b) 的範例資料集。


### 從 DeepSeek-R1 生成資料

若要運行更大的 DeepSeek-R1 模型，我們使用了 2 個節點，每個節點配備 8 個 H100 GPU，並使用此儲存庫中 `slurm/generate.slurm` 的 slurm 檔案。 首先，安裝依賴項：

(（目前）我們需要安裝 vllm 開發版本 wheel 包，它[修復了 R1 cuda graph capture 問題](https://github.com/vllm-project/vllm/commits/221d388cc5a836fa189305785ed7e887cea8b510/csrc/moe/moe_align_sum_kernels.cu))
```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

然後運行以下命令：

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]
> 在任務運行時，你可以透過集群登入節點建立 SSH 隧道，以便從你的電腦存取 Ray 儀表板。 運行 `ssh -L 8265:ray_ip_head_node:8265 <login_node>` 後，即可瀏覽 `http://localhost:8265`。

## 貢獻

歡迎貢獻。 請參閱 https://github.com/huggingface/open-r1/issues/23。
