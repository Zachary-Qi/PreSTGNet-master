# PreSTGNet

Official implementation of **PreSTGNet**, a two-stage framework for traffic flow prediction based on **pre-training and fine-tuning**.

Our work has been published in **Advanced Engineering Informatics**.  
If you find this repository useful for your research, welcome to **star this repository** and **cite our work**.

---

## Highlights

- A two-stage framework for traffic flow prediction
- Pre-training on spatiotemporal data representation
- Fine-tuning for downstream forecasting
- Support for distributed training with multi-GPU
- Experiments on six public benchmark datasets:
  - **PeMS03**
  - **PeMS04**
  - **PeMS07**
  - **PeMS08**
  - **BikeNYC**
  - **TaxiNYC**

---

## Published Paper

**Spatiotemporal Graph Neural Network for Traffic Flow Prediction Based on a Two-Stage Pre-training and Fine-tuning Framework**  
*Advanced Engineering Informatics*, 2026

---

## Repository Structure

```bash
.
├── lib/
│   └── generate_training_data.py   # dataset generation
├── main.py                         # pre-training / training
├── test.py                         # testing
├── requirements.txt               # dependencies
└── experiments/                   # checkpoints and logs
```

---

## Environment Setup

Please first install the required packages:

```bash
pip install -r requirements.txt
```

---

## Datasets

The experimental evaluation is conducted on six public benchmark datasets:

- **PeMS03, PeMS04, PeMS07, PeMS08**: freeway traffic flow datasets
- **BikeNYC, TaxiNYC**: urban mobility datasets

### Dataset Description

The **PeMS** datasets are collected by the **California Performance Measurement System (PeMS)** and record freeway traffic volume every **5 minutes** using loop detectors deployed across the road network.

To further evaluate the generalization ability of the framework beyond freeway scenarios, two citywide mobility datasets are also included:

- **BikeNYC**: shared-bike demand flows in New York City
- **TaxiNYC**: taxicab trip records in New York City

Following prior work, both BikeNYC and TaxiNYC are aggregated into **grid-based spatiotemporal sequences** with **30-minute intervals**, where the prediction targets correspond to urban mobility demand flows.


---

## Trained Model Weights

The trained model weights are available through Google Drive:

[Download trained weights](https://drive.google.com/file/d/174tHQt3lBjxs1XQf6y5P7_jEBwGSPDBf/view?usp=sharing)

After downloading the weights, please place them in the corresponding experiment directory or update the `PRE_TRAINED_WEIGHT_PATH` and `save_dir` settings in `main.py` and `test.py` according to your local file path.

---

## Quick Start

The complete workflow consists of four steps:

1. Generate dataset
2. Pre-train the model
3. Fine-tune / train the model
4. Test the trained model

---

## Step 1: Generate Dataset

Before training, you need to generate the processed dataset.

Open `lib/generate_training_data.py` and set:

```python
DATASET_NAME = "PeMS08"
MODE = "PreTrain"   # PreTrain or Train
```

### Parameters

- `DATASET_NAME`: dataset name, e.g., `PeMS04`, `PeMS08`
- `MODE`:
  - `PreTrain`: generate data for the pre-training stage
  - `Train`: generate data for the second-stage training

### Run

From the project root directory, execute:

```bash
python lib/generate_training_data.py
```

Please run this script **twice**:

- once with `MODE = "PreTrain"`
- once with `MODE = "Train"`

---

## Step 2: Pre-training

In `main.py`, set the following parameters for the pre-training stage:

```python
DATASET_NAME = "PeMS04"
MODEL_NAME = "PreSTGNet"
MODE = "PreTrain"      # only Train or PreTrain
PRE_TRAINED_WEIGHT_PATH = ""   # path to pretrained model weights
Breakpoint_Destination_Folder = ""  # path to resume checkpoint folder
```

### Notes

- `MODE = "PreTrain"` means the model runs in the pre-training stage
- `PRE_TRAINED_WEIGHT_PATH` can be left empty if training from scratch
- `Breakpoint_Destination_Folder` can be used to resume interrupted training

### Run

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=29507 main.py
```

### Command Explanation

- `CUDA_VISIBLE_DEVICES=2,3`: use GPU 2 and GPU 3
- `--nproc_per_node=2`: use 2 GPUs / 2 processes
- `--master_port=29507`: communication port for distributed training

---

## Step 3: Fine-tuning / Second-stage Training

After pre-training is completed, modify `main.py` as follows:

```python
MODE = "Train"
PRE_TRAINED_WEIGHT_PATH = ""
```

Set `PRE_TRAINED_WEIGHT_PATH` to the pretrained checkpoint obtained in Step 2.

### Example

```python
PRE_TRAINED_WEIGHT_PATH = "./experiments/PeMS04_lsfgcn/PreTrain/2025-04-28_17-50/session_1/checkpoint.pth"
```

### Run

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=29507 main.py
```

### Command Explanation

- `CUDA_VISIBLE_DEVICES=2,3`: use GPU 2 and GPU 3 simultaneously
- `--nproc_per_node=2`: number of GPUs used for training
- `--master_port=29507`: port number for distributed training

---

## Step 4: Testing

In `test.py`, set:

```python
PRE_TRAINED_WEIGHT_PATH = "./experiments/PeMS04_lsfgcn/PreTrain/2025-04-28_17-50/session_1/checkpoint.pth"  # pretrained model weight path

save_dir = "./experiments/PeMS04_lsfgcn/Train/2025-04-29_14-33/session_1"  # trained model folder
```

### Run

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=29507 test.py
```

---

## Example Workflow

A typical experiment pipeline is:

```text
Generate PreTrain data
   ↓
Generate Train data
   ↓
Run pre-training
   ↓
Run second-stage training with pretrained weights
   ↓
Run testing
```

---

## Checkpoints and Logs

Typical outputs are saved in:

```bash
./experiments/
```

Example:

```bash
./experiments/PeMS04_lsfgcn/PreTrain/2025-04-28_17-50/session_1/checkpoint.pth
./experiments/PeMS04_lsfgcn/Train/2025-04-29_14-33/session_1
```

---

## Notes

- Please ensure that `DATASET_NAME` and `MODE` are correctly set before each stage.
- Please ensure that `PRE_TRAINED_WEIGHT_PATH` is correctly specified before fine-tuning and testing.
- If the distributed port is occupied, change `--master_port` to another available port.
- If you want to use different GPUs, modify `CUDA_VISIBLE_DEVICES` accordingly.

---

## Citation

If you find this repository useful for your research, please kindly cite our work:

```bibtex
@article{ZHANG2026104572,
title = {Spatiotemporal graph neural network for traffic flow prediction based on a two-stage pre-training and fine-tuning framework},
journal = {Advanced Engineering Informatics},
volume = {73},
pages = {104572},
year = {2026},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2026.104572},
url = {https://www.sciencedirect.com/science/article/pii/S1474034626002648},
author = {Shiqi Zhang and Zhen Liu and Chenliang Wo and Jialong Qian and Yonghong Liu and H. Oliver Gao},
keywords = {Traffic flow prediction, Long-range dependencies, Pretraining–finetuning, Masked sequence reconstruction},
abstract = {Accurate traffic flow prediction is essential for intelligent transportation systems. Existing short-horizon models rely on limited recent observations, which makes it difficult to capture long-range dependencies and global spatiotemporal structures and can yield overly smooth or unrealistic forecasts. We propose a two-stage pre-training and fine-tuning framework that first learns long-range spatiotemporal regularities from long historical sequences through masked sequence reconstruction, and then transfers the learned representations to short-horizon forecasting via a lightweight predictor with feature alignment and fusion. The framework also accounts for time-varying spatial dependencies and heterogeneous traffic patterns to improve robustness. Experiments on multiple benchmark datasets show consistent improvements over strong baselines, and further analyses suggest that the method better preserves spatial differences across sensors, reducing overly smoothed forecasts. The source code is available at: https://github.com/Zachary-Qi/PreSTGNet-master.}
}
```

We sincerely welcome researchers and practitioners to use this repository and cite our work.

---

## Contact

For questions about the code or experiments, please contact the authors.
