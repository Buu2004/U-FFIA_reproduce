# U-FFIA Reproduction

Reproduction code for the U-FFIA framework (Unified Multimodal + Knowledge Distillation), as proposed by Meng Cui et al.

## 🛠 Installation & Setup

First, clone the repository and set up the Conda environment using the provided YAML file.

```bash
# Clone the repository
git clone https://github.com/Buu2004/U-FFIA_reproduce.git
cd U-FFIA_reproduce

# Create and activate the environment
conda env create -f environment.yml
conda activate u-ffia

```

## 📂 Data Preparation

### 1. Download Datasets

Download the required datasets from the following Zenodo repositories:

* [Part 1](https://zenodo.org/records/11059975)
* [Part 2](https://zenodo.org/records/11060195)
* [Part 3](https://zenodo.org/records/11060370)

### 2. Organize Directories

Place the downloaded files into the respective directories as follows:

* **Audio files:** `audio_dataset/`
* **Video files:** `video_dataset/`

## ⬇️ Pretrained Models

Download the required checkpoints from [this Google Drive link](https://drive.google.com/drive/folders/1fh-Lo3S7-aTgfPni5-IeG5_-P7MBKBfL).

Ensure the following weight files are placed inside the `pretrained_models/` directory:

1. `PANNs/Cnn10.pth`
2. `video_best.pt`
3. `MV2/audio_best.pt`

**Directory Structure:**

```text
pretrained_models/
├── PANNs/
│   └── Cnn10.pth
├── video_best.pt
└── MV2/
    └── audio_best.pt

```

## 🚀 Execution

### 1. Preprocessing

Run the following script to generate the necessary pickle files for the video dataset:

```bash
python dataset/fish_video_dataset.py

```

### 2. Training / Running the Model

Execute the main script to run the proposed unified model (Multimodal + Knowledge Distillation).

```bash
python main_kl_unified.py

```

> **Note:** Hyperparameters for training can be modified in the configuration file: `config/unified/exp1_av.yaml`.