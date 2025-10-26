
# 🧬 Difftopo Topology Generation and Design Pipeline

This repository provides a complete workflow for paper **Leveraging protein representations to explore uncharted fold spaces with generative models** using **RFdiffusion** and **ColabDesign**.

---

## 📦 Environment Setup

### 1. Create Conda Environment

First, create the environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate your_env_name
```

### 2. Install Required External Tools

#### 🔹 RFdiffusion

Clone and install **RFdiffusion**:

```bash
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion
pip install -e .
cd ..
```

#### 🔹 ColabDesign

Clone and install **ColabDesign**:

```bash
git clone https://github.com/sokrypton/ColabDesign.git
cd ColabDesign
pip install -e .
cd ..
```

---

## 🧱 Dataset Preparation

### 1. Install STRIDE

To reproduce our training dataset, you must first install **STRIDE**, a tool for secondary structure assignment:
👉 [STRIDE Download Page](https://webclu.bio.wzw.tum.de/stride/)

After installation, use STRIDE to generate `.ss` secondary structure files for your protein database.

Example:

```bash
stride input.pdb > output.ss
```

### 2. Preprocess the Dataset

Once `.ss` files are ready, preprocess the dataset using:

```bash
python preprocessing.py --pdbdir  --pdbssdir --savedir 
```

We also provide a preprocessed **CATH dataset** under google drive: https://drive.google.com/file/d/1fCb-DbtBwBf35AAtUUMMqNiDNXGItS7h/view?usp=sharing

```
./dataset/
```

---

## 🧩 Topology Generation

To generate protein topologies, open and run:

```
generate_topology.ipynb
```

This notebook supports:

* **Unconditional generation**
* **Specified SSE string sampling**

Two example cases are included in the notebook for reference.

The generated topologies will be saved as `.npy` files.

---

## 🎨 Sketch Building and Downstream Design

After topology generation, use:

```bash
python build_sketch.py --input topology npy --outputdir ...
```

This script:

* Builds **protein sketches** from generated topologies
* Prepares **RFdiffusion** input scripts
* Prepares **ColabDesign** scripts for downstream protein design



