# HyperBind2 (Open-Source Edition with ESM3)

Welcome to **HyperBind2**—an open-source deep learning framework for antibody design and binding affinity prediction, originally developed at **EVQLV** and adapted here for broader research use. This repository provides an accessible, scaled-down version of our internal platform, integrating **state-of-the-art protein language models** for antibody sequence embeddings.

## 🏆 Built with ESM3  
This open-source model leverages **ESM3**, a protein language model from EvolutionaryScale, for **state-of-the-art sequence and structure prediction**.  

> **Attribution:**  
> This model is a derivative of the **ESM3** model provided by EvolutionaryScale, licensed under the **EvolutionaryScale Cambrian Non-Commercial License Agreement**.

## ⚠️ About Our Internal Commercial Model  
While this open-source release demonstrates key concepts, **our commercial antibody AI model is separate** from ESM3 and **is not a derivative** of it.  
- Our **internal proprietary model** is a **custom antibody transformer**, uniquely trained on **EVQLV's proprietary antibody datasets**.
- **ESM3 is used only in this open-source version** to provide a strong baseline for non-commercial research.
- The **internal commercial version includes structure-based embeddings and additional enhancements** not present here.

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [What’s Inside](#whats-inside)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Inference / Prediction](#inference--prediction)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🌟 Overview  

**HyperBind2** provides a deep learning framework for **antibody design and binding affinity prediction**.  
- This open-source version utilizes **ESM3 (Open-Source)** to generate **antibody sequence embeddings**.
- We demonstrate **fine-tuning** for antibody **heavy-light chain pair prediction** using contrastive learning.
- This release provides **an end-to-end workflow** while omitting proprietary components (e.g., **custom transformer architectures** and **structure-based embeddings** used internally).

### ✅ Key Tasks Supported:
- **Binder vs. Non-Binder Classification**
- **Preliminary Affinity Ranking**
- **Exploring antibody–antigen modeling**

> **Why ESM3?**  
> ESM3 is currently the **leading BERT-based protein language model for sequence and structure prediction**, making it an ideal choice for open-source research.

---

## 🚀 Features
- **ESM3 (Open-Source Edition)** for **high-quality protein sequence embeddings**
- **PyTorch & LoRA support** for efficient fine-tuning
- **Paired heavy-light chain input support** (real-world antibody data)
- **Two Sample Datasets Included**:
  - **Real World Training Data** (anonymized heavy-light chain pairs, minimal labels)
  - **Synthetic Dataset** (Absolut! dataset from Greiff Lab)
- Example scripts for:
  - **Data preprocessing**
  - **Fine-tuning**
  - **Inference/prediction**

---

## 📂 What’s Inside?

```
HyperBind2/
|-- data/
|   |-- real_world/           # Paired heavy-light chain data (anonymized)
|   |-- synthetic/            # Synthetic dataset from Absolut!
|
|-- models/
|   |-- hyperbind_model.py    # Model architecture (PyTorch)
|
|-- scripts/
|   |-- train.py              # Demonstration training script
|   |-- inference.py          # Sample inference/prediction
|   |-- data_preprocess.py    # Data loading & preprocessing
|
|-- README.md                 # This README
|-- requirements.txt          # Dependencies
|-- LICENSE                   # License file
```

### **Key components include:**
- `hyperbind_model.py`: Defines the **PyTorch model architecture**.
- `train.py`: Runs a simplified **training loop** for antibody sequence ranking.
- `data_preprocess.py`: **Tokenizes** antibody sequences for ESM3 embedding.

---

## 🔧 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/evqlv/hyperbind2.git
    cd hyperbind2
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 Usage  

### 🗄️ Data Preparation  

1. **Download the Provided Data**:
   - `data/real_world/`: Partial **heavy-light chain pairs** (minimal binding labels).
   - `data/synthetic/`: **Synthetic dataset** from Absolut!.

2. **Preprocess Data**:
    ```bash
    python scripts/data_preprocess.py --data_dir data/real_world
    ```

### 🎯 Training the Model  

1. **Configure hyperparameters in** `train.py` (batch size, epochs, learning rate).  
2. **Run training**:
    ```bash
    python scripts/train.py --train_dir data/real_world --epochs 5 --batch_size 16
    ```

3. **Monitor**:
   - Training logs print metrics (accuracy, loss).
   - Add **TensorBoard support** for deeper visualization.

### 🔍 Inference / Prediction  

Run predictions on new sequences:
```bash
python scripts/inference.py --model_ckpt path/to/checkpoint --input_file data/synthetic/test_sequences.csv
```
Outputs a **binding probability score** per sequence pair.

---

## ⚠️ Limitations

1. **No Proprietary Model Weights**  
   - This repository does **not include** our **internal commercial model weights**.
   - Users must train on provided sample data.

2. **Scaled-Down Architecture**  
   - **Our internal model integrates advanced antibody-specific transformers** & **structure-based embeddings** not included in this release.

3. **Dataset Limitations**  
   - The **real-world dataset is anonymized** and **reduced in scope** compared to internal training data.

---

## 🤝 Contributing  

We welcome:
- **Bug fixes**
- **Documentation improvements**
- **General ML best practices**  

For feature requests related to our **full HyperBind pipeline**, open an issue—but note that **some IP-protected components remain proprietary**.

---

## 📜 License  

Distributed under the **Apache 2.0 License**.  
See `LICENSE` for details.

---

## 🎓 Acknowledgments  

- **ESM3 Model** from **EvolutionaryScale** (Open-Source Edition).  
- **Greiff Lab’s Absolut! dataset** for synthetic antibodies.  
- **EVQLV team** for discussions and dataset preparation.

> **Note:** This open-source edition is **for research use only** and is **not intended for clinical or commercial deployment** without further validation.

---

## 📢 Final Notes  

We hope **HyperBind2 (Open-Source Edition)** helps accelerate **antibody discovery research**!  

For questions, open an issue or **email us at info@evqlv.com**.  
