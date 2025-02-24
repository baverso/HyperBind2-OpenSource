# HyperBind2 (Open-Source Edition with ESM3)

Welcome to **HyperBind2**—an open-source deep learning framework for antibody design and binding affinity prediction, originally developed at **EVQLV** and adapted here for broader research use. This repository provides an accessible, scaled-down version of our internal platform, integrating **state-of-the-art protein language models** for antibody sequence embeddings.

---

## 🌟 Overview  

**HyperBind2** provides a deep learning framework for **antibody design and binding affinity prediction**.  
- This open-source version utilizes **ESM3 (Open-Source Edition: esm3-open-2024-03)**, the open 1.4B-parameter model, to generate **antibody sequence embeddings**.
- We demonstrate **fine-tuning** for antibody **heavy-light chain pair prediction** and leverage **contrastive learning** for robust representation learning:
  - **Contrastive Learning Explained:**  
    It teaches the model to distinguish between similar and dissimilar examples by learning from both positive and negative pairs. This process helps the model form a more nuanced and robust representation of antibody features.
  - **Small-Scale Advantage:**  
    Contrastive learning excels with small-scale scientific datasets—a common scenario in antibody research—by maximizing the informative signal from limited data.
- This release provides **an end-to-end workflow** featuring a streamlined, small version of our contrastive learning head while omitting proprietary components. Notably, we do not incorporate the mixture-of-experts approach used in our internal regression and classifier heads, underscoring that our open-source version is fundamentally different from our commercial system. Also our **custom transformer architectures** is substituted with ESM3 Open-Source Edition. 

> **Why ESM3 as the Encoder?**  
> ESM3 is currently the **leading BERT-based protein language model for sequence and structure prediction**, making it an ideal choice for open-source research.

## 🏆 Built with ESM3

This open-source model uses **ESM3 (Open-Source Edition: esm3-open-2024-03)**, the open 1.4B-parameter model from EvolutionaryScale, as its backbone for **state-of-the-art sequence and structure prediction**. By fine-tuning ESM3 on **antibody structures**, we enhance performance in structure-based learning for our open-source model, **HyperBind2**.

> **Attribution**  
> This model is derived from the **ESM3** model by EvolutionaryScale, licensed under the **EvolutionaryScale Cambrian Non-Commercial License Agreement**.  
> [ESM3 GitHub Repository](https://github.com/evolutionaryscale/esm)

---

### Risk Mitigation and Responsible Use

- **Data Filtering**: The original ESM3-open release excludes viral and toxin sequences as a precaution, reducing misuse risks.  
- **Local Regulations**: Please respect all community standards and legal requirements in synthetic biology or medical research.  
- **Intended Use**: This fine-tuned model is offered for **non-commercial** research and education. Note that we have only fine-tuned on antibodies—not viral proteins.

## ⚠️ About Our Internal Commercial Model

While this open-source release demonstrates key concepts, please note that **our commercial antibody AI model is entirely separate from ESM3 and is not a derivative of it**. Here’s what we can share about our proprietary antibody transformer:

- Our **internal model** is a **custom-built antibody transformer**, exclusively trained on **EVQLV's proprietary antibody datasets**.
- It leverages a **Mixture of Experts (MoE) approach** combined with a custom Transformer architecture tailored specifically for positional embeddings critical to antibody modeling, ensuring **CDR-sensitive sequence and structure understanding**.

These innovations empower our commercial solution to capture the complex nuances of antibody variable regions, delivering state-of-the-art performance in antibody design and analysis. To learn more about our commercial team and explore collaboration opportunities on discovery and optimization projects, please visit [evqlv.com](https://evqlv.com) and get in touch!

---

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

## 🚀 Features
- **ESM3 (Open-Source Edition: esm3-open-2024-03)** for **high-quality protein sequence embeddings**
- **PyTorch & PeFT (using LORA) support** for efficient fine-tuning  
  Our framework integrates the latest from [PEFT](https://github.com/huggingface/peft) with LoRA to streamline the fine-tuning process while reducing the number of trainable parameters.
- **Paired heavy-light chain input support** (real-world antibody data)
- **Contrastive Learning:**  
  - **Intuitive Explanation:**  
    Contrastive learning teaches the model to differentiate similar from dissimilar pairs by leveraging both positive and negative examples, which builds a robust internal representation of antibody features.
  - **Small-Scale Advantage:**  
    This method is especially effective with small-scale scientific datasets—typical in antibody research—maximizing the use of limited data.
- **Two Sample Datasets Included:**
  - **Real World Training Data** (anonymized heavy-light chain pairs, minimal labels)
  - **Synthetic Dataset** (Absolut! dataset from Greiff Lab)
- Example scripts for:
  - **Data preprocessing**
  - **Fine-tuning**
  - **Inference/prediction**

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
