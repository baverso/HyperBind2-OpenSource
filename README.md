# HyperBind2 (Open-Source Edition)

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
# Contrastive Learning Model Head with Structural Embeddings

The contrastive learning component of **HyperBind2** is designed to fine-tune representations of **multimerized antibody sequences**—a concatenation of the **heavy chain**, **light chain**, and **antigen**. Each chain is first transformed using **ESM3-like embeddings**, capturing both sequence and predicted structural features (e.g., residue-level contexts, secondary structure, and long-range attention signals). These embeddings serve as the foundation for a **contrastive learning** model head that differentiates higher vs. lower binding affinity in a pairwise context.

---

## What is Contrastive Learning?

Contrastive learning encourages the model to learn rich, discriminative features by contrasting pairs of sequences. Rather than predicting an absolute binding value, the model compares two sequences and determines which one exhibits the higher affinity. This process is typically more data-efficient than traditional supervised methods when data is limited.

**Key Principles**:
1. **Positive/Negative Pairing**: For any given pair of antibody clones $(x_i, x_j)$, the model attempts to push together the representations of sequences that have higher affinity and push apart those that do not.  
2. **Relative Comparison**: The model focuses on *relative* differences between sequences under the context of a given antigen, which is especially helpful in scientific scenarios where absolute labels may be noisy or limited.

A common formulation of contrastive learning uses an **InfoNCE**-style objective, where the network is asked to maximize the log-likelihood of correctly identifying a positive pair among a set of negatives. While HyperBind2 simplifies this concept into a binary classification problem (“higher vs. lower” affinity), the underlying philosophy remains similar—**learn robust embeddings through explicit comparisons**.

---

## Architecture and Hidden Layers

### Multimerized Input Assembly

1. **Embeddings**  
   - **Heavy Chain**, **Light Chain**, and **Antigen** each produce high-dimensional embeddings that incorporate:
     - **Residue-level context** (e.g., local and global sequence information).
     - **Predicted structural cues** (e.g., secondary structure, angles, side chain orientations, distances, and contact maps).
     - **Attention-based contact patterns** (long-range dependencies).

2. **Concatenation**  
   - These three embeddings are **concatenated** into a single “multimerized” representation, capturing interactions across the entire antibody-antigen complex.

### Base Architecture

The architecture can be visualized as follows:
```
[ Heavy Embedding ]   [ Light Embedding ]   [ Antigen Embedding ]
↓                    ↓                       ↓
Dropout → Dense(4) → Flatten         (repeat for each chain)
↓                    ↓                       ↓
–––– Concatenate all three flattened vectors ––––
↓
┌──────────────────────────────────────────────────────────┐
│        Block 1 → Dropout → Dense(256, ReLU) → Norm      │
│        Block 2 → Dropout → Dense(128, ReLU) → Residual  │
│        Block 3 → Dropout → Dense(64,  ReLU) → Norm      │
└──────────────────────────────────────────────────────────┘
↓
Dropout → Dense(2, Softmax)
↓
Probability (Higher/Lower)
```
1. **Dimension Reduction and Feature Flattening**:  
   - Each embedded chain passes through:
     - A **dropout layer** (e.g., 10–20%).
     - A **dense layer** (4 units, ReLU) to reduce dimensionality.
     - **Flatten** to form a compact vector.
2. **Joint Representation**:  
   - The three flattened vectors (heavy, light, antigen) are concatenated.
3. **Intermediate Processing**:  
   - The concatenated vector is fed into another **dropout layer** and a **dense layer** with 256 units (ReLU activation).
4. **Final Classification**:  
   - A **dropout** is applied before a **dense layer** with 2 units and softmax activation, yielding a probability distribution indicating higher vs. lower affinity.

By stacking multiple feed-forward blocks (each with dropout, ReLU, and optional residual or normalization layers), the network gains additional capacity to model complex antibody-antigen interactions.

---

## Loss (Objective) Functions

### Binary Classification Setting

Because HyperBind2’s contrastive learning is operationalized as a *binary classification* (higher vs. lower affinity), the objective function is the **sparse categorical cross-entropy**:

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \Big(p_{i,c}(x_i; \theta)\Big)
$$

- $(N)$: Number of training examples  
- $(C)$: Number of classes (2: higher or lower)  
- $(y_{i,c})$: One-hot encoded label for example $(i)$ in class $(c)$  
- $(p_{i,c})$: Predicted probability of class \(c\) for example \(i\) given parameters \(\theta\)

### Pairwise Contrastive Formulation (Alternate View)

For certain tasks, you might adopt a margin-based or InfoNCE-like formulation. A simplified variant is:

$$
\mathcal{L}_{\text{contrast}}(\theta) = \sum_{(i,j)\in P} \max \Big(0,\; m - \big(f_{\theta}(x_i) - f_{\theta}(x_j)\big)\Big)
$$

where $(i,j)$ in $P$ denotes pairs of sequences known to have a relative difference, $(m)$ is a margin constant, and $(f_{(\theta)}(x))$ is the model’s scoring function. However, in HyperBind2’s open-source release, we stick to the cross-entropy classification approach as described above.

---

## Optimization Methods

1. **Adam Optimizer**  
   - Adaptive learning rate and momentum help navigate the potentially noisy high-dimensional space common in protein models.  

2. **Dropout and Batch Management**  
   - Dropout layers scattered throughout the network mitigate overfitting.  

3. **Parameter-Efficient Fine-Tuning (PEFT)**  
   - To fine-tune large models efficiently, we leverage **PEFT techniques** such as **LoRA, AdaLoRA, and Prefix Tuning**.  
   - **LoRA (Low-Rank Adaptation)** injects lightweight trainable matrices into attention layers, reducing computational costs while maintaining performance.  
   - PEFT allows **targeted fine-tuning** of antibody models while preserving the base model’s pre-trained knowledge.  
   - **Repo:** [PEFT GitHub](https://github.com/huggingface/peft)  
   - With a set of baseline $(K_D$ values, one can combine pairwise predictions using a softmax-weighted average in the log scale, producing a final $(K_D)$ estimate.

---

## Benchmark Methods and Evaluation

1. **KD Metric Calculation**  
   - Rather than directly predicting absolute affinity, the model outputs a probability distribution over which sequence in a pair has higher affinity.  

where \(z_i\) are the network logits associated with each baseline.

2. **Performance Metrics**  
   - **Log-Scale Absolute Error (\(\Delta\log_{10} KD\))** compares the model’s predicted KD values to reference data.  
   - **Classification Accuracy** measures the binary correctness (higher vs. lower) in pairwise comparisons.

3. **Hyperparameter Settings**  
   - Typical dropout rates: **10–20%**  
   - Intermediate dense layers: **4 → 256 → 128 (optionally deeper)**  
   - Use of **softmax** at the final layer for probability distributions.

---

## Summary

By integrating **structural embeddings** with a **multimerized sequence representation** (heavy chain, light chain, antigen) and applying a **contrastive learning** approach, HyperBind2’s model head robustly classifies relative affinities. This combination of **rich embeddings**, **contrastive objectives**, and **carefully tuned feed-forward blocks** has proven effective for small-scale, high-value antibody design problems.

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
