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

### Tested Environments & Prerequisites

HyperBind2 (Open-Source Edition) was developed and tested on the following configurations:

- **Operating System**: Ubuntu 20.04 / 22.04 (Linux)
- **Python Version**: 3.9
- **CUDA Version**: 11.6
- **cuDNN Version**: 8.x

Other configurations (e.g., Windows 10/11 with a compatible NVIDIA GPU) may also work but are not officially tested.  
A GPU with at least **16GB of VRAM** is recommended for smaller-scale experiments, though larger memory (e.g., 24GB or more) will significantly improve training throughput. For CPU-only operation, you can run certain scripts (like data preprocessing), but training and inference will be very slow without GPU acceleration.

Make sure you install the dependencies listed in [`requirements.txt`](./requirements.txt) or create a dedicated Conda/virtual environment that matches the versions we specify. Having a compatible version of **PyTorch** (>= 1.12.0) with CUDA support is crucial for model training.


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

## Data Availability & Instructions

This repository references two primary datasets:

1. **Real-World Antibody Data**  
   - Anonymized heavy-light chain pairs stored under `data/real_world/`.
   - Minimal binding labels are included for demonstration purposes.
   - **Download Link**: [Placeholder Link]  
     *(Replace this with the actual URL or instructions. For instance, if you're hosting on Zenodo, insert the Zenodo link here. E.g., “Download from [Zenodo link].”)*

2. **Synthetic Dataset (Absolut! from Greiff Lab)**  
   - Synthetic antibody sequences stored under `data/synthetic/`.
   - **Download Link**: [Placeholder Link]  
     *(Again, replace with your actual link or repository reference.)*

**How to Use These Datasets**  
1. **Download** the dataset archives (or individual files) from the placeholder links.  
2. **Unzip/Extract** them into the `data/real_world/` and `data/synthetic/` directories, respectively.  
3. **Run** the data preprocessing script:
    ```bash
    python scripts/data_preprocess.py --data_dir data/real_world
    ```
   This will tokenize and format sequences for model training.

If you have your own data, simply place the FASTA or CSV files into a new directory (e.g., `data/custom/`) and run the same preprocessing command with the appropriate path.


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


## Fine-Tuning Train with with LoRA (PEFT)

HyperBind2 leverages [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning of large models like ESM3. Specifically, we use **LoRA** (Low-Rank Adaptation) to train small rank-decomposition matrices while keeping the rest of ESM3’s weights frozen.

### Why LoRA?
- **Reduced Memory Footprint**: Only a fraction of the total parameters are updated.
- **Faster Training**: Training is significantly quicker compared to full fine-tuning of 1.4B+ parameters.
- **Modular**: It’s easy to swap out with other PEFT strategies (e.g., Prefix Tuning, Adapter methods) if desired.

### High-Level Setup
1. **Load Pre-Trained ESM3**:  
   We first load the base ESM3 (Open-Source Edition: esm3-open-2024-03).  
2. **Apply LoRA Configuration**:
  \
  You can tweak LoRA hyperparameters (rank, alpha, etc.) in the config file or within the script. 

   ```python
   from peft import LoraConfig, get_peft_model

   lora_config = LoraConfig(
       r=8,                 # rank
       lora_alpha=16,       # scaling factor
       lora_dropout=0.1,    # dropout on the LoRA layers
       target_modules=["..."]  # define which layers to inject LoRA
   )

   # "base_model" is the loaded ESM3 model
   peft_model = get_peft_model(base_model, lora_config)

3. Run inference with LoRA as follows:
    ```
    python scripts/train.py \
      --train_dir data/real_world \
      --epochs 5 \
      --batch_size 16 \
      --use_peft lora \
      --peft_config configs/lora_config.json
    ```
> **Note**: Adjust the exact LoRA arguments, file paths, or script references based on your project’s structure.


### 🔍 Inference / Prediction  

Run predictions on new sequences:
```bash
python scripts/inference.py --model_ckpt path/to/checkpoint --input_file data/synthetic/test_sequences.csv
```
Outputs a **binding probability score** per sequence pair.

---

## ⚠️ Limitations

1. **No Proprietary Model Weights**  
   - This repository does **not include** our **internal commercial model weights** for the contrastive head model.
   - Users must train the contrastive head on provided sample data.
   - *Note we **do** provide encoder weights based on finetuned ESM3 Open Source, finetuned for antibody sequence and antibody structure.*

2. **Scaled-Down Architecture**  
   - **Our internal model integrates advanced antibody-specific transformers** & **structure-based embeddings** not included in this release.

3. **Dataset Limitations**  
   - The **real-world dataset is anonymized** and **reduced in scope** compared to internal training data.

---
## Hardware & Performance Notes

- **Training**:
  - Our internal benchmarks were conducted on **8× NVIDIA A100 80GB GPUs**, allowing us to fine-tune the ESM3-based model in parallel with ample memory.
  - On smaller setups (e.g., a single GPU with 16GB VRAM), training is still possible but will be slower. You may need to reduce batch sizes or sequence lengths.
  - Expect training times for our sample dataset to vary from a few hours to a day, depending on batch size, sequence length, and GPU capability.

- **Inference**:
  - We demonstrated inference on **2× NVIDIA V100 (16GB)** GPUs for smaller datasets. Inference can be done on a single GPU if the batch size is kept low.
  - Typical inference times on a single V100 for a batch of ~100 sequences range from several seconds to a minute, depending on model settings.

- **Memory Considerations**:
  - ESM3 (1.4B parameters) is large; using **LoRA** or other PEFT methods is essential for reducing the memory footprint during training.
  - For best results on large-scale datasets, multi-GPU setups are recommended.

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

![Contrastive Loss](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{contrast}}(\theta)%20=%20\sum_{(i,j)\in%20P}%20\max%20\Big(0,%20m%20-%20\big(f_{\theta}(x_i)%20-%20f_{\theta}(x_j)\big)\Big))

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

Distributed under the **GPL 3.0 License**.  
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
