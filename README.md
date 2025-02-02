# HyperBind (Open-Source Edition)

Welcome to HyperBind—an open-source deep learning framework for antibody design and binding affinity prediction, originally developed at EVQLV and adapted here for a broader research community. This repository contains a scaled-down version of our internal platform, focusing on demonstrating workflow concepts, data handling, and the integration of modern protein language models (e.g., ProtBert from ProtTrans5).

> **Note**: We’ve removed certain proprietary components—particularly our custom transformer and structure-based embeddings—while retaining a tangible end-to-end example of how antibody-antigen modeling can be approached.

## Table of Contents
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

## Overview

HyperBind is a deep learning pipeline aimed at antibody design and binding affinity prediction. It was written using TensorFlow Keras and leverages ProtBert (from the ProtTrans5 suite) to generate sequence embeddings. We demonstrate how to train a model on paired antibody heavy-light chain data, facilitating tasks like:
- Binder vs. Non-Binder Classification
- Preliminary Affinity Ranking

While this open-source version focuses on a scaled-down architecture, it retains essential modules for researchers to explore de novo antibody screening and basic antibody–antigen modeling.

## Features
- TensorFlow Keras-based pipeline with clear modular code.
- ProtBert Integration for generating high-quality sequence embeddings.
- Support for paired heavy-light chain inputs, reflecting real-world antibody structure.
- **Two Datasets Included:**
  - **Real World Training Data**: An anonymized set of complete heavy-light chain pairs across multiple targets (with minimal labels).
  - **Synthetic Dataset** from Greiff Lab’s Absolut! to experiment with or supplement training.
- Example scripts for data preprocessing, model training, and inference.

## What’s Inside

```
HyperBind/
|-- data/
|   |-- real_world/           # Paired heavy-light chain data (lightly anonymized)
|   |-- synthetic/            # Synthetic dataset from Absolut
|
|-- models/
|   |-- hyperbind_model.py    # Model architecture (TensorFlow Keras)
|   |-- prot_bert_embedder.py # Code integrating ProtBert embeddings
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

### Key components include:
- `hyperbind_model.py`: Defines the top-level Keras architecture.
- `prot_bert_embedder.py`: Contains utility functions to fetch and apply ProtBert for sequence embeddings.
- `train.py`: Runs a simplified training loop using a classification approach.
- `data_preprocess.py`: Illustrates how to load, tokenize, and embed the antibody sequences.

## Installation

1. Clone this repo:
    ```bash
    git clone https://github.com/evqlv/hyperbind.git
    cd hyperbind
    ```

2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    - The requirements file includes TensorFlow, Transformers (for ProtBert), and standard data libraries.

## Usage

### Data Preparation

1. **Download/Access the Provided Data:**
   - `data/real_world/`: Contains partial heavy-light chain pairs and minimal binding labels (e.g., 1 = binder, 0 = non-binder).
   - `data/synthetic/`: Synthetic dataset from Absolut, used for experimenting with or supplementing real data.

2. **(Optional) Preprocess:**  
Run the following command to process the dataset:
    ```bash
    python scripts/data_preprocess.py --data_dir data/real_world
    ```
    - Cleans/formats sequences
    - Tokenizes them using a standard dictionary
    - Saves them for ProtBert embedding.

### Training the Model

1. **Configure:**  
Adjust hyperparameters in `train.py` (batch size, epochs, learning rate) and set `--train_dir` to the desired dataset.

2. **Run Training:**
    ```bash
    python scripts/train.py --train_dir data/real_world --epochs 5 --batch_size 16
    ```

3. **Monitor:**  
Training logs and metrics (accuracy, loss) will be printed. You can add TensorBoard for in-depth visualization.

### Inference / Prediction

- Use `inference.py` with a saved model checkpoint:
    ```bash
    python scripts/inference.py --model_ckpt path/to/checkpoint --input_file data/synthetic/test_sequences.csv
    ```
- Outputs a probability score for each sequence pair indicating potential binding affinity.

## Limitations

1. **No Model Weights Included**: This repo does not provide our proprietary weights. You must train on the provided sample data or your own.
2. **Scaled-Down Architecture**: Our internal version integrates structure-based embeddings and advanced transformers not released here.
3. **Limited Dataset**: The real-world dataset is anonymized and reduced in scope; actual large-scale training data remains proprietary.

## Contributing

We welcome pull requests for bug fixes, documentation improvements, or general ML best practices. For feature requests related to the full HyperBind pipeline (including advanced structural embeddings), please open an issue—but note that some IP-protected components may not be open-sourced.

## License

Distributed under the Apache GNU 3.0 Affero open source license. See `LICENSE` for more information.

## Acknowledgments

- **ProtBert** from ProtTrans for sequence embeddings.
- **Greiff Lab’s Absolut dataset** for synthetic antibody sequences.
- **Our colleagues at EVQLV** for helpful discussions and partial dataset preparation.

## Disclaimer

This open-source edition is for research use only and is not intended for clinical or commercial deployment without further validation.

---

We hope **HyperBind (Open-Source Edition)** proves helpful in exploring how machine learning can accelerate antibody discovery! For questions or feedback, open an issue or email us at **info@evqlv.com**.
