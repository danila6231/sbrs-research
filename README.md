# SASRecFPlus: Feature-Rich Sequential Recommendation

## Overview
This project extends the base [SASRec](https://arxiv.org/abs/1808.09781) (Self-Attentive Sequential Recommendation) architecture to support user and item features. The main innovation is the ability to append user and feature embeddings to the beginning of the item sequence, allowing the model to leverage additional contextual information for improved recommendations.

- **Main model:** `models/sas_rec_f_plus.py`
- **Benchmark scripts:** `run.py`, `run_base.py`

---

## Model: `models/sas_rec_f_plus.py`

This file implements the `SASRecFPlus` model, which extends the original SASRec by:
- Allowing the use of user and item features (e.g., demographics, attributes).
- Concatenating user and feature embeddings to the input item sequence.
- Supporting flexible feature selection and pooling modes.

Main idea of the implementation:
- Concatenate the user embedding with the embedding of user features
- Process it with fully connected layer and append it the beginning of the item embedding sequence. This allows the transformer to pay attention to the user features if needed.

**Key components:**
- `UltimateFeatureSeqEmbLayer`: Embeds selected user/item features.
- `SASRecFPlus`: Main model class, supports both standard and feature-rich sequential recommendation.

---

## Running Benchmarks

### 1. `run.py`
- **Purpose:** Custom training and evaluation pipeline using RecBole's lower-level API.
- **How it works:**
  - Loads configuration and dataset.
  - Initializes the `SASRecFPlus` model.
  - Trains and evaluates the model, logging results.
- **Usage:**
  ```bash
  python run.py
  ```
- **Customization:**
  - Edit the `parameter_dict` in `run.py` to change data, features, or model hyperparameters.

### 2. `run_base.py`
- **Purpose:** Quick benchmarking using RecBole's `run_recbole` utility.
- **How it works:**
  - Loads configuration and dataset.
  - Runs the base SASRec model (or can be modified to use `SASRecFPlus`).
- **Usage:**
  ```bash
  python run_base.py
  ```
- **Customization:**
  - Edit the `parameter_dict` in `run_base.py` for different settings.

---

## Notes
- The project is built on [RecBole](https://recbole.io/), a unified recommendation library.
- To use your own dataset, place it in the `dataset/` directory and update the configuration accordingly.
- For feature-rich recommendation, ensure `selected_user_features` and/or `selected_item_features` are set in the config.

---
