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

## 🔍 Benchmark Results

### Dataset: **Ta-Feng**
| Model                                | Max Seq Len | NDCG@10 | MRR@10 | Hit@10 |
|-------------------------------------|-------------|---------|--------|--------|
| Base (SASRec)                       | 50          | 0.5098  | 0.4442 | 0.7203 |
| + User Features (age_group, pin_code) | 50          | 0.4854  | 0.4189 | 0.6992 |
| Base (SASRec)                       | 5           | 0.4982  | 0.4335 | 0.7000 |
| + User Features (age_group, pin_code) | 5           | 0.4792  | 0.4133 | 0.6910 |

---

### Dataset: **ML-1M**
| Model                                | Max Seq Len | NDCG@10 | MRR@10 | Hit@10 |
|-------------------------------------|-------------|---------|--------|--------|
| Base (SASRec)                       | 50          | 0.6114  | 0.5484 | 0.8096 |
| + User Features (gender, occupation) | 50          | 0.5569  | 0.4866 | 0.7790 |
| Base (SASRec)                       | 5           | 0.5898  | 0.5265 | 0.7896 |
| + User Features (gender, occupation) | 5           | 0.5473  | 0.4741 | 0.7793 |

## Comments

### Possible Reasons for No Improvement with User Features

- **Weak user features**: Attributes like gender, occupation, age group, and pin code are often too coarse or generic to add meaningful signal.
- **User token under-attended**: The user embedding is prepended as a single token to the sequence and may get ignored by the attention mechanism, especially with longer item histories.
- **Simple feature fusion**: Concatenating user features with ID and passing through a single linear layer may not be expressive enough to capture useful interactions.
- **Redundant information**: SASRec already models user preferences through their item sequences, so adding explicit user info might introduce noise or overlap.
- **Insufficient tuning**: Adding user features increases model complexity, which may require re-tuning of hyperparameters and stronger regularization to avoid overfitting.

---
