# Longitudinal Ordinal Risk Model for Lung Cancer Survival

This project implements a **Longitudinal Transformer** with a **Consistent Rank Logits (CORAL)** ordinal head to predict the risk of lung cancer over a 6-year horizon.

## 🧠 Architecture: The Ordinal Head (CORAL)

Standard binary classification treats "Cancer in Year 1" and "Cancer in Year 6" as equally "positive," losing critical severity information. Regression implies a linear distance between years.

We use the **CORAL (Consistent Rank Logits)** framework to treat time-to-cancer as an **ordinal regression problem**.

### How it Works
The model predicts a single latent "risk score" for a patient, but compares it against **$K-1$ learnable thresholds** (bias terms), where $K$ is the number of time bins (e.g., Healthy, Year 6, ..., Year 1).

1.  **Latent Score:** The Transformer outputs a feature vector $h$. A dense layer projects this to a scalar score:
    $$s = W^T h$$

2.  **Cumulative Probabilities:** The probability that the true label $y$ exceeds rank $k$ is modeled as:
    $$P(y > k) = \sigma(s + b_k)$$
    where $\sigma$ is the sigmoid function and $b_k$ are the learnable thresholds.

3.  **Consistency:** Because $b_k$ are fixed bias terms for the same shared weight $W$, the probabilities are guaranteed to be monotonic:
    $$P(y > \text{Year 1}) \le P(y > \text{Year 2}) \le \dots \le P(y > \text{Healthy})$$

4.  **Class Probabilities:** The probability for a specific year is derived by subtraction:
    $$P(y = k) = P(y > k-1) - P(y > k)$$

This ensures that a high risk for "Year 1" implies a high risk for all subsequent years, enforcing logical consistency in the risk predictions.

## 📊 Test Set Results (Patch-8 Features)

The model was evaluated on a Test set of **N=2,203** subjects. Patch 8.

| Horizon (Years) | AUC | AUPRC |
| :--- | :--- | :--- |
| **Year 1** | **0.9692** | **0.8350** |
| **Year 2** | **0.9499** | **0.7759** |
| **Year 3** | **0.9135** | **0.6960** |
| **Year 4** | **0.8866** | **0.6341** |
| **Year 5** | **0.8888** | **0.6284** |
| **Year 6** | **0.8888** | **0.6284** |
| **AVERAGE** | **0.9161** | **0.6996** |

**Global Calibration Metric:**
* **Concordance Index (C-Index):** `0.8873`
