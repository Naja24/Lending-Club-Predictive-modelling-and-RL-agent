# Policy Optimization for Financial Decision-Making

This project implements and compares two machine learning frameworks to optimize loan approval decisions from the [LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

The core of the project is a direct comparison between:
1.  **Model 1 (Supervised Learning):** A Deep Learning (MLP) model trained to **predict the probability of loan default**.
2.  **Model 2 (Offline RL):** An Offline Reinforcement Learning agent (Discrete-CQL) trained to **learn an optimal loan approval policy** that directly maximizes financial return.

The analysis demonstrates the difference between simply *predicting risk* and *optimizing for a direct business metric* (profit/loss).

## Table of Contents
- [Project Objective](#project-objective)
- [Methodology](#methodology)
- [How to Run](#how-to-run)
- [Summary of Results](#summary-of-results)
- [Required Libraries](#required-libraries)

## Project Objective

The goal is to develop an intelligent system for a fintech company to decide whether to approve or deny new loan applications. The system should be optimized to **maximize the company's financial return**, not just to identify risky applicants.

## Methodology

This project is broken into three main parts, contained in separate notebooks.

### 1. EDA and Preprocessing
-   **Data Loaded:** `accepted_2007_to_2018.csv.gz`
-   **EDA:** Analyzed the distribution of `loan_status` to define the target variable.
-   **Feature Engineering:**
    -   **Target:** `is_default` (1 for "Charged Off" / "Default", 0 for "Fully Paid").
    -   **Reward:** Engineered a reward signal for the RL agent based on the project brief:
        -   `Deny:` reward = 0
        -   `Approve (Paid):` reward = `loan_amnt * int_rate`
        -   `Approve (Default):` reward = `-loan_amnt`
-   **Preprocessing:** Selected 13 key features, handled missing values, one-hot encoded categorical variables, and standardized numeric features using an `sklearn` pipeline.
-   **Output:** `processed_lending_club.parquet`

### 2. Model 1: The Predictive Deep Learning Model
-   **Model:** A Multi-Layer Perceptron (MLP) built in PyTorch.
-   **Training:** Included a validation set, Batch Normalization, Dropout, Early Stopping, and Learning Rate Scheduling for a robust training harness.
-   **Target:** `is_default` (binary classification).
-   **Key Metrics:** AUC (Area Under the ROC Curve) and F1-Score.

### 3. Model 2: The Offline Reinforcement Learning Agent
-   **Framework:** Offline Reinforcement Learning (Batch RL).
-   **Algorithm:** `DiscreteCQL` (Conservative Q-Learning) from the `d3rlpy` library, which is designed for discrete action spaces and offline datasets.
-   **RL Definition:**
    -   **State (s):** The vector of preprocessed features for an applicant.
    -   **Action (a):** {0: Deny Loan, 1: Approve Loan}.
    -   **Reward (r):** The engineered financial return (see above).
-   **Evaluation:** Used `DiscreteFQE` (Fitted Q Evaluation) to estimate the average financial return of the new policy on an unseen test set.
-   **Key Metric:** Estimated Policy Value (in $).

## How to Run

This project is designed to run in a Kaggle Notebook environment.

1.  **Clone the Repository (Optional):**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/[YOUR_REPO_NAME].git
    ```

2.  **Set Up the Kaggle Environment:**
    -   Create a new Kaggle Notebook.
    -   In the notebook editor, click **"+ Add Data"** in the right-hand sidebar.
    -   Search for and add the **LendingClub Loan Data** by `wordsforthewise`.
    -   Upload the notebooks from this repository (`.ipynb` files) to your Kaggle notebook.

3.  **Run the Notebooks in Order:**
    You must run the notebooks sequentially, as they produce files used by the next notebook.
    1.  `1-EDA-and-Preprocessing.ipynb` (Produces `processed_lending_club.parquet`)
    2.  `2-DL-Model-Training.ipynb` (Produces the analysis for Model 1)
    3.  `3-RL-Agent-Training.ipynb` (Produces the analysis for Model 2)

## Summary of Results

This project successfully trained two distinct models and found that:

-   **Model 1 (DL)** achieved an **AUC of 0.716** and an **F1-Score of 0.43**. This demonstrates it is a capable classifier, significantly better than chance at *ranking* risky applicants.
-   **Model 2 (RL)** learned a policy with an **Estimated Policy Value of $962.9542** (fill in from your result).
-   This compares to the bank's **Historical Policy Value of -$1372.25**, which represents the average loss per loan in the dataset.
-   The RL agent's policy demonstrates a significant financial improvement, validating the approach of optimizing for business metrics directly.

## Required Libraries

A `requirements.txt` file is included. The key libraries are:

pandas

numpy

scikit-learn

torch

matplotlib

seaborn

d3rlpy

kaggle

pyarrow

You can install them by running:

pip install -r requirements.txt
