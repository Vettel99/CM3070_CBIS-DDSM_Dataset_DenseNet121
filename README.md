# CM3070: Breast Cancer Detection on Full Screening Mammography

A reproducible, state-of-the-art transfer learning pipeline for benign vs. malignant classification on the Curated Breast Imaging Subset of DDSM (CBIS-DDSM).

## Project Overview
This project provides a robust clinical decision support tool designed to assist radiologists in identifying malignant tissue. By bypassing standard architectures in favor of a two-stage DenseNet121 transfer learning model, the system preserves microscopic indicators deep into the network. 

## Key Architectural Decisions
* **DenseNet121 Transfer Learning:** Employs a strict two-stage training methodology with batch normalization protection to prevent catastrophic forgetting during targeted unfreezing.
* **Fused CLAHE & Artifact Masking:** Enhances medical-grade tissue contrast and isolates the breast using OpenCV to eliminate "Clever Hans" shortcut learning from scanner artifacts.
* **Spatial Attention & Dual Pooling:** Focuses on dense tissue anomalies while preserving background context by concatenating Global Average and Global Max Pooling heads.
* **Gradient-Free Explainability:** Integrates Score-CAM heatmaps for robust visual diagnostic verification, ensuring the model's predictions are clinically interpretable.

## Dataset Handling
To guarantee absolute scientific reproducibility, this project utilizes the public CBIS-DDSM dataset. The codebase automatically fetches the required files using the `kagglehub` API. A strict patient-level splitting protocol is enforced prior to training to prevent data leakage between the training (80%), validation (10%), and testing (10%) subsets.

## Installation and Execution
This pipeline is optimized for Google Colab utilizing an L4 GPU due to the 512x512 high-resolution image processing requirements.

1. Clone this repository.
2. Upload the `CM3070_Feature_Prototype.ipynb` file to Google Colab.
3. Navigate to **Runtime > Change runtime type** and select **L4 GPU**.
4. Run all cells. The script will automatically install dependencies, download the dataset, and commence the two-stage training loop.

## Evaluation Metrics
The model is evaluated using domain-specific clinical metrics rather than simple accuracy, accounting for the complexities of oncological diagnostics.
* Area Under the Receiver Operating Characteristic Curve (AUC-ROC)
* Precision and Recall Classification Reports
* Visual Diagnostic Confusion Matrices
