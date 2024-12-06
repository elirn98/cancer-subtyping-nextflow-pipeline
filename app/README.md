# cancer-subtyping-nextflow-pipeline

# Ovarian Cancer Subtype Classification Nextflow Pipeline

This repository contains a pipeline for ovarian cancer subtype classification using histopathological image patches. The project employs deep neural networks (DNNs) to classify ovarian cancer into its five primary subtypes: Clear Cell Carcinoma (CC), Endometrioid Carcinoma (EC), High-Grade Serous Carcinoma (HGSC), Low-Grade Serous Carcinoma (LGSC), and Mucinous Carcinoma (MC).

---

## Background and Rationale

Ovarian cancer is a significant contributor to gynecological cancer mortality. Accurate subtype classification is essential for effective diagnosis and treatment strategies. This project leverages deep learning techniques to train a DNN model for ovarian cancer subtype classification, using histopathological image patches. This approach improves diagnostic precision and reproducibility.

---

## Dependencies

The following packages are required to execute the pipeline:

- `torch`
- `torchvision`
- `tqdm`
- `numpy`
- `timm`
- `Pillow`
- `scipy`
- `pandas`
- `h5py`
- `scikit-learn`

---

## Usage

### Installation

1. Install Nextflow:
   ```bash
   curl -s https://get.nextflow.io | bash \
   && chmod +x nextflow \
   && mv nextflow /usr/local/bin/
Navigate to the application directory:
bash
Copy code
cd /app
Running the Pipeline
Run the pipeline using Nextflow with Docker:

bash
Copy code
nextflow run pipeline.nf -with-docker cancer-subtyping:v1.0
Dataset Preparation
Ensure the dataset is organized in the following structure:

javascript
Copy code
data/<dataset_name>/patches/<patch_size>/Mix/<subtype>/<slide_name>/<patch_size>/<magnification>/<patch_name>
Example:

bash
Copy code
data/MKobel/patches/1024/Mix/CC/112962/1024/20
Training Data: 1 slide per subtype, totaling 1,000 patches (200 per subtype).
Test Data: 1 slide per subtype, totaling 200 patches (40 per subtype).
Input
Training Phase
MODEL: Name of the model architecture to be used.
params.SAVE_PATH: Path to save model checkpoints and training statistics (accuracy and loss).
params.DATA_PATH: Path to the training dataset.
Testing Phase
hoptimus0_model: Path to the trained model checkpoint from the training phase.
params.SAVE_PATH: Path to save testing results.
params.DATA_PATH: Path to the testing dataset.
Visualization Phase
model_statistics: Path to the training statistics file (accuracy and loss).
test_results: Path to the testing results file (accuracy and loss).
params.PLOT_PATH: Path to save generated plots.
Output
Training Phase
hoptimus0_model: Trained model checkpoint.
model_statistics: Training accuracy and loss metrics.
Testing Phase
test_results: Testing accuracy and loss metrics.
Visualization Phase
Plots saved in the directory specified by params.PLOT_PATH:
Accuracy vs. Epoch plot.
Loss vs. Epoch plot.
ROC Curve.
Expected Results
Comprehensive Training Results (Full Dataset)
Dataset: 5 slides for training (1,000 patches) and 1 slide for testing (200 patches).
Training Epochs: 3 epochs.
Expected Output:
A trained model with reasonable accuracy for subtype classification.
Training and testing accuracy/loss statistics.
Visualizations:
Accuracy vs. Epoch plot.
Loss vs. Epoch plot.
ROC Curve.
Reproducibility and Quick TA Test Results (Reduced Dataset)
Dataset: 1 slide with 100 patches for training and 1 slide with 100 patches for testing.
Training Epochs: 1 epoch.
Expected Output:
A model trained in minimal time for reproducibility and testing purposes.
Accuracy and loss statistics for training and testing phases.
Quick visualizations:
Accuracy vs. Epoch and Loss vs. Epoch plots.
Simplified ROC Curve.
