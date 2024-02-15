# Medical Knowledge-Enabled Multi-Task Learning for Gastric Cancer Survival Prediction

This repository contains the implementation of the research paper presented at the International Symposium on Biomedical Imaging (ISBI) in 2024. The paper is authored by Degan Hao, Qiong Li, Yudong Zhang, and Shandong Wu. This work introduces an innovative approach for predicting gastric cancer survival using multi-task learning and 3D medical imaging.

## Prerequisites
Before running the experiment, ensure you have the following prerequisites installed:
- Python 3.6 or higher
- Required Python libraries: [scikit-survival, NumPy, PyTorch, and Nibabel]

## Data preparation
You can either request data from the author or load your own data. The data includes medical imaging data, clinical data (i.e., pathological staging data in this work), and survival data (patient follow-up data). All data are linked by their patient IDs.  
## Running the Experiment
To replicate the experiment as described in the paper, run the following command:
```
python train_3DMTCNNpathology.py
```
