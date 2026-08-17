# Transfer Learning - Steel Surface Defect Classification

## Overview

This project investigates the use of transfer learning for automated classification of steel surface defects.

The project uses an ImageNet-pretrained ResNet50V2 model and adapts it to the NEU Surface Defect Dataset. The goal is to classify steel-surface images into six different defect categories.

The project compares two approaches:

1. **Stage 1:** ResNet50V2 with the pretrained base layers frozen.
2. **Stage 2:** Fine-tuning of later layers of the pretrained model.

The purpose is to investigate whether adapting the pretrained visual features to the steel-defect domain improves classification performance.

## Dataset

The project uses the **NEU Surface Defect Dataset**, containing 1,800 grayscale images across six defect categories:

- Crazing
- Inclusion
- Patches
- Pitted surface
- Rolled-in scale
- Scratches

The dataset contains 300 images per class.

The images are originally 200 × 200 pixels. During preprocessing, they are resized to 224 × 224 pixels and converted to three channels so they can be used with ResNet50V2.

The data is divided into:

- **1,296 training images**
- **144 development validation images**
- **360 final evaluation images**

The final evaluation set is kept separate during model development and is only used for the final evaluation.

## Transfer Learning

Transfer learning is used by starting with a ResNet50V2 model that has already been trained on ImageNet.

The pretrained model contains visual features learned from a large general image dataset. These pretrained weights are transferred to the steel-defect classification task.

The original ImageNet classification layer is replaced with a new classification head containing six output classes.

### Stage 1 – Frozen Base

The pretrained ResNet50V2 layers are frozen so their weights are not updated.

Only the new classification head is trained on the steel-defect dataset.

### Stage 2 – Fine-Tuning

In the second stage, later layers of ResNet50V2 are unfrozen and fine-tuned using the steel-defect training data.

This allows some of the pretrained visual features to adapt to the specific characteristics of steel surface defects.

## Technologies

- Python
- TensorFlow / Keras
- ResNet50V2
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook / VS Code

## Model Evaluation

The models are evaluated using:

- Accuracy
- Loss
- Precision
- Recall
- F1-score
- Confusion matrix

The same final evaluation set of 360 images is used to compare Stage 1 and Stage 2.

## Results

### Stage 1 – Frozen ResNet50V2

- Final accuracy: **97.22%**
- Final loss: **0.1027**

### Stage 2 – Fine-Tuned ResNet50V2

- Final accuracy: **97.78%**
- Final loss: **0.0596**

Fine-tuning improved the final evaluation accuracy by **0.56 percentage points** and substantially reduced the evaluation loss.

The improvement was not identical across all defect classes. Pitted surface showed one of the largest improvements, while inclusion remained the most difficult class in terms of recall.

## Limitations

The dataset consists of 2D images of defects occurring on a three-dimensional steel surface. Information about depth and three-dimensional structure is therefore not fully available to the model.

Some visually similar defects may consequently be difficult to distinguish from a single 2D image.

Further improvements could include additional training data, image augmentation, higher-resolution imaging, or methods that provide more information about the three-dimensional structure of the defects.

## Project Structure

```text
09_TransferLearning/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── resnet50v2_frozen.keras
│   └── resnet50v2_finetuned.keras
│
├── notebooks/
│   └── transfer_learning.ipynb
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── .gitignore
├── README.md
└── requirements.txt
```
