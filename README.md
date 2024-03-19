# Pneumonia X-Ray Detection using CNN

This project utilizes Convolutional Neural Networks (CNNs) implemented with Keras to detect pneumonia in chest X-ray images. It leverages the dataset available on Kaggle, provided by [Paul Mooney](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Overview

Pneumonia is a serious respiratory infection affecting the lungs, and its diagnosis via chest X-ray interpretation can be challenging. This project aims to automate this process using deep learning techniques, assisting medical professionals in accurate and timely diagnosis.

## Dataset

The dataset consists of chest X-ray images collected from various sources, including normal cases and cases with bacterial or viral pneumonia. You can access the dataset on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Dependencies
Install this libraries
```bash
pip install Keras matplotlib scikit-learn numpy
```
These libraries were utilized for building and evaluating the CNN model, as well as for data visualization and preprocessing.

## Model Architecture

The CNN architecture comprises convolutional layers, max-pooling layers, and fully connected layers, implemented using Keras. The final layer utilizes a softmax activation function for multi-class classification.

## Training and Evaluation

The model was trained on the provided dataset, employing techniques such as data augmentation to enhance its performance and generalization. Evaluation metrics including accuracy, precision, recall, and F1-score were computed using scikit-learn to assess the model's performance.

## Results

The trained CNN model achieved a high accuracy in detecting pneumonia from X-ray images. Further details on performance metrics and results can be found in the project's code and documentation.

## Usage

To utilize the trained model for pneumonia detection:

1. Clone this repository:

```bash
git clone https://github.com/aka964/Pneumonia-X-Ray-Detection-using-CNN.git
