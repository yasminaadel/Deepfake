# Fake Image Detection using Transfer Learning

Detect real vs. fake images using five state-of-the-art pretrained CNN architectures, fine-tuned and optimized for high classification performance. The project also compares results with a Vision Transformer baseline.

## Project Overview

This project addresses the challenge of classifying *real and fake images* using a dataset of *10,826 images* evenly distributed between the two classes. We leverage *transfer learning* and *Optuna*-based hyperparameter tuning to evaluate five popular CNN architectures:

* VGG16
* ResNet50
* DenseNet121
* Inception
* MobileNet

We apply 3-fold cross-validation for reliable evaluation and test a *Vision Transformer (ViT)* as an additional benchmark.

## Objectives

* Build a robust real vs. fake image classifier.
* Compare different pretrained CNNs using transfer learning.
* Tune learning rates with *Optuna* and the *Adam* optimizer.
* Fine-tune last convolutional layers and add custom layers for improved performance.
* Apply 3-fold cross-validation for performance stability.
* Explore *Vision Transformer* as a comparison model.

## Dataset

* *Total images*: 10,826
* *Classes*: Real, Fake
* *Split*: 3-fold Cross-Validation (Training, Validation, Test)

## Techniques & Tools

* *Frameworks*: PyTorch, Torchvision
* *Hyperparameter Optimization*: Optuna (Learning Rate Search in \[1e-5, 1e-1])
* *Optimizer*: Adam
* *Image Preprocessing*:
  * Resize, Flip, Crop, Rotate, Normalize, Color Jitter, Affine, Random Erasing
  * Resize: 224x224 for most models, 299x299 for Inception
* *Fine-Tuning*:
  * Last 2 convolutional layers
  * Additional Conv layer (DenseNet121)

## Results

| Model                      | Accuracy | F1-Score |
| -------------------------- | -------- | -------- |
| *DenseNet121 (Enhanced)* | *92%*  | *0.92* |
| ResNet50                   | \~89%    | \~0.89   |
| Inception                  | \~88%    | \~0.88   |
| MobileNet                  | \~87%    | \~0.87   |
| VGG16                      | \~86%    | \~0.86   |
| Vision Transformer         | \~91%    | \~0.91   |

*Best Model*: DenseNet121 + additional convolutional layer (92% accuracy)

## Evaluation Strategy

* *3-Fold Cross-Validation* for all models
* Metrics collected:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix

## Insights

* Fine-tuning improved all models significantly.
* DenseNet121 benefited most from additional convolutional layers.
* Vision Transformer was competitive but slightly behind DenseNet121.
