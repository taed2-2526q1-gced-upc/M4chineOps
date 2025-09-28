---
language:
- en
library_name: pytorch
tags:
- computer-vision
- classification
- deepfake
- video
- xception
datasets:
- FaceForensics++
metrics:
- accuracy
- f1
- auc
base_model: Xception

model-index:
- name: Deepfake Detection
  results:
  - task:
      type: image-classification
      name: Video Classification (Real vs Deepfake)
    dataset:
      type: FaceForensics
      name: FaceForensics++
      config: original
      split: test
    metrics:
      - type: accuracy
        value: 0.85
        name: Test Accuracy
      - type: f1
        value: 0.83
        name: Test F1
      - type: auc
        value: 0.87
        name: Test AUC

---

# Model Card for Deepfake Detection

<!-- Provide a quick summary of what the model is/does. -->

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Xception (Extreme Inception) is a convolutional neural network architecture introduced by François Chollet in 2017.  
It is based on the idea of **depthwise separable convolutions**, which factorize a standard convolution into two steps:  
1. A **depthwise convolution** (applies a single filter per input channel).  
2. A **pointwise convolution** (1×1 convolution combining the outputs).  

This makes Xception more efficient and powerful than Inception, enabling better feature extraction while keeping the model lightweight.  

In this project, we use a pre-trained **Xception network (ImageNet weights)** as the base, and fine-tune the last layers for **binary classification (real vs fake videos)**.

- **Developed by:** M4chineOps (Maite Blasi, Maria Gestí, Martina Massana, Maria Sans)
- **Project:** TAED2_M4chineOps – Deepfake Recognition
- **Language(s):** Not applicable (visual model)  

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

- **Research and academic study** of deepfake detection.  
- **Benchmarking** deepfake detection performance on FaceForensics++.  
- **Educational purposes**, e.g. teaching about computer vision and media forensics.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Not suitable for **legal, forensic, or production deployment**.  
- Not reliable against **novel deepfake techniques not present in training data**.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- May fail on **low-resolution or heavily compressed** videos.  
- Limited to **FaceForensics++ manipulation techniques** (does not generalize perfectly to unseen methods).  
- Potential for **false positives** (flagging real videos as fake) and **false negatives** (missing actual deepfakes).  
- Dataset bias: trained on YouTube videos only, so performance may vary on other domains (TikTok, Instagram, etc.).  

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

- Complement with **multiple detection methods** (audio, metadata, multimodal).  
- Regularly retrain on updated datasets including **newer deepfake generation techniques**.  
- Use only in **controlled, experimental environments**.  

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

For more information look at: https://github.com/taed2-2526q1-gced-upc/M4chineOps/blob/main/docs/dataset_card.md (Dataset card)

#### Preprocessing

We implemented a custom preprocessing pipeline in Python to prepare the FaceForensics++ dataset before training.  

Steps:  
1. **Folder reorganization**: structured DVC-tracked files for original and manipulated sequences.  
2. **Metadata extraction**: used OpenCV to iterate through `.mp4` files and extract frame count, width, height. Organized metadata in dataframes (with fields filepath, label (0=real, 1=fake), frames, width, height) to be able to split data.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

Fine-tuning setup:  
- **Base model:** Xception with ImageNet weights.  
- **Frozen layers:** convolutional backbone initially frozen, then partially unfrozen for further training.  
- **Classifier head:** replaced with a dense layer + sigmoid for binary classification.  
- **Optimizer:** Adam.  
- **Loss:** Binary Cross-Entropy.  
- **Regularization:** dropout and early stopping to reduce overfitting.  
- **Metrics:** Accuracy, F1-score, AUC.  

This transfer learning approach allowed us to leverage pre-trained features and adapt them efficiently to the FaceForensics++ dataset.

#### Training Hyperparameters

- Batch size: 32/64
- Epochs: 20–30
- Learning rate: 1e-4 (with decay)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

For more information look at: https://github.com/taed2-2526q1-gced-upc/M4chineOps/blob/main/docs/dataset_card.md (Dataset card)

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

The model was evaluated using three main metrics: **Accuracy**, **F1-score**, and **AUC**. Each provides complementary information about performance, especially considering the challenges of class imbalance and the risks of false positives/negatives in deepfake detection.  

- **Accuracy**  
  Measures the overall proportion of correctly classified samples (both real and fake).  
  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]  
  While accuracy gives a quick sense of performance, it can be misleading in imbalanced datasets, where one class dominates.  

- **F1-score**  
  The harmonic mean of **precision** and **recall**.  
  \[
  \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]  
  This metric is especially important in deepfake detection because:  
  - High **precision** ensures few real videos are wrongly flagged as fake (avoiding false accusations).  
  - High **recall** ensures most fake videos are correctly detected (avoiding undetected misinformation).  
  A balanced F1-score indicates that the model handles both errors fairly well.  

- **AUC**  
  Evaluates the trade-off between **True Positive Rate (TPR)** and **False Positive Rate (FPR)** across different thresholds.  
  - AUC close to **1.0** means the model separates well between real and fake videos.  
  - AUC around **0.5** means performance is no better than random guessing.  
  This is a robust metric in binary classification tasks, particularly when the decision threshold may vary depending on application.  

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Card Authors

M4chineOps Team: Maite Blasi, Maria Gestí, Martina Massana, Maria Sans

## Model Card Contact

m4chineops@gmail.com
