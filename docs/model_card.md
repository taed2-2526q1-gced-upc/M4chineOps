---
language:
- en
library_name: pytorch
tags:
- computer-vision
- classification
- deepfake
- video
- transformer
- videomae
datasets:
- FaceForensics++
metrics:
- accuracy
- f1
- auc
base_model: VideoMAE

model-index:
- name: Deepfake Detection
  results:
  - task:
      type: video-classification
      name: Video Classification (Real vs Deepfake)
    dataset:
      type: FaceForensics
      name: FaceForensics++
      config: original
      split: test
    metrics:
      - type: accuracy
        value:
        name: Test Accuracy
      - type: f1
        value:
        name: Test F1
      - type: auc
        value:
        name: Test AUC

---

# Model Card for Deepfake Detection

## Model Details

### Model Description

The developed model is based on **VideoMAE**, a Vision Transformer architecture specifically adapted for **spatio-temporal video understanding**. Unlike traditional CNNs such as Xception, VideoMAE processes **video patches** (spatial + temporal), allowing it to jointly capture both visual appearance and motion dynamics.  

- **Backbone:** VideoMAE initialized with pre-trained weights from **Kinetics-400**, a large-scale human action dataset.  
- **Adaptation for binary classification:**  
  - The original classification head (400 classes) was replaced with a **fully connected layer with 2 outputs** (real vs fake).  
  - This setup leverages pre-trained spatio-temporal representations while fine-tuning for deepfake detection.  
- **Input representation:** Videos are divided into groups of **16 consecutive frames**, each resized to **224×224 pixels**, ensuring compatibility with the VideoMAE backbone and enabling temporal learning within short sequences.  
- **Training setup:**  
  - Optimizer: Adam  
  - Loss: Binary Cross-Entropy  
  - Regularization: dropout and early stopping  
  - Transfer learning strategy: backbone fine-tuning  

- **Developed by:** M4chineOps (Maite Blasi, Maria Gestí, Martina Massana, Maria Sans)  
- **Project:** TAED2_M4chineOps – Deepfake Recognition  
- **Language(s):** Not applicable (visual model)  

---

## Uses

### Direct Use

- **Research and academic study** of video-based deepfake detection.  
- **Benchmarking** deepfake detection performance on FaceForensics++.  
- **Educational purposes**, particularly for demonstrating spatio-temporal modeling with Vision Transformers.  

### Out-of-Scope Use

This model is **not recommended** for high-stakes or real-world forensic use cases, such as:  

- **Detection of novel or advanced deepfake methods** not included in FaceForensics++.  
- **Analysis of low-quality or highly compressed content**, which may degrade accuracy.  
- **Deployment on uncontrolled video sources** (TikTok, Instagram, messaging platforms) without retraining on relevant datasets.  

---

## Bias, Risks, and Limitations

- Performance is limited to **FaceForensics++ manipulation techniques** and may not generalize well to unseen methods.  
- **Dataset bias:** trained on YouTube-like videos → may struggle with other platforms.  
- **False positives:** risk of mislabeling real content as fake.  
- **False negatives:** risk of missing sophisticated or unseen deepfakes.  
- Sensitive to **resolution and compression artifacts**.  

### Recommendations

- Use only in **controlled experimental setups**, not as a forensic decision-making tool.  
- Retrain periodically with **newer datasets** reflecting emerging deepfake generation techniques.  
- Combine with **multimodal approaches** (audio, metadata, physiological cues) for higher reliability.  

---

## Training Details

### Training Data

Dataset: **FaceForensics++**  
- Contains original and manipulated YouTube-based videos.  
- Training/validation/testing splits applied.  

For more details: [Dataset Card](https://github.com/taed2-2526q1-gced-upc/M4chineOps/blob/main/docs/dataset_card.md)  

#### Preprocessing

- Videos split into **16-frame clips**.  
- Frames resized to **224×224 pixels**.  
- Organized into DVC-tracked folders with metadata (filepath, label, frames, resolution).  

### Training Procedure

- **Base model:** VideoMAE (pretrained on Kinetics-400).  
- **Classifier head:** replaced with 2-class dense layer.  
- **Frozen layers:** initial fine-tuning with frozen backbone, then gradual unfreezing.  
- **Optimizer:** Adam.  
- **Loss:** Binary Cross-Entropy.  
- **Regularization:** dropout + early stopping.  
- **Metrics:** Accuracy, F1-score, AUC.  

#### Training Hyperparameters

- Batch size: 32/64  
- Epochs: 20–30  
- Learning rate: 1e-4 with decay  

---

## Evaluation

### Testing Data

Dataset: **FaceForensics++ (test split)**  
More info: [Dataset Card](https://github.com/taed2-2526q1-gced-upc/M4chineOps/blob/main/docs/dataset_card.md)  

### Factors

- Video compression level.  
- Manipulation method (specific FaceForensics++ manipulation types).  
- Frame quality and resolution.  

### Metrics

Three main metrics are used to evaluate the model: **Accuracy**, **F1-score**, and **AUC**. Each provides complementary information about performance, especially considering the challenges of class imbalance and the risks of false positives/negatives in deepfake detection.  

- **Accuracy**  
  Measures the overall proportion of correctly classified samples (both real and fake).  
  
  Accuracy = (TP + TN) / (TP + TN + FP + FN)

  While accuracy gives a quick sense of performance, it can be misleading in imbalanced datasets, where one class dominates. 

- **F1-score**  
  The harmonic mean of **precision** and **recall**. 
   
  F1 = 2 * (Precision * Recall) / (Precision + Recall)

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

[More Information Needed]

#### Summary

The VideoMAE-based model shows strong potential for deepfake recognition by leveraging spatio-temporal features, but requires careful retraining and multimodal complementarity for practical deployment.  

---

## Model Card Authors

M4chineOps Team: Maite Blasi, Maria Gestí, Martina Massana, Maria Sans  

## Model Card Contact

m4chineops@gmail.com  
