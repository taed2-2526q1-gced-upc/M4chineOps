---
language:
- en
library_name: tensorflow
tags:
- computer-vision
- classification
- deepfake
- video
- cnn
- xception
datasets:
- FaceForensics++
metrics:
- accuracy
- f1
- auc
base_model: Xception

model-index:
- name: Deepfake Detection (Xception + Logistic Regression)
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

# Model Card for Deepfake Detection (Xception + Logistic Regression)

## Model Details

### Model Description

The developed model is a **hybrid deepfake detection system** that combines **Xception** for visual feature extraction with a **Logistic Regression classifier** for final **frame-level prediction**.  
Instead of aggregating embeddings to represent each video globally, the model generates **one embedding per detected face frame** (10 frames per video). Each frame embedding is treated as an independent sample for training and evaluation.

- **Backbone:** Xception pretrained on **ImageNet**, used as a frozen feature extractor (`include_top=False`) with **GlobalAveragePooling2D**.  
- **Embeddings per frame:** 2048-dimensional vector representing each detected face.  
- **Frame sampling:** 10 frames per video, uniformly distributed.  
- **Classifier:** Scikit-learn **Logistic Regression**, trained on frame-level embeddings using **GridSearchCV** for hyperparameter tuning (penalty, solver, C, l1_ratio).  
- **Input representation:** Frames resized to 299×299 and normalized with Xception preprocessing.  
- **Scaling:** StandardScaler fitted on training and validation embeddings.  
- **Tracking and reproducibility:** MLflow for experiment logging; CodeCarbon for emission tracking.  

- **Developed by:** M4chineOps (Maite Blasi, Maria Gestí, Martina Massana, Maria Sans)  
- **Project:** TAED2_M4chineOps – Deepfake Recognition  
- **Language(s):** Not applicable (visual model)  

---

## Uses

### Direct Use

- **Research** on interpretable and efficient deepfake detection pipelines at the frame level.  
- **Benchmarking** hybrid architectures combining CNN embeddings with classical ML classifiers.  
- **Educational purposes** to demonstrate transferable CNN features in deepfake detection.  

---

## Bias, Risks, and Limitations

- **Dataset bias:** trained solely on FaceForensics++ (YouTube-style videos), may not generalize to other platforms or compression levels.  
- **Loss of temporal information:** as each frame is classified independently, the model does not capture motion or consistency across frames.  
- **Face detection dependency:** relies on MTCNN; missed or incorrect detections reduce model robustness.  
- **Potential demographic bias:** model performance may vary across age, ethnicity, and gender groups.  
- **Risk of false positives/negatives:** typical in deepfake detection; requires threshold tuning for reliable operation.  

### Recommendations

- Retrain or fine-tune the Logistic Regression classifier with domain-specific data before deployment.  
- Combine with temporal, audio, or metadata-based features for improved robustness.  
- Use human review for any critical decision; do not rely solely on model predictions.  

---

## Training Details

### Training Data

Dataset: **FaceForensics++ (original)**  
- Contains both authentic and manipulated videos.  
- Balanced sampling ensured by internal preprocessing scripts (`data_sampling_and_metadata.py`).  

For more details: [Dataset Card](https://github.com/taed2-2526q1-gced-upc/M4chineOps/blob/main/docs/dataset_card.md)  

#### Preprocessing

- **Face extraction:** using **MTCNN** (`extract_face_frames()` and `extract_and_save_face_paths()`).  
- **Frame sampling:** uniform selection of 10 frames per video.  
- **Image size:** resized to 299×299 pixels for Xception input.  
- **Embedding generation:** Xception + `GlobalAveragePooling2D`, producing a 2048-dimensional feature vector per frame.  

### Training Procedure

- **Base model:** Xception (`weights='imagenet'`, `include_top=False`).  
- **Feature dimension:** 2048 per frame.  
- **Classifier:** Logistic Regression with **GridSearchCV** (`scoring='roc_auc'`).  
- **Scaler:** StandardScaler fitted on train+val embeddings.  
- **Metrics:** Accuracy, F1-score, ROC-AUC.  
- **Tracking:** MLflow for experiment logging, CodeCarbon for emission tracking.  

#### Training Hyperparameters

- Penalty: `['l2', 'elasticnet']`  
- Solvers: `['saga', 'lbfgs']`  
- C: `[0.01, 0.1, 1, 10]`  
- l1_ratio: `[0.3, 0.5, 0.7]` (for elasticnet)  
- Max iterations: `3000`  
- Frames per video: `10`  

---

## Evaluation

### Testing Data

Dataset: **FaceForensics++ (test split)**  
For further details: [Dataset Card](https://github.com/taed2-2526q1-gced-upc/M4chineOps/blob/main/docs/dataset_card.md)  

### Factors

- Compression level and manipulation type.  
- Accuracy of detected face crops.  
- Frame sampling variability.  

### Metrics

Three main metrics are used to evaluate the model: **Accuracy**, **F1-score**, and **AUC**. Each provides complementary information about performance, especially considering class imbalance and the potential consequences of misclassification.  

- **Accuracy**  
  Measures the overall proportion of correctly classified frames (both real and fake).  
  
  Accuracy = (TP + TN) / (TP + TN + FP + FN)

  While accuracy provides a general sense of model performance, it may not fully capture performance in imbalanced scenarios.  

- **F1-score**  
  The harmonic mean of **precision** and **recall**. 
   
  F1 = 2 * (Precision * Recall) / (Precision + Recall)

  This metric is critical for deepfake detection because:  
  - High **precision** reduces false accusations (real videos wrongly flagged as fake).  
  - High **recall** ensures detection of most fake frames (avoiding undetected manipulations).  
  A balanced F1-score indicates good performance on both dimensions.  

- **AUC**  
  Measures the trade-off between **True Positive Rate (TPR)** and **False Positive Rate (FPR)** across thresholds.  
  - AUC near **1.0** implies good separation between real and fake frames.  
  - AUC near **0.5** implies performance close to random guessing.  
  This metric is particularly valuable when decision thresholds may vary depending on deployment context.  

### Results

Results are tracked and visualized via MLflow.  
All model metrics, hyperparameters, and artifacts are logged automatically, ensuring full reproducibility and traceability of the experiments.

---

## Summary

The **Xception + Logistic Regression** model provides a **lightweight, interpretable**, and **efficient** solution for deepfake detection.  
It achieves competitive results on FaceForensics++ using only frame-level information.  
However, due to the absence of temporal modeling, it should be retrained or fine-tuned before being used in real-world or cross-dataset scenarios.

---

## Model Card Authors

**M4chineOps Team:** Maite Blasi, Maria Gestí, Martina Massana, Maria Sans  

## Model Card Contact

📩 **m4chineops@gmail.com**
