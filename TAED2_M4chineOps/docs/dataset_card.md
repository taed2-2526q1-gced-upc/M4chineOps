# Dataset Card: FaceForensics++ (adapted for TAED2_M4chineOps)

## Identification
- **Dataset name:** FaceForensics++  
- **Original authors:** Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner  
- **Integrated in project:** TAED2_M4chineOps – Deepfake Recognition  
- **Original repository:** [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)  
- **Paper:** [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)  
- **Version used:** ------- 
- **Date integrated in project:** 2025-09-18  

---

## Motivation
The FaceForensics++ dataset was created to detect and analyze manipulated human faces.  
In this project, we use it to train, validate, and evaluate models for deepfake detection within an MLOps workflow.

---

## Composition
- **Original videos:** ~1,000 YouTube videos with human faces.  
- **Manipulation techniques included:**  
 ----
- **Resolution:** 720p (available at different quality levels: high, medium, low).  
- **Labels:**  
  - 'real' (unaltered)  
  - 'fake' (manipulated)  
---

## Collection and Preprocessing
- **Source:** YouTube videos, selected by the original authors for academic use.  
- **Manipulation:** Automated generation using specific face manipulation algorithms.  
- **Preprocessing in this project:**  
    - Extracted frames from videos.  
    - Balanced classes (real vs fake).  
    - Normalized image size (e.g. 224x224 px).  
    - Converted to formats compatible with our models (`.jpg`, `.png`, `.mp4` quin???).
  - Applied **data augmentation techniques** to increase dataset diversity and improve model generalization, including:  
    - Horizontal mirroring (flipping)  
    - Color adjustments (brightness, contrast, saturation)  
    - Rotations and affine transformations  
    - Other minor transformations for robustness   

---

## Intended Use
- **In this project:**  
  - Train binary classifiers (real vs fake).    
  - Demonstrate reproducible pipelines with MLOps (tracking, model registry, deployment).  

- **Recommended uses:**  
  - Academic research.  
  - Development of deepfake detection model.  
---

## Limitations
- **Domain restriction:** Only includes videos of human faces.  
- **Bias:** Source videos from YouTube may not represent global diversity.  
- **Warning:** Not suitable for real-world security applications without further validation.  

---

## Distribution
- **Location in this repository:** `/deepfake_recognition/data/`  
- **Original download:** [FaceForensics++ Download](https://github.com/ondyari/FaceForensics#download)  
- **License:** Academic use only.  
- **Restrictions:** No commercial use allowed.  

---

## Maintenance
- **Responsible in this project:** TAED2_M4chineOps Team  
- **Contact:** maria.gesti@estudiantat.upc.edu  (????)


- **Next steps:**  
  - Document model performance trained with this dataset.  


