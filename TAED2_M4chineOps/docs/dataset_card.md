# Dataset Card: FaceForensics++ (adapted for TAED2_M4chineOps)

## Identification
- **Dataset name:** FaceForensics++  
- **Original authors:** Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner  
- **Integrated in project:** TAED2_M4chineOps – Deepfake Recognition  
- **Original repository:** [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)  
- **Paper:** [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)  
- **Version used:** 15.07.2020 (date of last change: FaceShifter added)  
- **Date integrated in project:** 2025-09-18  

---

## Motivation
The FaceForensics++ dataset was created to detect and analyze manipulated human faces.  
In this project, we use it to train, validate, and evaluate models for deepfake detection within an MLOps workflow.

---

## Composition
- **Original videos (YouTube):** 1,000 original videos  
- **Original videos (actors set):** 363 original actor videos
- **Manipulated videos (YouTube-derived):** For each of the 1,000 YouTube originals there are 5 manipulated versions; 5,000 manipulated videos from YouTube originals.  
- **Manipulated videos (actors-derived):** Approximately 3,000 manipulated videos in the actors folder, produced with a variety of methodologies.
- **Total approximate manipulated videos:** ~8,000 (5,000 from YouTube originals + ~3,000 from actors set)_  
- **Labels:**  
  - "real" (unaltered)  
  - "fake" (manipulated)  
- **Resolution:** videos are available at multiple resolutions (not all files share the exact same spatial dimensions).  

---

## File formats and characteristics
- All video files are ".mp4".  
- All videos are stored without audio.
- Frame sizes are not identical across all files.  

---

## Collection and Preprocessing
- **Source:** YouTube videos and actor-recorded videos selected by the original authors for academic use.  
- **Manipulation:** Automated generation using a set of face-manipulation algorithms
  
- **Preprocessing in this project:**  
  - Extract frames from ".mp4" videos.  
  - **Optionally standardize frame size (224×224 px) or keep original resolutions depending on experimental setup.**  
  - Balance classes (real vs fake) when necessary for training.  
  - Convert/save frames to model-compatible image formats (".jpg", ".png").  
  - Apply data augmentation (flips, color jitter, rotations, affine transforms, distortions).  
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
- **Domain restriction:** Contains only videos with human faces.  
- **Bias:** Source videos (YouTube and actor sets) may not represent global demographic diversity.  
- **Audio absence:** Videos have no audio — explicitly noted.  
- **Heterogeneous resolutions: ** Not all videos share the same frame dimensions (this must be considered in preprocessing).  
- **Not production-ready:** Intended for academic use only.  

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


