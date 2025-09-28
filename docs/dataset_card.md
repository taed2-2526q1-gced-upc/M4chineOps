# Dataset Card: FaceForensics++ (adapted for TAED2_M4chineOps)

## Identification
- **Dataset name:** FaceForensics++  
- **Original authors:** Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner  
- **Integrated in project:** TAED2_M4chineOps – Deepfake Recognition  
- **Original repository:** [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)  
- **Paper:** https://arxiv.org/abs/1901.08971  
- **Version used:** 15.07.2020 (date of last change: FaceShifter added)  
- **Date integrated in project:** 2025-09-18  

---

## Motivation
The FaceForensics++ (FF++) dataset was created to detect and analyze manipulated human faces.    
In this project, it serves as the primary source to train, validate, and evaluate a deepfake detector within a reproducible MLOps pipeline (data → metadata → splits → training), keeping large assets out of Git and pulling them via DVC.

---

## Composition
**Original sources**
- **YouTube originals:** 1,000 videos.  
- **Actors originals (DeepFakeDetection/actors):** 363 videos.

**Manipulation methods (as used in our code)**  
- Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter (FF++ methods)  
- DeepFakeDetection (actors release)

**Coverage per original**
- For the **YouTube set**, each of the 1,000 originals has five manipulated versions → 5,000 manipulated videos.  
- For the **actors set**, there is a folder with ~3,000 manipulated videos generated using multiple methodologies.  
- Every original video has at least one manipulated counterpart produced by one of the methods above.

**Labels**
- "real" (unaltered) for originals, "fake" (manipulated) for edited versions.  
- In our preprocessing code, labels are mapped as: 0 = original, 1 = manipulated.

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
  - Optionally standardize frame size (224×224 px) or keep original resolutions depending on experimental setup.  
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
- **Audio absence:** Videos have no audio.  
- **Heterogeneous resolutions:** Not all videos share the same frame dimensions (this must be considered in preprocessing).  
- **Not production-ready:** Intended for academic use only.  

---

## Distribution
- **Location in this repository:** "/deepfake_recognition/data/"  
- **Original download:** [FaceForensics++ Download](https://github.com/ondyari/FaceForensics#download)  
- **License:** Academic use only.  
- **Restrictions:** No commercial use allowed.  

---

## Maintenance
- **Responsible in this project:** TAED2_M4chineOps Team  
- **Contact:** maria.gesti@estudiantat.upc.edu  


- **Next steps:**  
  - Document model performance trained with this dataset.  


