# CNN-BoostForest: A Synergistic Approach for Pneumonia Detection

**Repository:** [Pneumonia Detection](https://github.com/yourusername/pneumonia-detection)  
**Authors:** Jazzlyn Amelia Lim, Cindy Noveiren

---

## Overview
This project implements a **hybrid deep learning and ensemble learning model** for detecting pneumonia from chest X-ray images. We combine **DenseNet201** for feature extraction with **BoostForest** for classification, and compare its performance against:

- End-to-end CNN  
- CNN + Random Forest  
- CNN + Extra Trees  
- CNN + XGBoost  
- CNN + LightGBM  

**Highlights:**
- Achieved **93.95% accuracy**, **94.03% precision**, **F1-score 93.95%**, **AUC 0.97**  
- Hybrid CNN-BoostForest outperforms CNN end-to-end and is competitive with other CNN-hybrid models  
- Computationally heavier but offers strong predictive stability  

---

## Notebooks

1. **`Pneumonia_CNN.ipynb`** – End-to-end CNN training and evaluation  
2. **`Pneumonia_CNN_Hybrid.ipynb`** – Hybrid CNN ensemble models training, evaluation, and comparison  

---

## Methodology

![Methodology Workflow](assets/methodology_workflow.png)

![DenseNet201 Feature Extractor](assets/densenet201_feature_extractor.png)

![BoostForest Architecture](assets/boostforest_architecture.png)
