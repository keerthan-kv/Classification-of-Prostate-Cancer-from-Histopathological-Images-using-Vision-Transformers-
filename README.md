# Deep Learning-Based Classification of Prostate Cancer from Histopathological Images

A deep learning project using **Swin Transformer** for binary classification of prostate cancer from histopathological images, with interactive visualization tools built using Gradio.

## 🎯 Project Overview

This project implements a robust pipeline for detecting and classifying prostate cancer from histopathological tissue images. It leverages the power of Vision Transformers (specifically Swin Transformer) to distinguish between benign (GG0) and malignant (GG1-GG5) prostate tissue samples.

## ✨ Features

- **Advanced Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement
- **State-of-the-art Model**: Swin Transformer architecture for accurate classification
- **Cancer Region Detection**: Automatic detection and highlighting of suspicious regions
- **Interactive Visualization**: Gradio web interfaces for real-time predictions and analysis
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Confusion Matrix Analysis**: Interactive tool for analyzing model performance

## 📊 Dataset Structure

```
prostrate_cancer_data/
├── GG0/    # Benign (Gleason Grade 0)
├── GG1/    # Malignant
├── GG2/    # Malignant
├── GG3/    # Malignant
├── GG4/    # Malignant
└── GG5/    # Malignant
```

**Label Mapping**: Binary classification (GG0 → 0, GG1-GG5 → 1)

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, timm (Swin Transformer)
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn, Gradio
- **Image Preprocessing**: CLAHE, Denoising, Normalization

## 📋 Requirements

```bash
pip install torch torchvision
pip install timm
pip install opencv-python
pip install gradio
pip install scikit-learn
pip install matplotlib seaborn
pip install pandas numpy tqdm
```

## 🚀 Usage

### 1. Data Preprocessing
Run the preprocessing cells to:
- Apply CLAHE enhancement
- Resize images to 224×224
- Normalize pixel values
- Save preprocessed data as `.npy` files

### 2. Model Training
```python
# Train the Swin Transformer model
# 10 epochs with AdamW optimizer
# Binary Cross-Entropy with Logits Loss
```

### 3. Model Evaluation
- Confusion Matrix
- Classification Report
- ROC Curve & AUC Score
- Performance Metrics (Accuracy, Precision, Recall, F1)

### 4. Interactive Prediction
Launch the Gradio interface for real-time predictions:
```python
# Provides:
# - Original image view
# - Denoised image
# - Cancer region highlighting
# - Prediction with confidence score
```

### 5. Confusion Matrix Analyzer
Upload true and predicted labels to visualize model weaknesses:
```python
# Helps identify:
# - False positives
# - False negatives
# - Model biases
```

## 📈 Model Architecture

**Swin Transformer** (Shifted Window Transformer)
- Model: `swin_tiny_patch4_window7_224`
- Pre-trained: ImageNet
- Output: Binary classification (1 logit)
- Activation: Sigmoid for probability

## 🎓 Key Concepts

- **CLAHE**: Enhances local contrast in histopathological images
- **Adaptive Thresholding**: Detects potential cancer regions
- **Stratified Split**: Maintains class balance (70% train, 15% val, 15% test)
- **Vision Transformers**: Captures global context better than CNNs

## 📊 Results

The model achieves:
- High accuracy on test set
- Strong ROC-AUC performance
- Effective detection of malignant tissue
- Visual highlighting of suspicious regions

## 🔮 Future Enhancements

- [ ] Multi-class classification (GG0 through GG5)
- [ ] Grad-CAM visualization for model interpretability
- [ ] Data augmentation techniques
- [ ] Ensemble models
- [ ] Mobile deployment
- [ ] Integration with medical imaging systems

## 📝 File Structure

```
kvpc/
├── trials.ipynb                    # Main notebook
├── train_split.csv                 # Training data paths
├── val_split.csv                   # Validation data paths
├── test_split.csv                  # Test data paths
├── .gitignore                      # Git ignore rules
├── preprocessed_data/              # Preprocessed .npy files
└── prostrate_cancer_data/          # Original dataset
```
