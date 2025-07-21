# 🩺 DermaAI – AI-Powered Skin Disease Classification

> A deep learning framework for automated skin disease detection using dermatoscopic images and state-of-the-art CNN architectures.

---

## 📌 Abstract

DermaAI is a deep learning-powered diagnostic system developed to classify multiple skin conditions from dermatoscopic images. Leveraging two powerful CNN architectures—EfficientNetB3 and ConvNeXtBase—the model is trained on over **27,153 images** spanning **10 dermatological classes** from the publicly available Skin Disease Image Dataset.  

This project addresses major challenges in dermatological imaging, including **class imbalance** and **disease similarity**, through extensive data augmentation and transfer learning techniques. Our experiments highlight the strengths of ConvNeXt in generalization compared to EfficientNetB3, making it a reliable tool for real-world deployment.

---

## 📚 Keywords

`deep learning` • `convolutional neural networks` • `skin disease classification` • `medical imaging` • `EfficientNet` • `ConvNeXt` • `AI in dermatology`

---

## 🧠 Introduction

Skin diseases impact **900+ million people** globally, with disparities in diagnosis often arising due to limited access to dermatologists and variability across skin tones. Manual diagnosis is time-consuming and subjective, calling for **automated, AI-based diagnostic tools**.  

DermaAI aims to bridge this gap using CNN-based classification models trained on a diverse dataset. This approach brings efficiency, scalability, and fairness to skin disease diagnosis—especially for **under-resourced regions**.

---

## 🗂 Dataset

- **Source**: Skin Disease Image Dataset by Ismail Promus (Kaggle)
- **Total Images**: 27,153
- **Classes**: 10
- **Preprocessing Steps**:
  - Resized to 256×256
  - Label encoding
  - Data augmentation (flip, rotate, brightness)
  - Normalization
  - Stratified train-test split

---

## 🏗️ Model Architectures

### 1️⃣ EfficientNetB3 (Transfer Learning)

- Pretrained on ImageNet
- Frozen base layers
- 4 dense layers (1024 → 256)
- Softmax output (10 classes)
- Optimizer: Adam (`lr=1e-3`)
- Observed: High training accuracy, signs of **overfitting**

### 2️⃣ ConvNeXtBase (Transfer Learning)

- Pretrained on ImageNet
- Global average pooling
- Dropout regularization
- Dense layer (512 units)
- Softmax output (10 classes)
- Optimizer: Adam (`lr=1e-4`)
- Observed: **Higher validation accuracy**, better **generalization**

---

## 🧪 Training Setup

- **Frameworks**: TensorFlow & Keras
- **Hardware**: NVIDIA T4 GPU
- **Batch Size**: 32
- **Epochs**: 7 (with Early Stopping, ReduceLROnPlateau)
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Metrics**: Accuracy

---

## 📊 Results

| Model         | Train Acc | Train Loss | Val Acc | Val Loss | Learning Rate |
|---------------|-----------|------------|---------|----------|----------------|
| EfficientNetB3 | 96.28%    | 0.1037     | 91.14%  | 0.3126   | 1e-6           |
| ConvNeXt       | 89.48%    | 0.3029     | 92.64%  | 0.2077   | 1e-4           |

### 🔍 Analysis:

- **EfficientNetB3** showed strong training performance but **overfitted**.
- **ConvNeXt** offered better **validation performance** and generalization.
- ConvNeXt had a **stable learning curve** with a consistent accuracy-to-loss ratio.

---

## 🧾 Conclusion

DermaAI demonstrates the practical utility of AI in medical imaging, especially for dermatological diagnosis. While EfficientNetB3 achieved impressive training accuracy, ConvNeXt showed **superior generalization**—a crucial trait for clinical applications. This research supports the integration of deep learning in **scalable, accessible** healthcare tools to address diagnostic challenges globally.

---

## 🚀 Features

- Transfer learning with two CNN backbones
- Extensive augmentation pipeline
- GPU-accelerated training
- Comparison of model generalization
- Visualizations and metrics tracking

---

## 🛠️ Future Work

- Expand dataset to include more disease classes
- Integrate explainable AI techniques (Grad-CAM, SHAP)
- Optimize model deployment (e.g., TensorFlow Lite, ONNX)
- Web-based prediction UI for public health usage

---

## 🤝 Contributing

Open-source contributions are welcome! If you’d like to contribute improvements or deploy the model for clinical usage, feel free to open a pull request or fork the repository.

