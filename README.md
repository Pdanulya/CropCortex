# 🌿 Plant Disease Detection using Deep Learning

## 📌 Project Overview
This project is a deep learning-based Plant Disease Classification system that identifies diseases in plant leaves using image input.  
It uses **Transfer Learning (MobileNetV2)** to achieve high accuracy in classifying multiple plant disease categories.

---

## 🎯 Objective
To build an AI model that can automatically detect plant diseases from leaf images and assist in early agricultural disease diagnosis.

---

## Download Model from Google Drive: 
https://drive.google.com/file/d/17bfN9TKm21FZL_VzpAVLKud95K7iNa0U/view?usp=drive_link

---

## 🧪 How to Use the Model

1. Download the trained model from the link
2. Place it in the project folder
3. Run:

```bash
python predict.py
```

4. Replace image path with your own leaf image


5. What user workflow looks like

```text
Clone repo → Download model → Install requirements → Run predict.py
```

---

## 📊 Dataset
- Dataset used: PlantVillage Dataset
- Total classes: 38 plant disease categories
- Includes healthy and diseased leaf images

---

## 🧠 Model Architecture
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Transfer Learning approach used
- Custom classification head added:
  - Global Average Pooling
  - Dense Layer (128 units, ReLU)
  - Dropout (0.3)
  - Output Layer (Softmax for 38 classes)

---

## ⚙️ Training Details
- Input image size: 224x224
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metric: Accuracy
- Training technique: Transfer Learning 

---

## 📈 Performance
- Training Accuracy: ~93–94%
- Validation Accuracy: ~95%
- Test Accuracy: ~95%

---

## 🔍 Evaluation
- Confusion Matrix used for class-wise performance analysis
- Classification Report generated (Precision, Recall, F1-score)

---
## 📃 Medium Article: 
https://medium.com/@poojanidanulya/ever-wondered-how-ai-sees-images-heres-the-answer-575991c83d0f 

---

## 🌿 Prediction System
The model can predict plant disease from a single leaf image.

### Example:
```python
predict_disease("leaf.jpg")
