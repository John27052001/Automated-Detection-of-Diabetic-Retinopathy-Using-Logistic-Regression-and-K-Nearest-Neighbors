# Automated Detection of Diabetic Retinopathy
**Implementation Project – Logistic Regression and KNN (From Scratch)**  

## 🎯 Overview
This project detects diabetic retinopathy (DR) from retinal fundus images using
two classical machine-learning algorithms — Logistic Regression and K-Nearest Neighbors — implemented entirely from scratch with NumPy.

## 🩺 Dataset
**Name:** RetinaMNIST (MedMNIST collection)  
**Images:** ≈ 1 600 fundus photos labeled 0–4 (DR severity)  
**Classes:** 0 = No DR  |  1–4 = DR  
**Split:** Train 1080  |  Val 120  |  Test 400  
**Source:** [https://medmnist.com](https://medmnist.com)

## ⚙️ Preprocessing
1. Grayscale conversion  
2. Resize → 64 × 64 pixels  
3. Flatten → 4 096 features  
4. Normalize pixels [0, 1]  
5. Z-score standardization (μ, σ from train set)

## 🧠 Algorithms
### Logistic Regression
Sigmoid activation, binary cross-entropy loss, gradient descent.  
### KNN
Euclidean distance metric, majority vote, best k = 11.

## 💻 Implementation
Python | NumPy | Matplotlib | Google Colab  

## 📈 Results
| Metric | Logistic Regression | KNN (k = 11) |
|:--:|:--:|:--:|
| Accuracy | 0.718 | 0.710 |
| Precision | 0.770 | 0.789 |
| Recall | 0.712 | 0.664 |
| F1 Score | 0.740 | 0.721 |
| AUC | 0.707 | 0.757 |

## 🔍 Confusion Matrix
Model	Confusion Matrix
Logistic Regression	[[126 48], [65 161]]
KNN (k=11)	[[134 40], [76 150]]

## 📈 ROC Curve
 <p align="center"> <img src="roc_curve.png" alt="ROC Curve" width="500"/> </p>

 ## 💬 Key Findings

Both models achieved ~71% accuracy on the test set.

Logistic Regression → Higher recall → Better at identifying DR cases.

KNN → Higher precision and AUC → Better at reducing false positives.

Confirms that classical algorithms can perform reliable DR screening even without deep learning.

## 🧠 Author
**Megha John Babu**  
School of Computer Science and Engineering  
California State University, San Bernardino  
📧 meghajohnbabu@csusb.edu
