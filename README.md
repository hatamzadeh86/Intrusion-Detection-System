# Intrusion-Detection-System
NIDS project using Neural Networks and Random Forest for network intrusion detection.




# 🛡️ Advanced Network Intrusion Detection System (NIDS)

> A production-oriented Intrusion Detection System built with Deep Learning and Machine Learning to classify network traffic as Normal or Malicious.

---

## 📌 Project Overview

This project implements a high-performance **Network Intrusion Detection System (NIDS)** using PyTorch and Scikit-learn.

The system analyzes structured network traffic features and determines whether a connection is:

- ✅ Normal Traffic  
- 🚨 Malicious Activity (Attack)

The objective is to build a reliable first-layer cybersecurity defense model capable of detecting intrusions with high recall and minimal false negatives.

---

# 🚀 Core Engineering Innovation

## 🔄 Problem Re-Engineering: From 20 Classes to Binary Detection

### Original Scenario:
The dataset originally contained **20 different attack classes**, making it a multi-class classification problem.

### Strategic Decision:
The task was transformed into a **Binary Classification Problem**:

- Class 0 → Normal Traffic
- Class 1 → Attack (All 20 attack types aggregated)

### Why This Matters Technically:

- Reduces decision boundary complexity
- Improves model stability
- Increases Recall for attack detection
- Makes the system practical for real-world IDS pipelines

Mathematically:

Multi-class:
f: R^n → {1,2,…,20}

text

Binary:
f: R^n → {0,1}

text

This transformation improved generalization and reduced overfitting risk.

---

# 🧠 Deep Learning Architecture

## ✅ Model Type
Multi-Layer Perceptron (MLP)

## ✅ Framework
PyTorch

## ✅ Loss Function
`BCEWithLogitsLoss`

### Why Not Sigmoid + BCELoss?

Because:

- BCEWithLogitsLoss combines Sigmoid + BCE in one numerically stable implementation.
- Prevents floating-point instability.
- Avoids vanishing gradient issues in early training.

Formula:

Loss = -[ y log(sigmoid(x)) + (1-y) log(1 - sigmoid(x)) ]

Implemented internally in a stable way.

---

# ⚙️ Automated Hyperparameter Optimization (Optuna)

Instead of manual tuning, this project integrates **Optuna** for intelligent hyperparameter search.

### Optimized Parameters:
- Number of hidden layers
- Hidden units
- Learning rate
- Dropout rate

### Optimization Method:
- Bayesian Optimization
- Tree-structured Parzen Estimator (TPE)

### Why This Is Important:
- Prevents suboptimal architecture choices
- Maximizes F1-score
- Demonstrates advanced ML engineering practice

---

# 🛑 Overfitting Control: Early Stopping

Training showed signs of overfitting after several epochs.

Solution:
- EarlyStopping with patience=10

Result:
- Training stopped at Epoch 24
- Best validation performance preserved
- Improved generalization

---

# 🌲 Baseline Model: Random Forest

To validate performance, a classical ensemble method was implemented.

- Framework: Scikit-learn
- Purpose: Baseline comparison

This allows performance benchmarking between:
- Deep Learning
- Traditional Machine Learning

---

# 📊 Results

| Metric | Neural Network | Random Forest |
|--------|----------------|---------------|
| Accuracy | 98.5% | 99.1% |
| F1-Score | 0.97 | 0.98 |
| Recall (Attack) | High | Very High |

(*Update values according to your actual results*)

---

# 📈 Training Visualization

## Loss Curve

The following figure shows training and validation loss:

![Loss Curve](results/loss_curve.png)

This demonstrates:
- Proper convergence
- Early stopping intervention
- No severe overfitting

---

# ⚠️ Debugging & Engineering Challenges

Throughout development, several technical challenges were resolved:

### 1️⃣ Tensor Shape Mismatch
Resolved using:
```python
batch_Y = batch_Y.view(-1, 1)
2️⃣ GPU to CPU Conversion
python
.cpu().numpy().flatten()
3️⃣ File Path Issues
Solved using:

python
os.path.join()
4️⃣ Data Type Management
.float() for loss computation
.int() for metrics
📁 Project Structure
text
data/
src/
├── data_preprocessing.py
├── train_nn.py
├── train_rf.py
models/
results/
🛠 Installation
bash
pip install torch scikit-learn pandas matplotlib optuna joblib
Run:

bash
python src/train_nn.py
📬 Contact
Author: Hatamzadeh86

Field: Artificial Intelligence | Computer Vision | Cybersecurity

GitHub: https://github.com/hatamzadeh86
Email: amermohmd22213@gmail.com
LinkedIn: https://www.linkedin.com/in/amir-mohammad-hatemzadeh-44b2a138b
⭐ If you found this project interesting, feel free to star the repository!

text

---

results/

text


results/loss_curve.png

text

```markdown
![Loss Curve](results/loss_curve.png) 




![Confusion Matrix](results/confusion_matrix.png)
