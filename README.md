# 🧠 Visual Question Answering
This project implements a Visual Question Answering (VQA) pipeline using two model architectures:
- 🔹 LSTM + CNN
- 🔸 Attention

![img.png](images/img.png)
---
## 📁 Dataset
The dataset used for this project is the COCO-QA dataset.

![img_1.png](images/img_1.png)

## 🔧 Setup
### 1. Create a virtual environment
```bash
python -m venv vqa_env
source vqa_env/bin/activate
```
### 2. Install the required packages
```bash
pip install -r requirements.txt
```
### 3. Download the dataset
```bash
python data/cocoqa_preprocess.py
```
## 🧼 Preprocessing
### 1. Create Question Features
```bash
python data/preprocessing.py
```
### 2. Create Image Features
Run notebook: `data/processing.ipynb` on Kaggle/Colab to use GPU for faster processing.
## 🚀 Training
### 1. LSTM + Multimodal Fusion
```bash
python train_lstm.py --batch_size 16 --max_epochs 1000
```
### 2. Attention
```bash
python train_attention.py --batch_size 16 --max_epochs 1000
```
## 🧪 Evaluation
You can use notebook `vqa_main.ipynb` for end-to-end training and evaluation.
## 📄 References
- [1] VQA: Visual Question Answering (Agrawal et al, 2016): https://arxiv.org/pdf/1505.00468v6.pdf
- [2] Hierarchical Question-Image Co-Attention for Visual Question Answering (Lu et al, 2017): https://arxiv.org/pdf/1606.00061.pdf