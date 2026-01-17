# Frozen Transformer Rating Predictor

This repository demonstrates how to **train a custom rating prediction model using a frozen Transformer encoder** and a lightweight regression head.

Instead of fine-tuning the entire Transformer, the pretrained model is used purely as a **feature extractor**, while only a small fully connected layer is trained. This approach is efficient, stable, and works well on small-to-medium datasets.

---

## 🚀 Project Overview

* Task: **Predict customer ratings (0–10)** from textual reviews
* Input: Customer comments/reviews
* Output: Continuous rating score
* Model strategy:

  * Frozen Transformer encoder
  * Trainable linear regression head

---

## 🧠 Model Architecture

### Encoder

* **Model**: `sentence-transformers/all-MiniLM-L6-v2`
* Loaded via Hugging Face `AutoModel`
* All Transformer parameters are **frozen** (`requires_grad = False`)

### Regression Head

* Single fully connected layer
* Input: CLS token embedding
* Output: 1 scalar rating
* Output clipped to `[0, 10]`

---

## 🧪 Training Strategy

* Loss: Mean Squared Error (MSE)
* Optimizer: SGD (only regression head parameters)
* Epochs: 3
* Batch size: 4
* Validation split: 10%

This setup minimizes overfitting while maintaining interpretability and fast training.

---

## 📊 Dataset

* Input CSV: `0_<NAME>.csv`
* Required columns:

  * `comment` – text review
  * `rating` – ground truth score
  * `type` – `train` / `test`

The dataset is split into:

* Train
* Validation
* Test

Final predictions are written to:

```text
3_<NAME>.csv
```

---

## 🧩 Pipeline Steps

1. Load and inspect dataset
2. Analyze rating distribution and text length
3. Tokenize comments using Hugging Face tokenizer
4. Build custom PyTorch Dataset
5. Freeze Transformer encoder
6. Train regression head
7. Evaluate using MSE
8. Generate predictions for all samples
9. Export submission file

---

## 📦 Dependencies

```txt
pandas
numpy
scikit-learn
torch
transformers
datasets
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📈 Evaluation Metrics

Mean Squared Error (MSE) is computed separately for:

* Training set
* Test set

```python
MSE_train
MSE_test
```

---

## ⚠️ Notes & Design Decisions

* Transformer weights are frozen for:

  * Faster training
  * Reduced GPU memory usage
  * Better stability on small datasets
* CLS token embedding is used for regression
* Text truncation is unnecessary (max length < model context)

---

## 🔮 Future Improvements

* Unfreeze top Transformer layers (partial fine-tuning)
* Try alternative pooling strategies (mean pooling)
* Replace SGD with AdamW
* Compare against embedding + classical ML baselines
* Add cross-validation

---

## 📄 License

This project is intended for educational and research purposes.

---

⭐ If you find this repository useful, feel free to star it!
