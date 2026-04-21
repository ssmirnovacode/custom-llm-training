# Vanilla Neural Network for Price Prediction

This project trains a simple feedforward neural network to predict item prices from text summaries using a hashing vectorizer and PyTorch.

## Features

- Loads 800k training items from Hugging Face (`ed-donner/items_full`, pre-processed and curated)
- Uses `HashingVectorizer` (binary, 3000 features) for fast, memory‑efficient text encoding
- Implements a deep MLP with 8 linear layers (128 → 64 → … → 1) and ReLU activations
- Trains with MSE loss and Adam optimizer
- Evaluates on validation/test sets with visual reports (error trend chart, scatter plot)

## Project Files

- `app.py` – main training and evaluation pipeline
- `vanilla_nn.py` – defines the `NeuralNetwork` model
- `evaluator.py` – evaluation logic with charts and error reporting (uses `plotly`, `pandas`, `tqdm`)
- `items.py` – dataset loading utilities using `pydantic` and Hugging Face `datasets`

## Requirements

Install dependencies (minimal set for this project):

```bash
pip install torch scikit-learn tqdm python-dotenv datasets pydantic pandas plotly
```
