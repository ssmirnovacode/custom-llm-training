from dotenv import load_dotenv
from evaluator import evaluate
from items import Item
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from vanilla_nn import NeuralNetwork

SIZE = 3000

load_dotenv(override=True)


def load_dataset_from_hf():
    ds = "ed-donner/items_full"
    train, val, test = Item.from_hub(ds)
    print(f"Loaded {len(train):,}, {len(val):,}, {len(test):,}")
    return train, val, test


def prepare_documents(train):
    y = np.array([float(item.price) for item in train])
    documents = [item.summary for item in train]
    return documents, y


def vectorize_documents(documents):
    vectorizer = HashingVectorizer(n_features=SIZE, stop_words="english", binary=True)
    X = vectorizer.transform(documents)
    return X, vectorizer


def prepare_tensors(X, y):
    X_tensor = torch.FloatTensor(X.toarray())
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    return train_test_split(X_tensor, y_tensor, test_size=0.01, random_state=42)


def create_dataloader(X_train, y_train, batch_size=64):
    dataset = TensorDataset(X_train, y_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_model(input_size):
    model = NeuralNetwork(input_size)
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer


def batch_generator(documents, y, vectorizer, batch_size=64):
    for i in range(0, len(documents), batch_size):
        docs_batch = documents[i : i + batch_size]
        y_batch = y[i : i + batch_size]

        X_batch = vectorizer.transform(docs_batch)
        X_batch = torch.FloatTensor(X_batch.toarray())
        y_batch = torch.FloatTensor(y_batch).unsqueeze(1)

        yield X_batch, y_batch


def train_model(model, documents, y, vectorizer, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()

        last_loss = None

        for batch_X, batch_y in tqdm(batch_generator(documents, y, vectorizer)):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {last_loss:.3f}")


def evaluate_model(model, val, vectorizer, loss_fn, sample_size=SIZE):
    model.eval()

    val_sample = val[:sample_size]
    docs = [item.summary for item in val_sample]
    y = torch.FloatTensor([float(item.price) for item in val_sample]).unsqueeze(1)

    with torch.no_grad():
        X = vectorizer.transform(docs)
        X = torch.FloatTensor(X.toarray())

        outputs = model(X)
        loss = loss_fn(outputs, y)

    print(f"Validation Loss: {loss.item():.3f}")


def build_predict_fn(model, vectorizer):
    def predict(item):
        model.eval()
        with torch.no_grad():
            vector = vectorizer.transform([item.summary])
            vector = torch.FloatTensor(vector.toarray())
            result = model(vector)[0].item()
        return max(0, result)

    return predict


def main():
    train, val, test = load_dataset_from_hf()
    documents, y = prepare_documents(train)

    vectorizer = HashingVectorizer(n_features=SIZE, stop_words="english", binary=True)

    model = NeuralNetwork(input_size=SIZE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, documents, y, vectorizer, loss_fn, optimizer)

    evaluate_model(model, val, vectorizer, loss_fn)

    predict_fn = build_predict_fn(model, vectorizer)
    evaluate(predict_fn, test)


if __name__ == "__main__":
    main()
