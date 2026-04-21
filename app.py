from dotenv import load_dotenv
from evaluator import evaluate
from items import Item
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from vanilla_nn import NeuralNetwork

LITE_MODE = False

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
    vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
    X = vectorizer.fit_transform(documents)
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


def train_model(model, train_loader, X_val, y_val, loss_fn, optimizer, epochs=2):
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_fn(val_outputs, y_val)

        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.3f}, Val Loss: {val_loss.item():.3f}"
        )


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

    X, vectorizer = vectorize_documents(documents)
    X_train, X_val, y_train, y_val = prepare_tensors(X, y)

    train_loader = create_dataloader(X_train, y_train)

    model, loss_fn, optimizer = build_model(X_train.shape[1])

    train_model(model, train_loader, X_val, y_val, loss_fn, optimizer)

    predict_fn = build_predict_fn(model, vectorizer)
    evaluate(predict_fn, test)


if __name__ == "__main__":
    main()
