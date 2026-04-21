import os
from dotenv import load_dotenv
from huggingface_hub import login
from evaluator import evaluate
from litellm import completion
from items import Item
import numpy as np
from tqdm.notebook import tqdm
import csv
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

LITE_MODE = False

load_dotenv(override=True)

if __name__ == "__main__":
    username = "ed-donner"
    dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

    train, val, test = Item.from_hub(dataset)

    print(
        f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
    )
