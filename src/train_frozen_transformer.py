#import libraries 
import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn import set_config
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
set_config(transform_output="pandas")

pd.set_option('mode.chained_assignment', None)
pd.options.display.max_rows = 1000


def calc_metrics(submission, dtypes=["train", "test"]):
    result = {}
    for dtype in dtypes:
        name = f"MSE_{dtype}"
        mse = None
        sample = submission[submission["type"] == dtype]
        if not sample["rating"].isnull().all():
            mse = mean_squared_error(sample["rating"], sample["predict_rating"])

        result[name] = mse

    return result


NAME = "Belton_Manhica"
train_and_test = pd.read_csv(f"0_{NAME}.csv")

# split dataset for train and test part
train = train_and_test[train_and_test["type"] == "train"]
test = train_and_test[train_and_test["type"] == "test"]
train_and_test.shape, train.shape, test.shape


os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_train, data_valid = train_test_split(train[train["type"] == "train"], test_size=0.1, random_state=42)

#choose the right batch size 
BATCH_SIZE = 4

#transfor the data into Hugging face dataset
train_data = Dataset.from_pandas(data_train)
valid_data = Dataset.from_pandas(data_valid)
test_data = Dataset.from_pandas(test)

#tokenization
from transformers import AutoTokenizer, AutoModel

#select model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

#initialize the all-MiniLM-L6-v2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#create a class for the tokenization for the whole data

class MyDataset():
  def __init__(self, train_and_test, tokenizer, max_len = 128):
    self.comments = train_and_test['comment'].tolist()
    self.ratings = train_and_test['rating'].tolist()
    self.tokenizer = tokenizer
    self.max_len =  max_len

  def __len__(self):
    return len(self.comments)

  def __getitem__(self, idx):
    data_encoded = self.tokenizer(
        self.comments[idx],
        truncation = True,
        padding = 'max_length',
        max_length =self.max_len,
        return_tensors = 'pt'
    )

    rating = self.ratings[idx]
    rating = torch.tensor(0.0 if pd.isna(rating) else rating, dtype = torch.float)

    return {
            "input_ids": data_encoded["input_ids"].squeeze(0),
            "attention_mask": data_encoded["attention_mask"].squeeze(0),
            "rating": rating
        }

#dataloaders
train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
valid_dl = DataLoader(valid_ds, batch_size = BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size = BATCH_SIZE)

#check if there is a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Build the model (Freeze Transformer + Full connected layers)

class RatingModel(torch.nn.Module):
  def __init__(self, model_name):
    super().__init__()
    self.model = AutoModel.from_pretrained(model_name)

    #freeze model weights
    for p in self.model.parameters():
      p.requires_grad = False

    hidden_size = self.model.config.hidden_size
    self.fc = torch.nn.Linear(hidden_size, 1)

  def forward(self, input_ids, attention_mask):
    with torch.no_grad():
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    logits = self.fc(cls_emb).squeeze(-1)
    logits = torch.clip(logits, min =0, max = 10)
    return logits

model = RatingModel(model_name).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-3)

EPOCHS = 3
train_losses, valid_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for batch in train_dl:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["rating"].to(device)

        optimizer.zero_grad()
        preds = model(input_ids, mask)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_dl))

    # ---- validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valid_dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["rating"].to(device)

            preds = model(input_ids, mask)
            loss = criterion(preds, targets)
            val_loss += loss.item()

    valid_losses.append(val_loss / len(valid_dl))

    print(f"Epoch {epoch+1}/{EPOCHS} | Train={train_losses[-1]:.4f} | Valid={valid_losses[-1]:.4f}")

def predict_dataset(dataset):
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask).cpu().numpy().tolist()
            preds.extend(out)
    return preds

# Build predict_rating column in original order
preds_train = predict_dataset(train_ds)
preds_valid = predict_dataset(valid_ds)
preds_test  = predict_dataset(test_ds)

preds_full = np.concatenate([np.array(preds_train), np.array(preds_valid), np.array(preds_test)])

train_and_test["predict_rating"] = preds_full
scores = calc_metrics(train_and_test)
print(scores)

# Final checks and prepare submission
if train_and_test.shape[0] != 10000:
    raise ValueError(f'Incorrect train_and_test file shape should be a 10000. {train_and_test.shape[0]} are given')

if "predict_rating" not in train_and_test.columns:
    raise ValueError(f'Column "predict_rating" should be in train_and_test dataset')

if train_and_test["predict_rating"].isnull().sum() > 0:
    raise ValueError(f'Column "predict_rating" have null values')

if (train_and_test["predict_rating"] < 0.).sum() > 0:
    raise ValueError(f'Column "predict_rating" contain negative values')

if (train_and_test["predict_rating"] > 10.).sum() > 0:
    raise ValueError(f'Column "predict_rating" contain values more than 10.')

train_and_test[["predict_rating"]].to_csv(f'3_{NAME}.csv', index=False)

if __name__ == "__main__":


