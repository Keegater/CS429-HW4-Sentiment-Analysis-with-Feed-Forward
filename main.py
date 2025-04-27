import pyprind
import pandas as pd
import numpy as np
import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fnn_dropout import FNN_Dropout

## Load and label
# change the 'basepath' to the directory of the
# unzipped movie dataset
basepath = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000, stream=sys.stdout)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df._append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

# clean
def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:>:-)?(?:\)|\(|D|P)',text)
    text = re.sub(r'\W+', ' ', text.lower())
    if emoticons:
        text = text + ' ' + ' '.join(emoticons).replace('-', '')
    return text


df['review'] = df['review'].apply(preprocessor)

# vectorize to tf-idf
count = CountVectorizer()
x_counts = count.fit_transform(df['review'])
tfidf = TfidfTransformer(use_idf=True, norm = 'l2', smooth_idf=True)
x_tfidf = tfidf.fit_transform(x_counts)


# split data
y = df['sentiment'].values
x_train, x_test, y_train, y_test = train_test_split( x_tfidf, y, test_size=0.30, random_state=42)

print("Train set:", x_train.shape, y_train.shape)
print("Test set:", x_test.shape, y_test.shape)


"""--------------------------- PART 2 & 3 ----------------------------------------------"""

# Convert tfidf data into PyTorch tensors
X_train = torch.tensor(x_train.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(x_test.toarray(),  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_prob=0.5):
        super(FNN, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_prob))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(model, optimizer, criterion, loader, epochs=5):
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        return (preds == Y).float().mean().item()


# Baseline
print("=== Logistic Regression baseline ===")
start = time.time()
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
lr_time = time.time() - start
lr_acc  = accuracy_score(y_test, lr.predict(x_test))
print(f"Time: {lr_time:.1f}s — Accuracy: {lr_acc:.4f}")

# Gridsearch for best hyperparameters
input_dim = X_train.shape[1]
best = {'acc': 0}

for hidden in ([100], [200,100], [200,100,50]):
    for lr_rate in (1e-2, 1e-3):
        for wd in (0.0, 1e-4):
            model = FNN(input_dim, hidden, dropout_prob=0.3)
            optimizer = optim.Adam(model.parameters(),
                                   lr=lr_rate, weight_decay=wd)
            criterion = nn.CrossEntropyLoss()
            t0 = time.time()
            train_model(model, optimizer, criterion, train_loader, epochs=5)
            acc = evaluate_model(model, X_test, y_test)
            t_total = time.time() - t0

            print(f"H={hidden}, lr={lr_rate}, wd={wd} → "
                  f"Acc: {acc:.4f}, Time: {t_total:.1f}s")

            if acc > best['acc']:
                best.update({
                    'hidden': hidden,
                    'lr': lr_rate,
                    'wd': wd,
                    'acc': acc,
                    'time': t_total
                })


print("\n>>> Best FNN configuration:")
print(f" Hidden layers: {best['hidden']}")
print(f" Learning rate: {best['lr']}")
print(f" Weight decay:  {best['wd']}")
print(f" Test accuracy: {best['acc']:.4f}")
print(f" Training time: {best['time']:.1f}s")


######part 5 ################################

def bagging_predict(models, X):
    outputs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs.append(torch.softmax(model(X), dim=1))
    avg_output = torch.mean(torch.stack(outputs), dim=0)
    preds = torch.argmax(avg_output, dim=1)
    return preds



print("\n=== Task 5.1: Single Dropout Model vs Baseline ===")

# Train baseline model (NO DROPOUT)
baseline_model = FNN_Dropout(input_dim, best['hidden'], dropout_prob=None)
optimizer = optim.Adam(baseline_model.parameters(), lr=best['lr'], weight_decay=best['wd'])
criterion = nn.CrossEntropyLoss()

start = time.time()
train_model(baseline_model, optimizer, criterion, train_loader, epochs=5)
baseline_time = time.time() - start
baseline_acc = evaluate_model(baseline_model, X_test, y_test)
print(f"Baseline FNN — Time: {baseline_time:.1f}s — Accuracy: {baseline_acc:.4f}")

# Train single dropout model
dropout_model = FNN_Dropout(input_dim, best['hidden'], dropout_prob=0.5)
optimizer = optim.Adam(dropout_model.parameters(), lr=best['lr'], weight_decay=best['wd'])
criterion = nn.CrossEntropyLoss()

start = time.time()
train_model(dropout_model, optimizer, criterion, train_loader, epochs=5)
dropout_time = time.time() - start
dropout_acc = evaluate_model(dropout_model, X_test, y_test)
print(f"Single Dropout FNN (p=0.5) — Time: {dropout_time:.1f}s — Accuracy: {dropout_acc:.4f}")


print("\nTask 5.2: Bagging ≥5 Dropout Models")

n_models = 5
bagged_models = []
start = time.time()

for i in range(n_models):
    model = FNN_Dropout(input_dim, best['hidden'], dropout_prob=0.5)
    optimizer = optim.Adam(model.parameters(), lr=best['lr'], weight_decay=best['wd'])
    criterion = nn.CrossEntropyLoss()
    train_model(model, optimizer, criterion, train_loader, epochs=5)
    bagged_models.append(model)

bagging_time = time.time() - start

# Predict with bagging
bagged_preds = bagging_predict(bagged_models, X_test)
bagging_acc = (bagged_preds == y_test).float().mean().item()
print(f"Bagging Dropout Models — Time: {bagging_time:.1f}s — Accuracy: {bagging_acc:.4f}")


print("\n Final Task 5 Comparison")
print(f"{'Model':30s} {'Time (s)':>10s} {'Accuracy':>10s}")
print("-" * 50)
print(f"{'Baseline FNN (no dropout)':30s} {baseline_time:10.1f} {baseline_acc:10.4f}")
print(f"{'Single Dropout FNN (p=0.5)':30s} {dropout_time:10.1f} {dropout_acc:10.4f}")
print(f"{'Bagging Dropout FNNs (5 models)':30s} {bagging_time:10.1f} {bagging_acc:10.4f}")
