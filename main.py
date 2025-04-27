import pyprind
import pandas as pd
import numpy as np
import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, KFold
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    def __init__(self, input_dim, hidden_dims):
        super(FNN, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
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

for hidden in ([100], [200,100], [200,100,50]):     # [n] Hidden layer of n neurons
    for lr_rate in (1e-2, 1e-3):                    # Learning Rate
        for wd in (0.0, 1e-4):                      # Weight decay (L2 reg)
            model = FNN(input_dim, hidden)
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


"""--------------------------- PART 4 ----------------------------------------------"""


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


k_folds = 5
num_epochs = 5
loss_function = nn.CrossEntropyLoss()
results = {}

# Set fixed random number seed
torch.manual_seed(42)

# Combine the original train and test sets for K-Fold
full_X = torch.tensor(x_tfidf.toarray(), dtype=torch.float32)
full_y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(full_X, full_y)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# Start print
print('--------------------------------')

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        sampler=test_subsampler)

    # Create FNN model
    model = FNN(input_dim, hidden_dims=[100, 50])
    model.apply(reset_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Accuracy for fold {fold}: {accuracy:.2f}%')
    results[fold] = accuracy

# Process is complete.
print('Training process has finished. Saving trained model.')

# Print about testing
print('Starting testing')

# Saving the model
save_path = f'./model-fold-{fold}.pth'
torch.save(model.state_dict(), save_path)

# Evaluation
correct, total = 0, 0
model.eval()
with torch.no_grad():
    for xb, yb in testloader:
        outputs = model(xb)
        _, preds = torch.max(outputs, dim=1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()

acc = 100.0 * correct / total
print(f'Accuracy for fold {fold}: {acc:.2f}%')
results[fold] = acc

# Print final results
print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value:.2f}%')
    sum += value
print(f'Average: {sum / len(results.items())} %')


"""--------------------------- PART 5 ----------------------------------------------"""

class FNN_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_prob=None):
        super(FNN_Dropout, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def bagging_predict(models, X):
    bagging_outputs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            bagging_outputs.append(torch.softmax(model(X), dim=1))
    avg_output = torch.mean(torch.stack(bagging_outputs), dim=0)
    preds = torch.argmax(avg_output, dim=1)
    return preds



print("\n=== Task 5.1: Single Dropout Model vs Baseline ===")

# Train baseline model (NO DROPOUT)
baseline_model = FNN_Dropout(input_dim, best['hidden'], dropout_prob=None)
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=best['lr'], weight_decay=best['wd'])
baseline_criterion = nn.CrossEntropyLoss()

baseline_start = time.time()
train_model(baseline_model, baseline_optimizer, baseline_criterion, train_loader, epochs=5)
baseline_time = time.time() - baseline_start
baseline_acc = evaluate_model(baseline_model, X_test, y_test)
print(f"Baseline FNN — Time: {baseline_time:.1f}s — Accuracy: {baseline_acc:.4f}")

# Train single dropout model
dropout_model = FNN_Dropout(input_dim, best['hidden'], dropout_prob=0.5)
dropout_optimizer = optim.Adam(dropout_model.parameters(), lr=best['lr'], weight_decay=best['wd'])
dropout_criterion = nn.CrossEntropyLoss()

dropout_start = time.time()
train_model(dropout_model, dropout_optimizer, dropout_criterion, train_loader, epochs=5)
dropout_time = time.time() - dropout_start
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
