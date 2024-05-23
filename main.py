import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from typing import List
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import os

FINAL_EVALUATION_MODE = False  # W czasie sprawdzania twojego rozwiązання змінемо тą wartość на True
MODEL_PATH = 'model.pth'  # Nie змieniaj!
BATCH_SIZE = 32 # Зменьш, jeżeli твій комп'ютер має менш ніж 8gb ram

time_start = datetime.now()
if not FINAL_EVALUATION_MODE:
    TRAIN_PATH = 'C:\\Users\\nasti\\Desktop\\Neuron\\Owocowa_Burza_Neuronow\\train'
    VAL_PATH = 'C:\\Users\\nasti\\Desktop\\Neuron\\Owocowa_Burza_Neuronow\\test'
if not FINAL_EVALUATION_MODE:
    DEVICE = torch.device('cpu') # Finalne rozwiązanie буде sprawdzане на CPU!!!

if not FINAL_EVALUATION_MODE:
    class_names = [
        'apple',
        'blueberry',
        'blackberry',
        'pineapple',
        'strawberry',
        'watermelon',
        'grapes',
        'peanut',
    ]

# Діагностичний код для виведення файлів у TRAIN_PATH
print(f"Файли в TRAIN_PATH: {TRAIN_PATH}")
for root, dirs, files in os.walk(TRAIN_PATH):
    for file in files:
        print(os.path.join(root, file))

# Діагностичний код для виведення файлів у VAL_PATH
print(f"\nФайли в VAL_PATH: {VAL_PATH}")
for root, dirs, files in os.walk(VAL_PATH):
    for file in files:
        print(os.path.join(root, file))

def calculate_accuracy(y_true: list[float], y_pred: list[float]) -> float:
    y_true = torch.FloatTensor(y_true)
    y_pred = torch.FloatTensor(y_pred)
    return y_pred.eq(y_true.view_as(y_pred)).sum().detach().item() / y_true.numel()

def load_data(name_files: list[str], path: str):
    data = []
    for name in name_files:
        file_path = os.path.join(path, f"{name}.npy")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        class_data = np.load(file_path)
        class_data = class_data.astype('float32') / 255.0  # Normalization
        data.append(class_data)
    return data

def get_datasets(name_files: list[str], path: str):
    data = load_data(name_files, path)
    return (np.concatenate(data, dtype='f').reshape(-1, 28, 28, 1), 
            np.concatenate([np.array([i]*len(_)) for i,_ in enumerate(data)], axis=0))

if not FINAL_EVALUATION_MODE:
    data_train, label_train = get_datasets(class_names, TRAIN_PATH)
    data_val, label_val = get_datasets(class_names, VAL_PATH)

def get_tensor_dataset(x: np.ndarray, y: np.ndarray) -> torch.utils.data.TensorDataset:
    # Змінюємо розмірність з (height, width, channels) на (channels, height, width)
    x = x.transpose(0, 3, 1, 2)
    # Змінюємо тип міток на LongTensor
    y = torch.from_numpy(y).long()
    return torch.utils.data.TensorDataset(torch.from_numpy(x), y)

if not FINAL_EVALUATION_MODE:
    trainloader = torch.utils.data.DataLoader(get_tensor_dataset(data_train, label_train), batch_size=BATCH_SIZE, shuffle=True)
    valloader = torch.utils.data.DataLoader(get_tensor_dataset(data_val, label_val), batch_size=BATCH_SIZE, shuffle=False)

if not FINAL_EVALUATION_MODE:
    plt.figure(figsize=(20, 10))
    for i, name in enumerate(class_names):
        plt.subplot(2, 4, i + 1)
        plt.title(name)
        plt.imshow(data_train[np.where(label_train == i)[0][0]].reshape(28, 28), cmap='gray')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, len(class_names))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, dataloader, valloader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model = model.to(DEVICE)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 1):
            if i % 1000 == 0:
                print(f"  Batch {i}/{len(dataloader)}")
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader)}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Loss: {val_loss/len(valloader)}, Accuracy: {100 * correct / total}')

def evaluate_model(model, dataloader):
    y_true = []
    y_pred = []
    model = model.to(DEVICE)

    for images, labels in tqdm(dataloader, unit=' batch'):
        y_true += labels.detach().tolist()
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        output = model(images)
        y_pred += output.argmax(dim=1).detach().tolist()
    return calculate_accuracy(y_true, y_pred)

if not FINAL_EVALUATION_MODE:
    model_loaded = Net()
    model_loaded.load_state_dict(torch.load(MODEL_PATH))

    start_evaluation = datetime.now()
    acc = evaluate_model(model_loaded, valloader)
    stop_evaluation = datetime.now()

    print(f'\nWynik accuracy: {acc:.3f}\nCzas ewaluacji: {(stop_evaluation - start_evaluation).total_seconds():.2f} sekund')

time_stop = datetime.now()
print(f'Elapsed time: {(time_stop - time_start).total_seconds()/60:.2f} minutes')
