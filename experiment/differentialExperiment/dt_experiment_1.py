import datetime
import hashlib
import warnings

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from api.difference_api import get_feature_vector
from api.parser_api import *

warnings.filterwarnings("ignore")

CACHE_DIR = "./cache/"
LOG_DIR = "./log/"


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


def prepare_data():
    CACHE_DIR_2 = "../../tree/"
    df = pd.read_excel('./data.xlsx')
    excel_data = df.iloc[:, 0:4].values.tolist()
    vector_data = []
    for row in tqdm(excel_data):
        sentence1 = row[1]
        sentence2 = row[2]
        label = row[3]

        cache_filename = hashlib.sha256((sentence1 + sentence2).encode()).hexdigest() + ".joblib"
        cache_filepath = os.path.join(CACHE_DIR, cache_filename)

        if os.path.exists(cache_filepath):
            vector = joblib.load(cache_filepath)
        else:
            tree_cache_filename_1 = hashlib.sha256(sentence1.encode()).hexdigest() + ".joblib"
            tree_cache_filepath_1 = os.path.join(CACHE_DIR_2, tree_cache_filename_1)
            if os.path.exists(tree_cache_filepath_1):
                con_tree_1, dep_tree_1 = joblib.load(tree_cache_filepath_1)
            else:
                con_tree_1 = get_constituency_tree(sentence1)
                dep_tree_1 = get_dependency_tree(sentence1)
                joblib.dump((con_tree_1, dep_tree_1), tree_cache_filepath_1)

            tree_cache_filename_2 = hashlib.sha256(sentence2.encode()).hexdigest() + ".joblib"
            tree_cache_filepath_2 = os.path.join(CACHE_DIR_2, tree_cache_filename_2)
            if os.path.exists(tree_cache_filepath_2):
                con_tree_2, dep_tree_2 = joblib.load(tree_cache_filepath_2)
            else:
                con_tree_2 = get_constituency_tree(sentence2)
                dep_tree_2 = get_dependency_tree(sentence2)
                joblib.dump((con_tree_2, dep_tree_2), tree_cache_filepath_2)

            vector = get_feature_vector(con_tree_1, con_tree_2, dep_tree_1, dep_tree_2)
            joblib.dump(vector, cache_filepath)

        vector_data.append([vector, label])

    return vector_data


if __name__ == '__main__':
    print('Prepare data...')
    data = prepare_data()

    print('Splitting dataset...')
    inputs = torch.tensor([item[0] for item in data], dtype=torch.float32)
    labels = torch.tensor([item[1] for item in data], dtype=torch.float32)

    dataset = TensorDataset(inputs, labels)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print('Training...')
    input_size = inputs.shape[1]
    model = NeuralNet(input_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 1000
    best_accuracy = 0.0
    best_model = model

    log_filename = f"{LOG_DIR}log_1_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    with open(log_filename, 'w') as log_file:
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).int()
                    total += labels.size(0)
                    correct += (predicted.squeeze() == labels).sum().item()

            val_accuracy = 100 * correct / total

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model

        print('Training finished.')

        if best_model is not None:
            best_model_filename = f"{LOG_DIR}best_model_1_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
            torch.save(best_model, best_model_filename)
            print(f'Best model saved as {best_model_filename}')
        print(f'Best validation accuracy: {best_accuracy}%')
        log_file.write(f'Best validation accuracy: {best_accuracy}%\n')

        print('Testing...')
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = best_model(inputs)
                predicted = (outputs > 0.5).int()
                test_total += labels.size(0)
                test_correct += (predicted.squeeze() == labels).sum().item()

        test_accuracy = 100 * test_correct / test_total
        print(f'Test accuracy: {test_accuracy}%')
        log_file.write(f'Test accuracy: {test_accuracy}%\n')
