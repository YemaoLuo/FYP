import csv
import datetime
import hashlib
import warnings

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from api.difference_api import get_feature_vector
from api.parser_api import *

warnings.filterwarnings("ignore")

CACHE_DIR = "./cache/"


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
    CACHE_DIR = "./cache/"
    CACHE_DIR_2 = "../../tree/"
    with open('./data.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        flag = True
        input_list = []
        output_list = []
        original = ''
        num_rows = sum(1 for _ in reader)
        file.seek(0)

        progress_bar = tqdm(reader, total=num_rows)

        for row in progress_bar:
            if not any(row):
                log_file.write('====================\n')
                flag = False
                continue
            if flag:
                original = row[1]
                flag = False
                continue

            testcase = row[1]
            res = int(row[2])

            cache_filename = hashlib.sha256((original + testcase).encode()).hexdigest() + ".joblib"
            cache_filepath = os.path.join(CACHE_DIR, cache_filename)
            if os.path.exists(cache_filepath):
                vector = joblib.load(cache_filepath)
            else:
                tree_cache_filename_1 = hashlib.sha256(original.encode()).hexdigest() + ".joblib"
                tree_cache_filepath_1 = os.path.join(CACHE_DIR_2, tree_cache_filename_1)
                if os.path.exists(tree_cache_filepath_1):
                    con_tree_1, dep_tree_1 = joblib.load(tree_cache_filepath_1)
                else:
                    con_tree_1 = get_constituency_tree(original)
                    dep_tree_1 = get_dependency_tree(original)
                    joblib.dump((con_tree_1, dep_tree_1), tree_cache_filepath_1)

                tree_cache_filename_2 = hashlib.sha256(testcase.encode()).hexdigest() + ".joblib"
                tree_cache_filepath_2 = os.path.join(CACHE_DIR_2, tree_cache_filename_2)
                if os.path.exists(tree_cache_filepath_2):
                    con_tree_2, dep_tree_2 = joblib.load(tree_cache_filepath_2)
                else:
                    con_tree_2 = get_constituency_tree(testcase)
                    dep_tree_2 = get_dependency_tree(testcase)
                    joblib.dump((con_tree_2, dep_tree_2), tree_cache_filepath_2)

                vector = get_feature_vector(con_tree_1, con_tree_2, dep_tree_1, dep_tree_2)
                joblib.dump(vector, cache_filepath)

            input_tensor = torch.tensor(vector, dtype=torch.float32)
            output_tensor = torch.tensor([res], dtype=torch.float32)

            input_list.append(input_tensor)
            output_list.append(output_tensor)

    return input_list, output_list


if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_filename = "./log/log_1_" + current_time + ".log"
    best_model_filename = "./log/best_model_1_" + current_time + ".pt"

    best_accuracy = 0.0

    with open(log_filename, 'w') as log_file:
        print('Preparing data...')
        input_list, output_list = prepare_data()

        inputs = torch.stack(input_list)
        outputs = torch.stack(output_list).squeeze()

        dataset = TensorDataset(inputs, outputs)

        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                 [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_size = inputs.shape[1]
        model = NeuralNet(input_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        best_model = None

        num_epochs = 1000
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    for i in range(len(outputs)):
                        if outputs[i] >= 0.5 and labels[i] == 1 or outputs[i] < 0.5 and labels[i] == 0:
                            correct += 1
                        total += 1

            val_accuracy = 100 * correct / total

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model

        if best_model is not None:
            torch.save(best_model, best_model_filename)

        print('Training finished.')
        print(f'Best validation accuracy: {best_accuracy}%')
        log_file.write(f'Best validation accuracy: {best_accuracy}%\n')

        print('Testing...')
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = best_model(inputs)
                for i in range(len(outputs)):
                    if outputs[i] >= 0.5 and labels[i] == 1 or outputs[i] < 0.5 and labels[i] == 0:
                        test_correct += 1
                    test_total += 1

        test_accuracy = 100 * test_correct / test_total
        print(f'Test accuracy: {test_accuracy}%')
        log_file.write(f'Test accuracy: {test_accuracy}%\n')
