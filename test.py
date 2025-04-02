import numpy as np
from model import neuralNetwork#, neuralNetwork_v2
from dataset import Dataset, DataLoader
from tqdm import tqdm
from utils import one_hot_encode

if __name__ == '__main__':
    model_path = './model.pkl'
    print(f"Preparing test dataset")
    dataset = Dataset.load("dataset_cifar10")
    test_set = dataset.test
    x_test = test_set['data']
    y_test = one_hot_encode(test_set['label'])
    y_label = test_set['label']

    batch_size = 100
    test_loader = DataLoader(x_test, y_test, batch_size=batch_size, shuffle=False)

    input_dim = x_test.shape[1]
    output_dim = y_test.shape[1]
    hidden_dim = 1536
    model = neuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load(model_path)

    loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True, colour='yellow')
    label_pred = []
    for batch_idx, (batch_x, batch_y) in loop:
        y_pred = model.forward(batch_x)
        label = np.argmax(y_pred, axis=1)
        label_pred.append(label)
    label_pred = np.concatenate(label_pred).reshape(-1, 1)
    accuracy = np.sum(y_label == label_pred) / len(y_label)
    print(f"Accuracy over test set is {100 * accuracy:.2f}%")