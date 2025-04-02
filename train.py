import numpy as np
from model import neuralNetwork#, neuralNetwork_v2
from dataset import Dataset, DataLoader
from utils import one_hot_encode, lr_schechlar
from sklearn.model_selection import train_test_split
from loss_function import cross_entropy, validation
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description='Hyperparameters for training neural network')
parser.add_argument("--learning_rate", type=int, help='Learning rate for training', default=1e-4)
parser.add_argument("--activate_func", type=str, help='Activation function for neural network', default='relu')
parser.add_argument("--hidden_dim", type=int, help='Hidden layer size', default=2304)
parser.add_argument("--l2_lambda", type=int, help='L2 strength for loss function', default=1e-6)
parser.add_argument("--batch_size", type=int, help='Batch size for training', default=100)

args = parser.parse_args()

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.learning_rate = lr.get_lr()

    def step(self):
        # for param in self.params:
        self.params['input_layer_weights'] -= self.learning_rate * self.params['input_layer_weights_grad']
        self.params['input_layer_bias'] -= self.learning_rate * self.params['input_layer_bias_grad']
        self.params['hidden_layer_weights'] -= self.learning_rate * self.params['hidden_layer_weights_grad']
        self.params['hidden_layer_bias'] -= self.learning_rate * self.params['hidden_layer_bias_grad']
        self.params['output_layer_weights'] -= self.learning_rate * self.params['output_layer_weights_grad']
        self.params['output_layer_bias'] -= self.learning_rate * self.params['output_layer_bias_grad']
    
    def zero_grad(self):
        # for param in self.params:
        self.params['input_layer_weights_grad'] = 0
        self.params['input_layer_bias_grad'] = 0
        self.params['hidden_layer_weights_grad'] = 0
        self.params['hidden_layer_bias_grad'] = 0
        self.params['output_layer_weights_grad'] = 0
        self.params['output_layer_bias_grad'] = 0

if __name__=="__main__":
    # hyperparameters
    nb_epoch = 50
    learning_rate = args.learning_rate  #1e-4
    activation = args.activate_func     #'sigmoid'
    l2_lambda = args.l2_lambda          #1e-4
    hidden_dim = args.hidden_dim        #1736
    batch_size = args.batch_size        #100

    print(f"Hyperparameters are: nb_epoch {nb_epoch}, learning_rate {learning_rate}, activation function {activation}, l2_lambda {l2_lambda}, hiddent_dim {hidden_dim}, batch_size {batch_size}")

    # prepare dataset and do some preprocessing
    print(f"Preparing dataset")
    dataset = Dataset.load("dataset_cifar10")
    train_set = dataset.train
    test_set = dataset.test
    
    x_train, x_val, y_train, y_val = train_test_split(train_set['data'], train_set['label'], \
                                                      random_state=42, test_size=0.2, shuffle=True, stratify=train_set['label'])
    print(f"Done")
    
    print(f"Convering labels to one-hot-encode")
    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val)
    test_set['label'] = one_hot_encode(test_set['label'])
    x_test = test_set['data']
    y_test = test_set['label']

    train_loader = DataLoader(x_train, y_train, batch_size=batch_size)
    val_loader = DataLoader(x_val, y_val, batch_size=batch_size)
    test_loader = DataLoader(x_test, y_test, batch_size=batch_size)

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    # print(f"x_train {x_train.shape}, y_train {y_train.shape}, x_val {x_val.shape}, y_val {y_val.shape}, x_test {x_test.shape}, y_test {y_test.shape}")

    
    # training set
    lr_schechlar = lr_schechlar(learning_rate, gamma=0.98)
    Model = neuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, activate=activation)
    Model.save("Initial_model.pkl")
    optim = SGD(Model.parameters, lr=lr_schechlar)
    train_loss = []
    validation_loss = []
    validation_accuracy = []
    best_accuracy_val = 0

    # start training
    for i in range(nb_epoch):
        lr = lr_schechlar.get_lr()
        average_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, colour='green')
        for batch_idx, (batch_x, batch_y) in loop:
            y_pred = Model.forward(batch_x)
            loss = cross_entropy(y_pred, batch_y, Model, l2_lambda=l2_lambda)
            average_loss += loss
            Model.backward(batch_y, l2_lambda, learning_rate=lr)
            optim.step()
            optim.zero_grad()

            loop.set_description(f'Train Epoch: [{i+1}/{nb_epoch}]')
            loop.set_postfix({'batch loss': loss, 'learning rate': lr})
        
        average_loss = average_loss / train_loader.__len__()
        train_loss.append(average_loss)
        print(f"Train Epoch {i}: average loss={average_loss}, learning rate={lr}")
        
        
        print(f"Evaluating with validation dataset...")
        val_loss, val_accuracy = validation(Model, val_dataloader=val_loader)
        validation_loss.append(val_loss)
        validation_accuracy.append(val_accuracy)
        print(f"Evaluating done, validation loss = {val_loss}, validation accuracy = {100*val_accuracy:.2f}%")
        
        # save best model according to validation
        if val_accuracy > best_accuracy_val:
            Model.save("Model.pkl")
            best_accuracy_val = val_accuracy
        
        # learning rate decay
        lr_schechlar.step()
    print(f"During training process, best accuracy at validation set is {100*best_accuracy_val:.2f}")
    # plot train loss and validation loss
    epoch = np.arange(1, len(train_loss)+1)
    plt.figure()
    plt.plot(epoch, train_loss, label='Training loss', color='green')
    plt.plot(epoch, validation_loss, label='Validation loss', color='blue')
    plt.title('Training and validation loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig("Training_validation_loss.jpg")

    plt.figure()
    validation_accuracy = [i * 100 for i in validation_accuracy]
    plt.plot(epoch, validation_accuracy, label='Validation accuracy', color='blue')
    plt.title('Training and validation loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig("Training_validation_accuracy.jpg")

