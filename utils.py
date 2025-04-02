import numpy as np

def one_hot_encode(labels, num_classes=10):
    one_hot_matrix = np.zeros((labels.size, num_classes), dtype=int)
    flat_labels = labels.flatten()
    one_hot_matrix[np.arange(flat_labels.size), flat_labels] = 1
    return one_hot_matrix    

class lr_schechlar:
    def __init__(self, lr, gamma):
        self.lr = lr
        self.gamma = gamma

    def step(self):
        self.lr = self.lr * self.gamma
        return self.lr
    
    def get_lr(self):
        return self.lr