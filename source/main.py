import torch.optim
import torch.nn as nn
import argparse
from models import ProteinBERT
from collections import namedtuple

Hyperparameters = namedtuple('Hyperparameters', ['lr'])

# Constants
_NUM_EPOCHS = 100

def train(hyperparams: Hyperparameters, model: nn.Module):
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)
    for epoch in range(_NUM_EPOCHS):
        pass

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--learning_rate", required=True, type=float, help="Learning rate for Adam optimization")
    arg_parser.add_argument("--dataset", required=True, help="Filepath to dataset to use")
    arguments = arg_parser.parse_args()
    train(Hyperparameters(arguments.learning_rate), ProteinBERT())


if __name__ == '__main__':
    main()