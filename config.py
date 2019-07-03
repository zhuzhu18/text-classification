import argparse

class Configure:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def get_args(self):

        # data file
        self.parser.add_argument('--train_file', type=str, default='./data/NLP_TC/traindata.txt', help='Training file')
        self.parser.add_argument('--dev_file', type=str, default='./data/NLP_TC/devdata.txt', help='Development file')
        self.parser.add_argument('--test_file', type=str, default='./data/NLP_TC/testdata.txt', help='Test file')

        # model parameters
        self.parser.add_argument('--embedding_size', type=int, default=256, help='Default embedding size if embedding_file is not given')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--num_epochs', type=int, default=20, help='training epochs')
        self.parser.add_argument('--max_features', type=int, default=300, help='length of each sentence')
        self.parser.add_argument('--num_hiddens', type=int, default=300, help='number of hidden cells')
        self.parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')


        return self.parser.parse_args()