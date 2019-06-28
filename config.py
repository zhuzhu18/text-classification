import argparse

class Configure:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def get_args(self):

        # data file
        self.parser.add_argument('--train_file', type=str, default='./data/NLP_TC/traindata.txt', help='Training file')
        self.parser.add_argument('--dev_file', type=str, default='./data/NLP_TC/devdata.txt', help='Development file')
        self.parser.add_argument('--testfile', type=str, default='./data/NLP_TC/testdata.txt', help='Test file')

        # model parameters
        self.parser.add_argument('--embedding_size', type=int, default=256, help='Default embedding size if embedding_file is not given')

        return self.parser.parse_args()
