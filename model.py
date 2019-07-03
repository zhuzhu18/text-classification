import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, vocab_size, embedded_size, num_hiddens, num_layers, num_classes, **kwargs):
        super(Net, self).__init__()
        self.vocab_size = vocab_size
        self.embedded_size = embedded_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.embedding = nn.Embedding(self.vocab_size, embedded_size)
        self.encoder = nn.LSTM(input_size=embedded_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers)
        self.decoder = nn.Linear(num_hiddens*2, self.num_classes)

    def forward(self, x):
        embeddings = self.embedding(x)
        states, hidden = self.encoder(embeddings.permute([1,0,2]))
        encodding = torch.cat([states[0], states[-1]], dim=1)
        outs = self.decoder(encodding)

        return outs

