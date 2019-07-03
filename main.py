from config import Configure
from model import Net
from data_process import data_loader
import torch
import torch.nn as nn


config = Configure()
args = config.get_args()

train_set, val_set, num_total_words = data_loader(args)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
net = Net(num_total_words*args.max_features, args.embedding_size,
          args.num_hiddens, args.num_layers, num_classes=len(args.catogories))
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)


for epoch in range(args.num_epochs):
    train_set, val_set, num_total_words = data_loader(args)
    train_loss, val_loss = 0., 0.
    train_acc, val_acc = 0., 0.
    num_correct = 0
    for step, (feature, label) in enumerate(train_set):
        feature = torch.from_numpy(feature).to(device)
        label = torch.from_numpy(label).to(device)

        net.zero_grad()
        scores = net(feature)

        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.item()
        num_correct += torch.sum(torch.max(scores, 1)[1] == label).item()

        if step % 20 == 19:
            print('Training in epoch {}, Loss: {:.3f}, Accuracy: {:.3f}'.format(epoch, train_loss / (step+1), num_correct / ((step+1)*label.size(0))))

