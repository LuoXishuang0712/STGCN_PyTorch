import torch
import torch.nn as nn
from torch.utils import data as Data
import d2l.torch as d2l
from stgcn import STGCN
import numpy as np
import pickle

def train(net : nn.Module, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            # print(y_hat.shape)
            l = loss(y_hat, y)
#             print("loss:",l)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None))
                print("in epoch %d, train acc: %.3f" % (epoch, train_acc))
#             break
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # animator.add(epoch + 1, (None, None, test_acc))
        scheduler.step()
        print("epoch %d finish, test acc: %.3f" % (epoch, test_acc))
#         break
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

class dataset(Data.Dataset):
    def __init__(self, data_file, label_file) -> None:
        super(dataset, self).__init__()
        self.data = np.load(data_file)
        label_f = open(label_file, "rb")
        self.label = pickle.load(label_f)[1]
        label_f.close()
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return min(self.data.shape[0], len(self.label))

if __name__ == "__main__":
    batch_size = 16
    lr = 0.01
    num_classes = 60
    def get_dataloader_workers():
        return 8

    train_dataset = dataset(
        "../PaddleVideo_bak/NTU-RGB-D/xsub/train_data.npy", 
        "../PaddleVideo_bak/NTU-RGB-D/xsub/train_label.pkl")
    test_dataset = dataset(
        "../PaddleVideo_bak/NTU-RGB-D/xsub/val_data.npy", 
        "../PaddleVideo_bak/NTU-RGB-D/xsub/val_label.pkl")
    val_dataset = dataset(
        "../PaddleVideo_bak/NTU-RGB-D/xview/val_data.npy", 
        "../PaddleVideo_bak/NTU-RGB-D/xview/val_label.pkl")

    train_data = Data.DataLoader(train_dataset, batch_size, shuffle=True, 
                        num_workers=get_dataloader_workers(), pin_memory=True)
    test_data = Data.DataLoader(test_dataset, batch_size, shuffle=False, 
                        num_workers=get_dataloader_workers(), pin_memory=True)
    val_data = Data.DataLoader(val_dataset, batch_size, shuffle=False, 
                        num_workers=get_dataloader_workers(), pin_memory=True)

    net = STGCN(num_classes, layout='ntu-rgb+d', in_channels=3, edge_importance_weighting=False, device=d2l.try_gpu())
    train(
        net,
        train_data,
        test_data,
        num_epochs=50,
        lr=lr,
        device=d2l.try_gpu()
    )
    torch.save(net.state_dict(), "./first_test.torchmodel")  #save train result
    print(
        evaluate_accuracy_gpu(net, val_data, device=d2l.try_gpu())
    )