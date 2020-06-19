from model import ECGResNet
from dataset import EmotionDataset_v2 as EmotionDataset
import torch
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import torch.nn.functional as F
from lib import biodata


# Parsing Arguments
parser = argparse.ArgumentParser(description="Emotion classifier CNN")
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size_train', type=int, default=128)
parser.add_argument('--batch_size_test', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--path_to_dataset', type=str, default=None)
parser.add_argument('--output_path', type=str, default="/Users/etiennemontenegro/Desktop/MNIST_CLASSIFIER/results/")
parser.add_argument('--optim', type=str, default="adam", choices=('adam', 'sgd', 'sls'))
parser.add_argument('--downsampling', type=int, default=1)
parser.add_argument('--overlap', type=float, default=0.)

args = parser.parse_args()
# define hyperparameters
n_epochs = args.epochs
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
learning_rate = args.learning_rate
momentum = args.momentum
log_interval = args.log_interval
output_path = args.output_path
# define random seed manualy for a repeatable experiment
random_seed = 1
torch.backends.cudnn.enabled = False  # disable nondeterministics algorithms used by cuDNN
torch.manual_seed(random_seed)


class Preprocessing:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, x):
        #x = x[:, 0].view(-1, 1)
        x = biodata.enveloppe_filter(x)
        if self.normalize:
            std, mean = torch.std_mean(x, dim=0, keepdim=True)
            x = (x-mean)/std
        return x


preprocessing = Preprocessing()

trainset = EmotionDataset(args.path_to_dataset, train=True, preprocessing=preprocessing, overlap=args.overlap, downsampling=args.downsampling)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

testset = EmotionDataset(args.path_to_dataset, train=False, preprocessing=preprocessing, overlap=args.overlap, downsampling=args.downsampling)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)                                    

# initialize the network and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = ECGResNet(2, trainset.num_cat).to(device)

if args.optim == "sgd":
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
elif args.optim == "adam":
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
elif args.optim == "sls":
    from lib.sls import Sls
    n_batches_per_epoch = len(trainset)/float(batch_size_train)
    optimizer = Sls(network.parameters(), n_batches_per_epoch=n_batches_per_epoch)
else:
    raise ValueError()

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    correct = 0
    num_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        output = network(data)
        loss = F.nll_loss(output, target)

        if args.optim == "sls":
            def closure():
                output = network(data)
                loss = F.nll_loss(output, target)
                return loss
            optimizer.step(closure)
        else:
            loss.backward()
            optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        num_samples += len(data)
        
        if batch_idx % log_interval == 0 :
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}, Accuracy: {}/{} ({:.0f}%)'
                  .format(epoch, batch_idx * len(data), len(trainset),
                          100. * batch_idx / len(train_loader), loss.item(),
                          correct, num_samples, 100. * correct / num_samples))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(trainset)))
            # changer output path pour lib 
            torch.save(network.state_dict(), output_path + "model.pth")
            torch.save(optimizer.state_dict(), output_path + "optimizer.pth")


def test():
    network.eval()
    test_loss = 0
    correct = 0
    count_per_class = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            count_per_class += torch.bincount(target.view(-1), minlength=testset.num_cat)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(count_per_class)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, 
                                                                              len(test_loader.dataset), 
                                                                              100. * correct / len(test_loader.dataset)))


# test model before training
test()

# training loop
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# plot loss
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')