from model import Net
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import torch.nn.functional as F


#Parsing Arguments
parser = argparse.ArgumentParser(description = "Emotion classifier CNN")
parser.add_argument('--epochs' , type = int , default = 3)
parser.add_argument('--batch_size_train' , type = int , default = 64)
parser.add_argument('--batch_size_test' , type = int , default = 1000)
parser.add_argument('--learning_rate' , type = float , default = 0.01)
parser.add_argument('--momentum' , type = float , default = 0.5)
parser.add_argument('--log_interval' , type = int , default = 10)
parser.add_argument('--output_path' , type = str , default = "/Users/etiennemontenegro/Desktop/MNIST_CLASSIFIER/results/")
parser.add_argument('--mnist' , type = bool , default = True)
args = parser.parse_args()
#define hyperparameters
n_epochs = args.epochs
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
learning_rate = args.learning_rate
momentum = args.momentum
log_interval = args.log_interval
output_path = args.output_path
mnist = args.mnist
#define random seed manualy for a repeatable experiment
random_seed = 1
torch.backends.cudnn.enabled = False #disable nondeterministics algorithms used by cuDNN
torch.manual_seed(random_seed)


#load the data
if mnist == True:
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("", train = True, download = True,
                                          transform = torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                              (0.1307,),(0.3081,))
                                          ])), batch_size = batch_size_train, shuffle = True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("",train = False, download = True,
                                                                    transform = torchvision.transforms.Compose([
                                                                                torchvision.transforms.ToTensor(),
                                                                                torchvision.transforms.Normalize(
                                                                                                                 (0.1307,),(0.3081,))
                                                                    ])), batch_size = batch_size_test, shuffle = True)
else: 
    print("potato")                                       

#initialize the network and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = Net()
optimizer = optim.SGD(network.parameters(),lr = learning_rate, momentum = momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0 :
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'
                  .format(epoch,batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            #changer output path pour lib 
            torch.save(network.state_dict() , output_path + "model.pth")
            torch.save(optimizer.state_dict(), output_path + "optimizer.pth")




def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( test_loss, correct, 
                                                                             len(test_loader.dataset), 
                                                                            100. * correct / len(test_loader.dataset)))


#test model before training
test()

#training loop
for epoch in range(1, n_epochs +1):
    train(epoch)
    test()

#plot loss
fig = plt.figure()
plt.plot(train_counter , train_losses, color = 'blue')
plt.scatter(test_counter, test_losses, color = 'red')
plt.legend(['Train Loss' , 'Test Loss'],loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig