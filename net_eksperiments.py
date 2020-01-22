import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as mp
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500

# set up net generation for 3 layered networks
SIGMOID = "SIG"
SOFTMAX = "SOFT"
RELU = "RELU"

class NetworkLayers3(nn.Module):
    # l1 hidden layer 1
    def __init__(self, l1, activation_functions):
        super(NetworkLayers3, self).__init__()
        self.activation_functions = activation_functions
        self.fc1 = nn.Linear(784, l1)
        self.fc2 = nn.Linear(l1, 10)

    def forward(self, x):
        if self.activation_functions[0] == SIGMOID:
            x = torch.sigmoid(self.fc1(x))
        elif self.activation_functions[0] == SOFTMAX:
            x = F.softmax(self.fc1(x), 1)
        elif self.activation_functions[0] == RELU:
            x = F.relu(self.fc1(x))

        if self.activation_functions[1] == SIGMOID:
            x = torch.sigmoid(self.fc2(x))
        elif self.activation_functions[1] == SOFTMAX:
            x = F.softmax(self.fc2(x), 1)
        elif self.activation_functions[1] == RELU:
            x = F.relu(self.fc2(x))

        return x

def vectorize_labels(batch_size, labels):
    vector_labels = torch.zeros(batch_size, 10)
    for i in range(batch_size):
        vector_labels[i][labels[i].item()] = 1
    return vector_labels

def evaluate_network(net, test_amount, data_loader):
    correct = 0
    i = 0
    for images, labels in data_loader:
        images = images.view(images.shape[0], -1)
        vector_prediction = net(images)
        try:
            if(vector_prediction[0] == vector_prediction[0].max()).nonzero().item() == labels[0].item():
                correct += 1
        except:
            pass
            #print(vector_prediction)
        i += 1
        if i == test_amount:
            break

    return correct/test_amount

batch_size = 100
save = True
data_set = "digits"

#criterion = nn.BCEWithLogitsLoss()
#crt = "BCEWLL"
criterion = nn.MSELoss()
criterion_name = "MSEL"
net_count = 4

learning_rates = [0.1, 1, 3, 6, 9, 12]

#load the training and testing data
transform = transforms.Compose([transforms.ToTensor()])

if data_set == "digits":
    training_data = torchvision.datasets.MNIST(root="./../data", train=True, transform=transform, download=True)
    testing_data = torchvision.datasets.MNIST(root="./../data", train=False, transform=transform, download=True)

if data_set == "fashion":
    training_data = torchvision.datasets.FashionMNIST(root="./../data", train=True, transform=transform, download=True)
    testing_data = torchvision.datasets.FashionMNIST(root="./../data", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)

for ggg in [[RELU, SIGMOID], [RELU, SOFTMAX], [SOFTMAX, SIGMOID], [SOFTMAX, RELU]]:
    actf1 = ggg[0]
    actf2 = ggg[1]

    training_epoch_amounth = 40

    # iterrate over set network sizes
    for network_middle_layer_size in [10, 20, 40, 80, 160]:

        # set up plotting the data and data saveing
        plt.title("net:"+actf1+","+actf2+" ml:"+str(network_middle_layer_size)+" costf:"+criterion_name+" d:"+data_set)
        plt.xlabel("epochs")
        plt.ylabel("networks accuricity")

        data_save_file = open("net_size_and_lr/net:"+actf1+","+actf2+" ml:"+str(network_middle_layer_size)+" costf:"+criterion_name+" d:"+data_set, "w")

        # interate over the chosen test learnning rates
        for learning_rate in [0.125, 0.5, 1, 2, 8]:
            # create the network
            net = NetworkLayers3(network_middle_layer_size, [actf1, actf2])
            optimizer = optim.SGD(net.parameters(), lr = learning_rate) # train the network with stocastic gradient decent

            # reset data collectors for plotting
            x_data = []
            y_data = []

            # collect data for data saveing
            data_save_file.write("net:"+actf1+","+actf2+" ml:"+str(network_middle_layer_size)+" lr:"+str(learning_rate)+" costf:"+criterion_name+" d:"+data_set+"\n")

            # see the progress while training
            print("training net:"+actf1+","+actf2+" ml:"+str(network_middle_layer_size)+" lr:"+str(learning_rate)+" costf:"+criterion_name+" d:"+data_set)

            # train the created network for a set amounth of epochs
            for current_epoch in range(training_epoch_amounth):

                # collect data for plot
                x_data.append(current_epoch)
                y_data.append(evaluate_network(net, 100, test_loader))

                # collect data for data saveing
                data_save_file.write("e:"+str(x_data[-1])+" "+str(y_data[-1])+"\n")

                # see the progress while training
                print("epoch", x_data[-1], y_data[-1])

                for images, labels in train_loader: # get the next batch of data for training
                    # set up data for the network
                    images = images.view(images.shape[0], -1)
                    vector_labels = vectorize_labels(batch_size, labels)

                    optimizer.zero_grad()
                    prediction = net(images)
                    loss = criterion(prediction, vector_labels)

                    loss.backward()
                    optimizer.step()

            # collect data for plot
            x_data.append(current_epoch+1)
            y_data.append(evaluate_network(net, 100, test_loader))

            # collect data for data saveing
            data_save_file.write("e:"+str(x_data[-1])+" "+str(y_data[-1])+"\n")

            # see the progress while training
            print("epoch", x_data[-1], y_data[-1])# see the progress while training

            plt.plot(x_data, y_data, label="learning rate:"+str(learning_rate)+"\n")

        plt.legend()
        plt.savefig("net_size_and_lr/plots/net:"+actf1+","+actf2+" ml:"+str(network_middle_layer_size)+" costf:"+criterion_name+" d:"+data_set+".png")
        plt.clf()

        data_save_file.close()
