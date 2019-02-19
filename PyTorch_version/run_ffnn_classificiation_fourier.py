import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import sklearn.metrics

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


class Net(nn.Module):
	

	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(192,96)
		self.fc2 = nn.Linear(96,48)
		self.fc3 = nn.Linear(48,24)
		self.fc4 = nn.Linear(24,2)

	def forward(self, x):
		# The first convolution takes data from shape (1,27,48) to (12,27,48)
		# The first pooling takes data from shape (12,27,48) to (12,9,16)
		x = F.relu(self.fc1(x))
		# The second convolution takes data from shape (12,9,16) to i(24,9,16)
		# The second pooling take data from shape (24,9,16) to (24,6,12)
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		return F.softmax(x)

# Number of training examples
numb_train = 5000
# Number of testing examples
numb_test = 1000
# Batch size
batch = 30

seed = 42
np.random.seed(seed)

# Load dataset
series_train = np.loadtxt(open("fourier_coefficients_train.csv", "rb"), delimiter=",")
labels_train = np.loadtxt(open("labels_train_revised.csv","rb"), delimiter=",")
series_test = np.loadtxt(open("fourier_coefficients_test.csv", "rb"), delimiter=",")
labels_test = np.loadtxt(open("labels_test_revised.csv", "rb"), delimiter=",")

# Change numpy arrays of PyTorch tensors
series_train = torch.Tensor(np.transpose(series_train))
labels_train = torch.Tensor(labels_train)
series_test = torch.Tensor(np.transpose(series_test))
labels_test = torch.Tensor(labels_test)

print(np.shape(series_train))
print(np.shape(labels_train))
print(np.shape(series_test))
print(np.shape(labels_test))

# Unite features and labels
training_set = torch.utils.data.TensorDataset(series_train,labels_train)
test_set = torch.utils.data.TensorDataset(series_test,labels_test)

# Create data loader
train_loader = torch.utils.data.DataLoader(training_set,batch_size=30, shuffle=True,num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=30, shuffle=True, num_workers=4)

classes = (0, 1)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)

for epoch in range(50):  # loop over the dataset multiple times
	running_loss = 0.0
	for i, data in enumerate(train_loader,0):
        	# get the inputs
		inputs, labels = data
		labels = labels.long()
        	# zero the parameter gradients
		optimizer.zero_grad()
        	# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

        	# print statistics
		running_loss += loss.item()
		if i % 50 == 49:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')

dataiter = iter(test_loader)
series, labels = dataiter.next()

# print series
x = range(batch)
plt.plot(x,series.numpy())
indices = labels.numpy()
print(int(indices[0]))
print('GroundTruth: ', ' '.join('%5s' % classes[int(indices[j])] for j in range(4)))

outputs = net(series)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# Create variables to store all predictions and labels
labels_total = np.zeros((numb_test))
predicted_total = np.zeros((numb_test))

correct = 0
total = 0
batch_numb = 0
with torch.no_grad():
	for data in test_loader:
		images, labels = data
		labels = labels.long()
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		predicted_total[batch*batch_numb:batch*(batch_numb+1)] = predicted.numpy()
		labels_total[batch*batch_numb:batch*(batch_numb+1)] = labels.numpy()
		batch_numb = batch_numb +1

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# Create confusion matrix
confus = sklearn.metrics.confusion_matrix(labels_total,predicted_total)

print(confus)
