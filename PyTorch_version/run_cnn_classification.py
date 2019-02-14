import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


class Net(nn.Module):
	

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1,12, kernel_size=3, stride=1,padding=1)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.conv2 = nn.Conv2d(12, 24, kernel_size=3,stride=1,padding=1)
		self.fc1 = nn.Linear(24*6*12, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 2)

	def forward(self, x):
		# The first convolution takes data from shape (1,27,48) to (12,27,48)
		# The first pooling takes data from shape (12,27,48) to (12,9,16)
		x = self.pool(F.relu(self.conv1(x)))
		# The second convolution takes data from shape (12,9,16) to (24,9,16)
		# The second pooling take data from shape (24,9,16) to (24,6,12)
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,24*6*12 )
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.softmax(x)

# Number of training examples
numb_train = 4000
# Number of testing examples
numb_test = 1000

seed = 42
np.random.seed(seed)

# Load dataset
images_train = np.loadtxt(open("temps_pictures_train.csv", "rb"), delimiter=",")
labels_train = np.loadtxt(open("labels_train_revised.csv","rb"), delimiter=",")
images_test = np.loadtxt(open("temps_pictures_test.csv", "rb"), delimiter=",")
labels_test = np.loadtxt(open("labels_test_revised.csv", "rb"), delimiter=",")

# Create new numpy array to store reshaped images
images_train_reshape = np.zeros((numb_train,1,27,48))
images_test_reshape = np.zeros((numb_test,1,27,48))

labels_train_reshape = np.zeros((numb_train,2))

# Reshape training images
for i in range(numb_train):
	images_train_reshape[i,0,:,:] = np.reshape(images_train[27*i:27*(i+1),:],(1,27,48))
	if (labels_train[i] == 0):
		labels_train_reshape[i,0] = 1
	else:
		labels_train_reshape[i,1] = 1

# Reshape test images
for i in range(numb_test):
	images_test_reshape[i,0,:,:] = np.reshape(images_test[27*i:27*(i+1),:],(1,27,48))

# Change numpy arrays of PyTorch tensors
images_train = torch.Tensor(images_train_reshape)
labels_train = torch.Tensor(labels_train)
images_test = torch.Tensor(images_test_reshape)
labels_test = torch.Tensor(labels_test)

# Unite features and labels
training_set = torch.utils.data.TensorDataset(images_train,labels_train)
test_set = torch.utils.data.TensorDataset(images_test,labels_test)

# Create data loader
train_loader = torch.utils.data.DataLoader(training_set,batch_size=30, shuffle=True,num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=30, shuffle=True, num_workers=4)

classes = (0, 1)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

for epoch in range(500):  # loop over the dataset multiple times
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
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
indices = labels.numpy()
print(int(indices[0]))
print('GroundTruth: ', ' '.join('%5s' % classes[int(indices[j])] for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
	for data in test_loader:
		images, labels = data
		labels = labels.long()
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
