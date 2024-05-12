import torch.nn as nn

class AlexNet(nn.Module):
	def __init__(self, input_channel, n_classes=2):
		super().__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(input_channel, 96, kernel_size=11, stride=4, padding=3),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2))
		self.conv2 = nn.Sequential(
			nn.Conv2d(96, 256, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2))
		self.conv3 = nn.Sequential(
			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Flatten())
		self.fc = nn.Sequential(
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(4096, n_classes))

		self.conv1.apply(self.init_weights)
		self.conv2.apply(self.init_weights)
		self.conv3.apply(self.init_weights)
		self.fc.apply(self.init_weights)

	def init_weights(self, layer):
		if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
			nn.init.xavier_uniform_(layer.weight)

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.fc(out)

		return out