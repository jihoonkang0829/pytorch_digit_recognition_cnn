import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.cnn2 = nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(196, 11)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out

