import torch.nn as nn  
import torch  

class SimpleCNN(nn.Module):  
    def __init__(self, num_classes=10):  
        super(SimpleCNN, self).__init__()  
        self.conv_layers = nn.Sequential(  
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),  # 11x11 -> 5x5  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2)  # 5x5 -> 2x2  
        )  
        self.fc_layers = nn.Sequential(  
            nn.Linear(64 * 2 * 2, 128),  # Matches the output of Conv layers  
            nn.ReLU(),  
            nn.Linear(128, num_classes)  
        )  

    def forward(self, x):  
        if x.dtype != torch.float32:  
            x = x.to(torch.float32)  
        x = self.conv_layers(x)  
        x = x.view(x.size(0), -1)  # Flatten  
        x = self.fc_layers(x)  
        return x  