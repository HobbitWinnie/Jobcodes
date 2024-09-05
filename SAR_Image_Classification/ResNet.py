import torch  
import torch.nn as nn  
import torchvision.transforms as transforms  
import torchvision.models as models  
from torch.utils.data import DataLoader, Dataset  
from sklearn.model_selection import train_test_split  
from PIL import Image  
import numpy as np  

class PolSARClassifier:  
    def __init__(self, num_classes, learning_rate=0.001, batch_size=32, num_epochs=10):  
        self.num_classes = num_classes  
        self.learning_rate = learning_rate  
        self.batch_size = batch_size  
        self.num_epochs = num_epochs  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model = self._initialize_model()  
        self.criterion = nn.CrossEntropyLoss()  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  

    def _initialize_model(self):  
        # Load a pre-trained ResNet model  
        model = models.resnet50(pretrained=True)  
        # Modify the final layer to match the number of PolSAR classes  
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)  
        return model.to(self.device)  

    def _prepare_data(self, images, labels):  
        # Define transformations for the images  
        transform = transforms.Compose([  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])  
        # Create datasets and dataloaders  
        train_images, test_images, train_labels, test_labels = train_test_split(  
            images, labels, test_size=0.2, random_state=42  
        )  
        train_dataset = PolSARDataset(train_images, train_labels, transform=transform)  
        test_dataset = PolSARDataset(test_images, test_labels, transform=transform)  
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)  
        return train_loader, test_loader  

    def train(self, images, labels):  
        train_loader, _ = self._prepare_data(images, labels)  
        for epoch in range(self.num_epochs):  
            self.model.train()  
            running_loss = 0.0  
            for images, labels in train_loader:  
                images, labels = images.to(self.device), labels.to(self.device)  
                self.optimizer.zero_grad()  
                outputs = self.model(images)  
                loss = self.criterion(outputs, labels)  
                loss.backward()  
                self.optimizer.step()  
                running_loss += loss.item()  
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(train_loader):.4f}')  

    def evaluate(self, images, labels):  
        _, test_loader = self._prepare_data(images, labels)  
        self.model.eval()  
        correct = 0  
        total = 0  
        with torch.no_grad():  
            for images, labels in test_loader:  
                images, labels = images.to(self.device), labels.to(self.device)  
                outputs = self.model(images)  
                _, predicted = torch.max(outputs.data, 1)  
                total += labels.size(0)  
                correct += (predicted == labels).sum().item()  
        accuracy = 100 * correct / total  
        print(f'Accuracy: {accuracy} %')  
        return accuracy  

# Define your custom PolSAR Dataset  
class PolSARDataset(Dataset):  
    def __init__(self, images, labels, transform=None):  
        self.images = images  
        self.labels = labels  
        self.transform = transform  

    def __len__(self):  
        return len(self.images)  

    def __getitem__(self, idx):  
        image = self.images[idx]  
        label = self.labels[idx]  
        if self.transform:  
            image = self.transform(image)  
        return image, label  

# Example usage  
def load_data():  
    # Assume we have lists of images and corresponding labels  
    images = [...]  # You should replace this with your actual data loading logic  
    labels = [...]  # Replace with your labels loading logic  
    return images, labels  

images, labels = load_data()  
classifier = PolSARClassifier(num_classes=10)  
classifier.train(images, labels)  
classifier.evaluate(images, labels)