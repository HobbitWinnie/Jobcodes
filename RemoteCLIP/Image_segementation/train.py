import os  
import logging  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.metrics import accuracy_score  


def train_model(model, train_loader, val_loader, num_classes, model_save_path, num_epochs=500, lr=1e-4):  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model.to(device)  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        for inputs, labels in train_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  

            # Check if label values are valid  
            unique_labels = torch.unique(labels)  
            if unique_labels.max() >= num_classes or unique_labels.min() < 0:  
                raise ValueError("Label values are out of range!")  

            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        # Validation  
        acc = validate_model(model, val_loader, device)  
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {acc * 100:.2f}%')  

    # Save the trained model  
    torch.save(model.state_dict(), model_save_path)  
    logging.info(f'Model saved to {model_save_path}')  
    

def validate_model(model, val_loader, device):  
    model.eval()  
    all_preds = []  
    all_labels = []  
    with torch.no_grad():  
        for inputs, labels in val_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            _, predicted = torch.max(outputs, 1)  
            all_preds.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())  

    acc = accuracy_score(all_labels, all_preds)  

    return acc  
