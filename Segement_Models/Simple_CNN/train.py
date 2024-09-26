import torch  
import torch.optim as optim  
import torch.nn as nn  

from sklearn.metrics import classification_report, accuracy_score  
import logging  

def train_model(model, train_loader, val_loader, save_path, num_epochs=10):  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    model.to(device)  # 将模型移动到设备  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  

    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        for inputs, labels in train_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移动到设备  
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')  

        # Validation  
        model.eval()  
        all_preds = []  
        all_labels = []  
        with torch.no_grad():  
            for inputs, labels in val_loader:  
                inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移动到设备  
                outputs = model(inputs)  
                _, predicted = torch.max(outputs, 1)  
                all_preds.extend(predicted.cpu().numpy())  
                all_labels.extend(labels.cpu().numpy())  

        acc = accuracy_score(all_labels, all_preds)  
        logging.info(f'Validation Accuracy: {acc * 100:.2f}%')  
        logging.info(f'Classification Report:\n{classification_report(all_labels, all_preds)}')

    # Save the trained model  
    torch.save(model.state_dict(), save_path)  
    logging.info(f'Model saved to {save_path}')  