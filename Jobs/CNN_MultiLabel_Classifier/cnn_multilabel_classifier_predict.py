import os  
import pandas as pd  
from PIL import Image  
import torch  

class ImagePredictor:  
    def __init__(self, model_path, num_classes, device=None):  
        self.model = ResNetMultiLabelClassifier(num_classes, device)  
        self.model.load_state_dict(torch.load(model_path))  
        self.model.eval()  
        self.device = self.model.device  
        
    def predict_image(self, image_path, threshold=0.5):  
        try:  
            image = Image.open(image_path).convert("RGB")  
        except Exception as e:  
            print(f"Error loading {image_path}: {str(e)}")  
            return None  
            
        tensor = self.model.preprocess(image).unsqueeze(0).to(self.device)  
        with torch.no_grad():  
            output = self.model.backbone(tensor)  
            
        return torch.sigmoid(output).squeeze().cpu().numpy()  
    
    def predict_folder(self, folder_path, output_csv="results.csv"):  
        results = []  
        for fname in os.listdir(folder_path):  
            if fname.lower().split(".")[-1] not in ["jpg", "png", "jpeg"]:  
                continue  
                
            path = os.path.join(folder_path, fname)  
            pred = self.predict_image(path)  
            if pred is not None:  
                results.append({"filename": fname, "predictions": pred})  
                
        pd.DataFrame(results).to_csv(output_csv, index=False)  