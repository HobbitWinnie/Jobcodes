import os  
import pandas as pd  
from tqdm import tqdm  

def generate_csv_from_image_folder(folder_path, output_csv_path):  
    # Dictionary to track image paths and their labels  
    data = {}  
    seen_files = set()  
    all_labels = sorted(os.listdir(folder_path))  
    label_dict = {label: idx for idx, label in enumerate(all_labels)}  

    for label in tqdm(all_labels):  
        label_path = os.path.join(folder_path, label)  
        if not os.path.isdir(label_path):  
            continue  

        for image_name in os.listdir(label_path):  
            if image_name.endswith(('.jpg', '.jpeg', '.png', '.gif')):  
                image_path = os.path.join(label_path, image_name)  

                if image_name not in seen_files:  
                    seen_files.add(image_name)  
                    # Initialize label list with zeroes  
                    labels = [0] * len(all_labels)  
                    data[image_name] = [image_path, labels]  

                # Set the label for this directory  
                data[image_name][1][label_dict[label]] = 1  

    # Prepare data for CSV  
    csv_data = [[img_data[0]] + img_data[1] for img_name, img_data in data.items()]  

    # Create a DataFrame and save as CSV  
    columns = ['image_path'] + all_labels  
    df = pd.DataFrame(csv_data, columns=columns)  
    df.to_csv(output_csv_path, index=False)  
    print(f"CSV file has been saved to {output_csv_path}")  

# Example usage  
image_folder_path = '/home/Dataset/nw/GF2_Data/MultiLabel_dataset/data'  # The root folder containing subfolders of images  
output_csv_path = '/home/Dataset/nw/GF2_Data/MultiLabel_dataset/csv_file/labels_d813.csv'  
generate_csv_from_image_folder(image_folder_path, output_csv_path)