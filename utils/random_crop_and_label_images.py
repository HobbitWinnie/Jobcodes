import os  
import pandas as pd  
from PIL import Image  
from pathlib import Path  
import random  
from concurrent.futures import ThreadPoolExecutor  

def process_image(image_path, labels, crop_width, crop_height, num_crops, cropped_image_dir):  
    try:  
        with Image.open(image_path) as img:  
            if img.size != (1024, 1024):  
                print(f"Warning: Image {image_path.name} size is not 1024x1024, actual size is {img.size}. Skipping.")  
                return []  

            crops = []  
            for i in range(num_crops):  
                max_x = img.width - crop_width  
                max_y = img.height - crop_height  
                x1 = random.randint(0, max_x)  
                y1 = random.randint(0, max_y)  
                x2 = x1 + crop_width  
                y2 = y1 + crop_height  

                cropped_img = img.crop((x1, y1, x2, y2))  
                cropped_filename = f"{image_path.stem}_crop_{i}.png"  
                cropped_path = Path(cropped_image_dir) / cropped_filename  
                cropped_img.save(cropped_path)  
                crops.append([cropped_filename] + labels)  

            return crops  
    except Exception as e:  
        print(f"Error processing image {image_path.name}: {e}")  
        return []  

def random_crop_images(image_dir, label_csv_path, output_dir, crop_width, crop_height, num_crops, max_workers=8):  
    labels_df = pd.read_csv(label_csv_path)  
    os.makedirs(output_dir, exist_ok=True)  
    cropped_image_dir = os.path.join(output_dir, "cropped_images")  
    os.makedirs(cropped_image_dir, exist_ok=True)  

    new_labels = []  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = []  
        for index, row in labels_df.iterrows():  
            filename = row.iloc[0] + ".png"  # Append .png to the filename  
            labels = row.iloc[1:].tolist()  
            image_path = Path(image_dir) / filename  
            if not image_path.exists():  
                print(f"Warning: Image {filename} not found, skipping.")  
                continue  

            futures.append(executor.submit(process_image, image_path, labels, crop_width, crop_height, num_crops, cropped_image_dir))  

        for future in futures:  
            result = future.result()  
            if result:  
                new_labels.extend(result)  

    new_labels_df = pd.DataFrame(new_labels, columns=labels_df.columns)  
    output_csv_path = Path(output_dir) / "cropped_labels.csv"  
    new_labels_df.to_csv(output_csv_path, index=False)  
    print(f"Cropped images and labels have been saved to {output_dir}")  

# Example Usage  
image_directory = "/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel/image_tianji"  # Path to 1024x1024 images  
csv_label_path = "/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel/multilabel_all.csv"  # Path to labels CSV file  
output_directory = "/home/Dataset/nw/Multilabel-Datasets/TIANJI_512x512_dataset"  # Output directory for cropped images and labels  

# Specify the crop size and number of crops  
crop_width = 512  
crop_height = 512  
num_crops = 6  # Number of random crops per image  

random_crop_images(image_directory, csv_label_path, output_directory, crop_width, crop_height, num_crops)


