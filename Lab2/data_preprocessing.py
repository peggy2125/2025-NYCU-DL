import numpy as np
import glob
from PIL import Image
import os
import cv2

imagewithout_foreground=[]
current_dir = os.getcwd()
print(current_dir)
#trimap_dir = os.path.join(current_dir, "Lab2_Binary_Semantic_Segmentation_2025", "dataset", "annotations", "trimaps")
trimap_dir = os.path.join(current_dir, "Lab2_Binary_Semantic_Segmentation_2025", "Lab2_Binary_Semantic_Segmentation_2025", "dataset", "annotations", "trimaps")
trimap_files = glob.glob(trimap_dir + "/*.png")

print(f"Error: image with missing foreground labels:")
for file in trimap_files:
    trimap = np.array(Image.open(file))
    unique_values = np.unique(trimap)  
    filename = os.path.basename(file)  
    if 1 not in unique_values: 
        imagewithout_foreground.append(filename)
   
    if not set(unique_values).issubset({1, 2, 3}):
        print(f"Error (image with undefined pixel value): {file} e")
for filename in imagewithout_foreground:
    print(f"{filename}")
    
print("\n\n")

# check if the area of Not classified area is too large (define over 30% as too large)
trimap_files = glob.glob(trimap_dir + "/*.png")
print(f"Error:the area of Not classified area is too large:")

for file in trimap_files:
    trimap = np.array(Image.open(file))
    filename = os.path.basename(file)
    total_pixels = trimap.size
    uncertain_pixels = np.sum(trimap == 3)  # 假设 3 为未确定区域
    uncertain_ratio = uncertain_pixels / total_pixels
    
    if uncertain_ratio > 0.3:  # 若未確定區域超過 30%
        print(f"{filename} : {uncertain_ratio:.2%}")
print("\n\n")











# delete the error images