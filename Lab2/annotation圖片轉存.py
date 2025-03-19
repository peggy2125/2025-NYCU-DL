import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm  # 進度條

# 設定路徑
trimap_dir = r"D:\PUPU\2025 NYCU DL\Lab2\Lab2_Binary_Semantic_Segmentation_2025\Lab2_Binary_Semantic_Segmentation_2025\dataset\annotations\trimaps"
output_dir = r"D:\PUPU\2025 NYCU DL\\Lab2\Lab2personal\visualized_trimaps"
image_dir = r"D:\PUPU\2025 NYCU DL\\Lab2\Lab2_Binary_Semantic_Segmentation_2025\Lab2_Binary_Semantic_Segmentation_2025\dataset\images"

# 創建輸出目錄
os.makedirs(output_dir, exist_ok=True)

# 獲取所有 trimap 檔案
trimap_files = glob.glob(os.path.join(trimap_dir, "*.png"))

# 定義顏色映射
color_map = {
    1: [0, 0, 255],    # 前景 (藍色)
    2: [0, 255, 0],    # 背景 (綠色)
    3: [255, 0, 0]     # 未分類 (紅色)
}

# 轉換 Trimap 並儲存
for trimap_file in tqdm(trimap_files, desc="處理 trimap 檔案"):
    trimap = np.array(Image.open(trimap_file))

    # 確保 trimap 只有 1, 2, 3
    unique_values = np.unique(trimap)
    valid_values = set(color_map.keys())

    if not set(unique_values).issubset(valid_values):
        print(f"⚠️ {trimap_file} 包含未知像素值 {unique_values - valid_values}，可能有問題！")
        continue  # 跳過這個 trimap，避免錯誤

    # 建立彩色圖像
    height, width = trimap.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    for value, color in color_map.items():
        colored_image[trimap == value] = color

    # 保存彩色圖片（避免 plt.imsave 影響顏色）
    output_filename = os.path.splitext(os.path.basename(trimap_file))[0] + "_colored.png"
    output_path = os.path.join(output_dir, output_filename)
    Image.fromarray(colored_image).save(output_path)

print(f"已處理 {len(trimap_files)} 個 trimap 檔案")
print(f"彩色圖片已保存至: {output_dir}")

# 可視化示例
def visualize_samples(num_samples=5):
    import random
    sample_files = random.sample(trimap_files, min(num_samples, len(trimap_files)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    
    for i, trimap_file in enumerate(sample_files):
        base_name = os.path.splitext(os.path.basename(trimap_file))[0]
        output_filename = base_name + "_colored.png"
        output_path = os.path.join(output_dir, output_filename)

        trimap = np.array(Image.open(trimap_file))
        colored_img = plt.imread(output_path)

        axes[i, 0].imshow(trimap, cmap='gray')
        axes[i, 0].set_title(f"原始 Trimap: {base_name}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(colored_img)
        axes[i, 1].set_title("彩色可視化")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# 可視化原始圖片、Trimap 與彩色結果
def visualize_with_original(num_samples=5):
    import random
    sample_files = random.sample(trimap_files, min(num_samples, len(trimap_files)))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3*num_samples))
    
    for i, trimap_file in enumerate(sample_files):
        base_name = os.path.splitext(os.path.basename(trimap_file))[0]
        image_file = os.path.join(image_dir, base_name + ".jpg")
        output_filename = base_name + "_colored.png"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(image_file):
            original_img = plt.imread(image_file)
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f"原始圖片: {base_name}")
            axes[i, 0].axis('off')
        else:
            axes[i, 0].text(0.5, 0.5, "原始圖片不存在", ha='center', va='center')
            axes[i, 0].axis('off')

        trimap = np.array(Image.open(trimap_file))
        axes[i, 1].imshow(trimap, cmap='gray')
        axes[i, 1].set_title("Trimap")
        axes[i, 1].axis('off')

        colored_img = plt.imread(output_path)
        axes[i, 2].imshow(colored_img)
        axes[i, 2].set_title("彩色可視化")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
