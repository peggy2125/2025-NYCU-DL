import cv2
import os
import numpy as np
from tqdm import tqdm  # 进度条显示

# 定义路径
trimap_dir = r"D:\PUPU\2025 NYCU DL\Lab2\Lab2_Binary_Semantic_Segmentation_2025\Lab2_Binary_Semantic_Segmentation_2025\dataset\annotations\trimaps"
save_dir = r"D:\PUPU\2025 NYCU DL\Lab2\Lab2personal\trimap_visualized"

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 遍历 trimaps 目录中的所有 PNG 文件
for filename in tqdm(os.listdir(trimap_dir), desc="Processing trimaps"):
    if filename.endswith(".png"):  # 确保只处理 PNG 图片
        trimap_path = os.path.join(trimap_dir, filename)
        save_path = os.path.join(save_dir, filename)

        # 读取 trimap
        trimap = cv2.imread(trimap_path, cv2.IMREAD_UNCHANGED)

        # 确保 trimap 读取成功
        if trimap is None:
            print(f"Error: Unable to load {trimap_path}")
            continue

        # 显示原始 trimap 可能的像素值
        unique_vals = np.unique(trimap)
        if set(unique_vals) - {1, 2, 3}:
            print(f"Warning: {filename} contains unexpected pixel values: {unique_vals}")

        # 转换 trimap 为可视化格式 (映射 1, 2, 3 到 0, 127, 255)
        trimap_vis = np.zeros_like(trimap, dtype=np.uint8)
        trimap_vis[trimap == 1] = 0       # 背景 (黑色)
        trimap_vis[trimap == 2] = 127     # 未分类 (灰色)
        trimap_vis[trimap == 3] = 255     # 前景 (白色)

        # 另存为可查看的图片
        cv2.imwrite(save_path, trimap_vis)

print(f"All processed trimaps saved in: {save_dir}")
