import numpy as np
import glob
from PIL import Image
import os
import cv2

errorfile=[]
trimap_dir = r"D:\PUPU\2025 NYCU DL\Lab2\Lab2_Binary_Semantic_Segmentation_2025\Lab2_Binary_Semantic_Segmentation_2025\dataset\annotations\trimaps"

# 找出全部的 trimap
trimap_files = glob.glob(trimap_dir + "/*.png")

print(f"⚠️可能有錯誤：沒有標記前景的圖片")
for file in trimap_files:
    trimap = np.array(Image.open(file))
    unique_values = np.unique(trimap)  # 取得該 trimap 內的唯一像素值
    filename = os.path.basename(file)  # Extract file name from the path
    if 1 not in unique_values:  # 若缺少前景標籤
        errorfile.append(filename)
    # 确保没有超出范围的像素值
    if not set(unique_values).issubset({1, 2, 3}):
        print(f"⚠️ {file} 可能有錯誤：包含未知像素值 {unique_values}")
for filename in errorfile:
    print(f"{filename}")#顯示並檢查

visualized_image_file = r"D:\PUPU\2025 NYCU DL\Lab2\Lab2personal\visualized_trimaps"

# 遍历出错文件
for filename in errorfile:
    # 获取文件名并在 '.png' 前加上 '_colored'
    name, ext = os.path.splitext(filename)  # 分离文件名和扩展名
    new_filename = name + "_colored" + ext  # 在文件名中添加 '_colored'

    # 拼接新的文件路径
    filepath = os.path.join(visualized_image_file, new_filename)

    # 读取图片
    image = cv2.imread(filepath)
    
    if image is not None:  # 确保图片成功加载
        # 显示图像
        cv2.imshow(f"Trimap - {new_filename}", image)
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()  # 关闭窗口
    else:
        print(f"無法加載圖像: {new_filename}")
