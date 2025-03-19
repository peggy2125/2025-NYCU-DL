import os
import torch
import shutil
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import glob
import albumentations as A
import matplotlib.pyplot as plt

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        # mask = self._preprocess_mask(trimap)  # do it after data cleaning

        #sample = dict(image=image, mask=mask, trimap=trimap)
        sample = dict(image=image, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def is_dataset_complete(root):
        return (
            os.path.exists(os.path.join(root, "images")) and
            os.path.exists(os.path.join(root, "annotations")) and
            len(os.listdir(os.path.join(root, "images"))) > 0 and
            len(os.listdir(os.path.join(root, "annotations", "trimaps"))) > 0
        )

    @staticmethod
    def download(root):
        # 檢查 images/ 和 annotations/ 是否已經存在
        if OxfordPetDataset.is_dataset_complete(root):
            print("Dataset already exists and is complete. Skipping download.")
            return  # 如果數據已經存在，直接返回

        print("Downloading dataset...")
        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)
        
'''class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)
        
        # Apply mask conversion at this point, after cleaning and resizing
        mask = self._preprocess_mask(trimap)
        
        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample'''


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def claenup_dataset_without_foreground(dataset_root, train_filenames, valid_filenames, test_filenames):
    """分析整體資料集，找出無前景標記的圖片"""
    # 記錄問題圖片的檔案名
    no_foreground_images = []
    print("Analyzing training dataset without foreground...")
    
    modes = {
        "train": train_filenames,
        "valid": valid_filenames,
        "test": test_filenames
    }
    
    for mode, filenames in modes.items():
        for filename in tqdm(filenames, desc=f"Processing {mode} set"):
            trimap_path = os.path.join(dataset_root, "annotations", "trimaps", filename + ".png")
            
            # 檢查檔案是否存在
            if not os.path.exists(trimap_path):
                continue
                
            trimap = np.array(Image.open(trimap_path))
            
            # 檢查是否有前景標記
            unique_values = np.unique(trimap)
            if 1 not in unique_values:
                no_foreground_images.append(f"{mode}/{filename}.png")

    print(f"分析結果：\n- 無前景：{len(no_foreground_images)}張\n")
    return no_foreground_images

# 修改後的資料分析函數 - 只分析訓練資料
def analyze_dataset(dataset_root, train_filenames):
    """分析訓練資料集，找出未分類區域過多與前景過小的圖片"""
    # 記錄問題圖片的檔案名
    large_Notclassified_images = []
    small_foreground_images = []
    
    print("Analyzing training dataset with a large part of no classified area or a small part of pet label...")
    for filename in tqdm(train_filenames):
        trimap_path = os.path.join(dataset_root, "annotations", "trimaps", filename + ".png")
        
        # 檢查檔案是否存在
        if not os.path.exists(trimap_path):
            continue
            
        trimap = np.array(Image.open(trimap_path))
            
        # 檢查未定義區域比例
        total_pixels = trimap.size
        Notclassified_pixels = np.sum(trimap == 3)  # 3為未分類區域
        Notclassified_ratio = Notclassified_pixels / total_pixels
        
        if Notclassified_ratio > 0.3:  # 未分類區域超過30%
            large_Notclassified_images.append((filename + ".png", Notclassified_ratio))
            continue
            
        # 檢查前景占比是否過小
        foreground_pixels = np.sum((trimap == 1) | (trimap == 3))  # 前景及未分類區域
        foreground_ratio = foreground_pixels / total_pixels
        
        if foreground_ratio < 0.05:  # 前景占比小於5%
            small_foreground_images.append((filename + ".png", foreground_ratio))
    
    print(f"分析結果：\n- 未分類區域過多(training data)：{len(large_Notclassified_images)}張\n- 前景過小(training data)：{len(small_foreground_images)}張")
    return large_Notclassified_images, small_foreground_images

# 修改後的過濾資料集函數 - 只過濾訓練資料
def create_filtered_dataset(dataset_root):
    """創建已過濾的訓練資料集，保留原始的驗證和測試資料"""
    # 創建訓練、驗證和測試資料集實例
    train_dataset = OxfordPetDataset(root=dataset_root, mode="train")
    valid_dataset = OxfordPetDataset(root=dataset_root, mode="valid")
    test_dataset = OxfordPetDataset(root=dataset_root, mode="test")
    
    # 獲取檔案名列表
    train_filenames = set(train_dataset.filenames)
    valid_filenames = set(valid_dataset.filenames)
    test_filenames = set(test_dataset.filenames)
    
    # 清理沒有前景的資料
    no_foreground_images = claenup_dataset_without_foreground(dataset_root, train_filenames, valid_filenames, test_filenames)
    #清理其他(only on train data)
    large_Notclassified_images, small_foreground_images = analyze_dataset(dataset_root, train_filenames)
    
    # 創建過濾後的資料目錄
    filtered_dir = os.path.join(os.path.dirname(dataset_root), "filtered_oxford_pet")
    filtered_images_dir = os.path.join(filtered_dir, "images")
    filtered_masks_dir = os.path.join(filtered_dir, "annotations", "trimaps")
    
    os.makedirs(filtered_images_dir, exist_ok=True)
    os.makedirs(filtered_masks_dir, exist_ok=True)
    
    # 創建排除列表 
    exclude_files = set([f.replace(".png", "") for f in no_foreground_images])
    exclude_files.update([f[0].replace(".png", "") for f in large_Notclassified_images])
    exclude_files.update([f[0].replace(".png", "") for f in small_foreground_images])
    
    # 過濾訓練、驗證和測試資料 - 會處理所有資料集
    all_filenames = {
        'train': train_filenames,
        'valid': valid_filenames,
        'test': test_filenames
    }
    
    # 記錄過濾後的訓練資料集檔案名
    filtered_train_filenames = []    
    filtered_valid_filenames = []
    filtered_test_filenames = []
    # 複製所有資料
    for mode, filenames in all_filenames.items():
        skipfile_count = 0
        for filename in filenames:
            
            # 跳過被排除的檔案
            if filename in exclude_files:
                skipfile_count += 1
                continue
            
            # 複製圖像
            src_image_path = os.path.join(dataset_root, "images", filename + ".jpg")
            dst_image_path = os.path.join(filtered_images_dir, filename + ".jpg")
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
        
            # 複製遮罩
            src_mask_path = os.path.join(dataset_root, "annotations", "trimaps", filename + ".png")
            dst_mask_path = os.path.join(filtered_masks_dir, filename + ".png")
            if os.path.exists(src_mask_path):
                shutil.copy(src_mask_path, dst_mask_path)

            # 記錄過濾後的訓練資料檔案名
            if mode == 'train' and filename not in exclude_files:
                filtered_train_filenames.append(filename)
            elif mode == 'valid' and filename not in exclude_files:
                filtered_valid_filenames.append(filename)
            elif mode == 'test' and filename not in exclude_files:
                filtered_test_filenames.append(filename)                
        print(f"已過濾的{mode}資料：{len(filenames) - skipfile_count}張")
    
        # 創建並寫入過濾後的訓練、驗證和測試資料集檔案名（先清空內容）
    with open(os.path.join(filtered_dir, "train.txt"), "w") as f:
        for filename in filtered_train_filenames:
            f.write(f"{filename}.jpg\n")
    
    with open(os.path.join(filtered_dir, "valid.txt"), "w") as f:
        for filename in filtered_valid_filenames:
            f.write(f"{filename}.jpg\n")
    
    with open(os.path.join(filtered_dir, "test.txt"), "w") as f:
        for filename in filtered_test_filenames:
            f.write(f"{filename}.jpg\n")
                
    return filtered_dir , filtered_train_filenames, filtered_valid_filenames, filtered_test_filenames

def create_augmentation_pipeline():
    """創建數據增強管線"""
    # 數據增強組合
    transform = A.Compose([
        
        A.CropNonEmptyMaskIfExists(height=224, width=224, p=1.0),
        
        # 基本空間變換
        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
        ], p=0.5),
        
        # 色彩調整
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
        ], p=0.5),
        
        # 模糊和銳化
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        ], p=0.3),
        
        # # 填充、縮放、調整大小等
         A.OneOf([
             A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
             A.RandomSizedCrop(min_max_height=(180, 256), size=(256,256), interpolation=cv2.INTER_NEAREST, p=0.5),
         ], p=0.5),
        
        # 噪聲
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        ], p=0.2),
    ])
    
    return transform

def visualize_samples(aug_dir, num_samples=5):
    """可視化資料集樣本，區分訓練、驗證和測試資料"""
    
    # 讀取資料集檔案名
    with open(os.path.join(aug_dir, "train_filenames.txt"), "r") as f:
        train_filenames = [line.strip() for line in f.readlines()]
    
    # 區分增強資料和原始資料
    augmented_filenames = [f for f in train_filenames if "_aug" in f]
    original_train_filenames = [f for f in train_filenames if "_aug" not in f]
    
    # 驗證和測試資料的檔案名
    with open(os.path.join(aug_dir, "valid_filenames.txt"), "r") as f:
        valid_filenames = [line.strip() for line in f.readlines()]
    
    with open(os.path.join(aug_dir, "test_filenames.txt"), "r") as f:
        test_filenames = [line.strip() for line in f.readlines()]

    # 定義資料類型及對應的檔案名
    data_types = [
        ("原始訓練資料", original_train_filenames),
        ("增強訓練資料", augmented_filenames),
        ("驗證資料", valid_filenames),
        ("測試資料", test_filenames)
    ]
    
    # 創建一個大型圖形
    fig, axes = plt.subplots(len(data_types), num_samples, figsize=(15, 3*len(data_types)))
    
    for i, (data_type, filenames) in enumerate(data_types):
        if len(filenames) == 0:
            continue
        
        # 隨機選擇樣本
        selected_files = np.random.choice(filenames, min(num_samples, len(filenames)), replace=False)
        
        for j, filename in enumerate(selected_files):
            # 載入圖像和遮罩
            image_path = os.path.join(aug_dir, "images", filename + ".jpg")
            mask_path = os.path.join(aug_dir, "annotations", "trimaps", filename + ".png")
            
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                continue
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.array(Image.open(mask_path))
            
            # 可視化遮罩 (1: 前景, 2: 背景, 3: 未分類)
            # 創建一個帶有透明度的覆蓋遮罩
            overlay = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
            overlay[mask == 1] = [0, 255, 0, 128]    # 前景為綠色
            overlay[mask == 2] = [0, 0, 255, 128]    # 背景為藍色
            overlay[mask == 3] = [255, 0, 0, 128]    # 未分類為紅色
            
            # 將遮罩疊加到圖像上
            overlay_image = image.copy()
            overlay_rgb = overlay[..., :3]
            overlay_alpha = overlay[..., 3:4] / 255.0
            overlay_image = overlay_image * (1 - overlay_alpha) + overlay_rgb * overlay_alpha
            
            # 顯示圖像
            axes[i, j].imshow(overlay_image.astype(np.uint8))
            axes[i, j].set_title(f"{data_type}\n{os.path.basename(filename)}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(aug_dir, "sample_visualizations.png"))
    plt.close()
    print(f"樣本可視化已保存至 {os.path.join(aug_dir, 'sample_visualizations.png')}")
    
class PreprocessOxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, argdata_path, mode, augmentations=None, num_augmentations=1):
        """
        初始化自定義資料集類。
        
        :param data_path: 資料夾路徑
        :param mode: 模式 ('train', 'valid', 'test')
        :param augmentations: 數據增強操作，默認為 None
        """
        self.data_path = argdata_path
        self.mode = mode
        self.augmentations = augmentations
        self.num_augmentations = num_augmentations
        # 讀取對應模式的檔案列表 (train.txt, valid.txt, test.txt)
        self.filenames = self._load_filenames()
        
        # 如果是訓練模式，應用增強
        if self.mode == "train":
            self.apply_augmentations()
            
    def _load_filenames(self):
        """根據模式讀取相應的檔案名列表"""
        split_file = os.path.join(self.data_path, f"{self.mode}.txt")
        filenames = []
        
        with open(split_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        
        return filenames
        
    def _preprocess_mask(self, trimap):
        """將 trimap 轉換為二值 mask"""
        mask = trimap.astype(np.float32)
        mask[mask == 2.0] = 0.0  # Class 2 為背景
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0  # Class 1 和 3 為前景
        return mask
    
    # 修改後的增強函數 - 只增強訓練資料
    def apply_augmentations(self):
        """對過濾後的訓練資料應用數據增強，保留過濾後的驗證和測試資料"""

        # 讀取過濾資料集的訓練檔案名
        with open(os.path.join(self.data_path, "train.txt"), "r") as f:
            train_filenames = [line.strip().replace('.jpg', '') for line in f.readlines()]

        # 創建增強管線
        transform = create_augmentation_pipeline()
    
        # 增強的圖片和遮罩目錄
        aug_images_dir = os.path.join(self.data_path, "images")
        aug_masks_dir = os.path.join(self.data_path, "annotations", "trimaps")
        os.makedirs(aug_images_dir, exist_ok=True)
        os.makedirs(aug_masks_dir, exist_ok=True)
    
        augmented_filenames = []
        # 只對訓練資料生成增強版本
        print("Applying augmentations to training data...")
        aug_count = 0
        for filename in tqdm(train_filenames):
            image_path = os.path.join(self.data_path, "images", filename + ".jpg")
            trimap_path = os.path.join(self.data_path, "annotations", "trimaps", filename + ".png")
            
            # 檢查檔案是否存在
            if not os.path.exists(image_path) or not os.path.exists(trimap_path):
                continue
            
            # 讀取圖像和對應遮罩
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            trimap = np.array(Image.open(trimap_path))
            
            # 生成多個增強版本
            for i in range(self.num_augmentations):
                try:
                    augmented = transform(image=image, mask = trimap)
                    aug_image = augmented['image']
                    aug_mask = augmented['mask']
                    
                    # 確保增強後的遮罩保留原始的值
                    aug_mask = np.where(
                        (aug_mask > 0) & (aug_mask <= 3),
                        aug_mask,
                        trimap.flatten()[0]  # 使用原始遮罩中的默認值
                    )
                    
                    # 檢查前景占比
                    foreground_ratio = np.sum((aug_mask == 1) | (aug_mask == 3)) / aug_mask.size
                    if foreground_ratio < 0.05:  # 如果前景占比太小，則跳過
                        continue
                    
                    # 保存增強後的圖像和遮罩
                    aug_image_filename = f"{filename}_aug{i:02d}.jpg"
                    aug_mask_filename = f"{filename}_aug{i:02d}.png"
                    
                    # 將RGB圖像轉換回BGR後保存
                    cv2.imwrite(
                        os.path.join(aug_images_dir, aug_image_filename),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    )
                    Image.fromarray(aug_mask.astype(np.uint8)).save(
                        os.path.join(aug_masks_dir, aug_mask_filename)
                    )
                    
                    # 將增強後的檔案名稱添加到列表中
                    augmented_filenames.append(f"{filename}_aug{i:02d}")
                    aug_count += 1
                except Exception as e:
                    print(f"Error augmenting {filename}: {e}")
                    continue
        
        # 更新訓練集的 `train_filenames.txt`
        with open(os.path.join(self.data_path, "train_filenames.txt"), 'a') as f:
            for aug_filename in augmented_filenames:
                f.write(f"{aug_filename}\n")

        # 計算增強後的資料集大小
        final_train_count = len([f for f in os.listdir(aug_images_dir) if any(f.startswith(fn) for fn in train_filenames)])
        final_valid_count = len([f for f in os.listdir(os.path.join(filtered_dir, "valid_filenames.txt"))])
        final_test_count = len([f for f in os.listdir(os.path.join(filtered_dir, "test_filenames.txt"))])
        
        print(f"增強後的訓練資料：{final_train_count}張 (原始: {len(train_filenames)}張 + 增強: {aug_count}張)")
        print(f"驗證資料：{final_valid_count}張 (未增強)")
        print(f"測試資料：{final_test_count}張 (未增強)")
    
    def __getitem__(self, idx):
        """獲取處理過的樣本"""
        filename = self.filenames[idx]
        
        # 圖像和遮罩的路徑
        image_path = os.path.join(self.data_path, "images", f"{filename}.jpg")
        trimap_path = os.path.join(self.data_path, "annotations", "trimaps", f"{filename}.png")
        
        # 讀取圖像、遮罩和 trimap
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        trimap = np.array(Image.open(trimap_path))
        
        # 轉換 trimap -> mask
        mask = self._preprocess_mask(trimap)        
        
        # 清理與處理
        image = np.array(Image.fromarray(image).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(trimap).resize((256, 256), Image.NEAREST))
        
        # 轉換到 CHW 格式
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)
        trimap = np.expand_dims(trimap, 0)
        
        sample = {"image": image, "mask": mask, "trimap": trimap}
        
        # 如果有增強操作，應用增強
        '''if self.augmentations is not None and self.mode == "train":
            augmented = self.augmentations(image=image, mask=mask, trimap=trimap)
            sample["image"] = augmented["image"]
            sample["mask"] = augmented["mask"]
            sample["trimap"] = augmented["trimap"]
        
        return sample'''
    
    def __len__(self):
        """返回資料集大小"""
        return len(self.filenames)

def load_dataset(data_path, mode):
    # 檢查 mode 是否有效
    assert mode in {"train", "valid", "test"}, "Mode must be one of 'train', 'valid', 'test'"
    
    #資料清理
    filtered_dir, filtered_train_filenames, filtered_valid_filenames, filtered_test_filenames = create_filtered_dataset(data_path)
    
    class FloatTransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            sample = self.dataset[idx]
            # 將圖像數據轉換為 float 類型並歸一化到 [0, 1]
            sample['image'] = sample['image'].astype(np.float32) / 255.0
            sample['mask'] = sample['mask'].astype(np.float32)
            sample['trimap'] = sample['trimap'].astype(np.float32)
            return sample
    
    # 使用 PreprocessOxfordPetDataset 創建數據集對象
    dataset = PreprocessOxfordPetDataset(argdata_path=filtered_dir, mode=mode)
    
    # 應用數據類型轉換
    transformed_dataset = FloatTransformedDataset(dataset)
    return transformed_dataset

'''def process_oxford_pet_dataset(dataset_root):
    """完整的資料處理流程，只對訓練資料進行清理和增強"""
    
    # 確保資料集根目錄存在
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"資料集路徑不存在: {dataset_root}")

    # 2. 創建過濾後的資料集（只過濾訓練資料）
    filtered_dir, filtered_train_filenames, filtered_valid_filenames, filtered_test_filenames = create_filtered_dataset(dataset_root, no_foreground, large_unclassified, small_foreground)
    
    # 3. 增強訓練資料
    augmented_dir = apply_augmentations(filtered_dir, num_augmentations=3)
    
    # 4. 可視化樣本
    visualize_samples(augmented_dir)
    
    # 5. 載入處理後的資料集
    train_dataset = load_dataset(augmented_dir, mode="train")
    valid_dataset = load_dataset(augmented_dir, mode="valid")
    test_dataset = load_dataset(augmented_dir, mode="test")
    
    return train_dataset, valid_dataset, test_dataset'''

# 使用示例
if __name__ == "__main__":
    # 設置資料集根目錄
    dataset_root = r"D:\PUPU\2025 NYCU DL\Lab2\Lab2_Binary_Semantic_Segmentation_2025\Lab2_Binary_Semantic_Segmentation_2025\dataset"
    
    # 處理資料集
    try:
        
        train_dataset = load_dataset(dataset_root, mode="train")
        valid_dataset = load_dataset(dataset_root, mode="valid")
        test_dataset = load_dataset(dataset_root, mode="test")
        
        visualize_samples(r"D:\PUPU\2025 NYCU DL\Lab2\Lab2_Binary_Semantic_Segmentation_2025\Lab2_Binary_Semantic_Segmentation_2025\filtered_oxford_pet", num_samples=5)
        
        print(f"訓練資料集大小: {len(train_dataset)}")
        print(f"驗證資料集大小: {len(valid_dataset)}")
        print(f"測試資料集大小: {len(test_dataset)}")
    except Exception as e:
        print(f"發生錯誤: {e}")
