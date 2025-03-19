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
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

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
        
class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# data preprocessing
def delete_file_without_foreground():
    # 載入特定模式的資料集
    dataset = OxfordPetDataset(root="dataset")
    no_foreground_images = []
    print("Analyzing dataset without foregroud...")
    for i in tqdm(range(len(dataset))):
        
def analyze_dataset():
    """分析數據集，找出無前景標記及未分類區域過多的圖片"""
    current_dir = os.getcwd()
    trimap_dir = os.path.join(current_dir, "Lab2_Binary_Semantic_Segmentation_2025", "Lab2_Binary_Semantic_Segmentation_2025", "dataset", "annotations", "trimaps")
    trimap_files = glob.glob(trimap_dir + "/*.png")
    
    # 記錄問題圖片的文件名
    no_foreground_images = []
    large_Notclassified_images = []
    small_foreground_images = []
    
    print("Analyzing dataset...")
    for file in tqdm(trimap_files):
        trimap = np.array(Image.open(file))
        filename = os.path.basename(file)
        
        # 檢查是否有前景標記
        unique_values = np.unique(trimap)
        if 1 not in unique_values:
            no_foreground_images.append(filename)
            continue
            
        # 檢查未定義區域比例
        total_pixels = trimap.size
        Notclassified_pixels = np.sum(trimap == 3)  # 3為未分類區域
        Notclassified_ratio = Notclassified_pixels / total_pixels
        
        if Notclassified_ratio > 0.3:  # 未分類區域超過30%
            large_Notclassified_images.append((filename, Notclassified_ratio))
            continue
            
        # 檢查前景占比是否過小
        foreground_pixels = np.sum((trimap == 1) | (trimap == 3))  # 前景及未分類區域
        foreground_ratio = foreground_pixels / total_pixels
        
        if foreground_ratio < 0.05:  # 前景占比小於5%
            small_foreground_images.append((filename, foreground_ratio))
    
    return no_foreground_images, large_Notclassified_images, small_foreground_images

def create_filtered_dataset(no_foreground_images, large_Notclassified_images, small_foreground_images, min_foreground_ratio=0.05):
    """創建已過濾的數據集"""
    current_dir = os.getcwd()
    
    # 原始數據目錄
    original_images_dir = os.path.join(current_dir, "Lab2_Binary_Semantic_Segmentation_2025", "Lab2_Binary_Semantic_Segmentation_2025", "dataset", "images")
    original_trimaps_dir = os.path.join(current_dir, "Lab2_Binary_Semantic_Segmentation_2025", "Lab2_Binary_Semantic_Segmentation_2025", "dataset", "annotations", "trimaps")
    
    # 創建過濾後的數據目錄
    filtered_dir = os.path.join(current_dir, "Lab2_Binary_Semantic_Segmentation_2025", "Lab2_Binary_Semantic_Segmentation_2025", "dataset")
    filtered_images_dir = os.path.join(filtered_dir, "filtered_images")
    filtered_masks_dir = os.path.join(filtered_dir, "annotations", "filtered_trimaps")
    
    os.makedirs(filtered_images_dir, exist_ok=True)
    os.makedirs(filtered_masks_dir, exist_ok=True)
    
    # 創建排除列表 - 合併所有有問題的文件名
    exclude_files = set(no_foreground_images)
    exclude_files.update([f[0] for f in large_Notclassified_images])
    exclude_files.update([f[0] for f in small_foreground_images])
    
    # 複製過濾後的文件
    all_trimaps = glob.glob(original_trimaps_dir + "/*.png")
    copied_count = 0
    
    print("Creating filtered dataset...")
    for trimap_path in tqdm(all_trimaps):
        filename = os.path.basename(trimap_path)
        
        # 跳過被排除的文件
        if filename in exclude_files:
            continue
            
        # 複製trimap
        shutil.copy(
            trimap_path, 
            os.path.join(filtered_masks_dir, filename)
        )
        
        # 複製對應的圖像
        image_filename = filename.replace(".png", ".jpg")
        image_path = os.path.join(original_images_dir, image_filename)
        
        if os.path.exists(image_path):
            shutil.copy(
                image_path,
                os.path.join(filtered_images_dir, image_filename)
            )
            copied_count += 1
    
    print(f"Filtered dataset created with {copied_count} images")
    return filtered_dir

def create_augmentation_pipeline():
    """創建數據增強管線"""
    # 數據增強組合
    transform = A.Compose([
        # 確保前景/未分類區域占比至少50%的裁剪
        A.CropNonEmptyMaskIfExists(height=224, width=224, scale=(0.5, 1.0), p=1.0),
        
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
            A.MotionBlur(blur_limit=7, p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        ], p=0.3),
        
        # 填充、縮放、調整大小等
        A.OneOf([
            A.PadIfNeeded(min_height=256, min_width=256, p=0.5),
            A.Resize(height=256, width=256, p=0.5),
            A.RandomSizedCrop(min_max_height=(180, 256), height=256, width=256, p=0.5),
        ], p=0.5),
        
        # 噪聲
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        ], p=0.3),
    ])
    
    return transform

def apply_augmentations(filtered_dir, num_augmentations=3):
    """對過濾後的數據集應用數據增強"""
    # 設置源和目標目錄
    src_images_dir = os.path.join(filtered_dir, "images")
    src_masks_dir = os.path.join(filtered_dir, "annotations", "trimaps")
    
    aug_dir = os.path.join(os.path.dirname(filtered_dir), "augmented_oxford_pet")
    aug_images_dir = os.path.join(aug_dir, "images")
    aug_masks_dir = os.path.join(aug_dir, "annotations", "trimaps")
    
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_masks_dir, exist_ok=True)
    
    # 首先複製原始的過濾後文件
    image_files = glob.glob(src_images_dir + "/*.jpg")
    for image_path in image_files:
        filename = os.path.basename(image_path)
        mask_path = os.path.join(src_masks_dir, filename.replace(".jpg", ".png"))
        
        # 複製原始圖像和遮罩
        shutil.copy(image_path, os.path.join(aug_images_dir, filename))
        shutil.copy(mask_path, os.path.join(aug_masks_dir, filename.replace(".jpg", ".png")))
    
    # 創建增強管線
    transform = create_augmentation_pipeline()
    
    # 對每張圖像生成多個增強版本
    print("Applying augmentations...")
    for image_path in tqdm(image_files):
        filename = os.path.basename(image_path)
        base_name = filename.replace(".jpg", "")
        mask_path = os.path.join(src_masks_dir, base_name + ".png")
        
        # 讀取圖像和對應遮罩
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(mask_path))
        
        # 生成多個增強版本
        for i in range(num_augmentations):
            try:
                augmented = transform(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                
                # 確保增強後的遮罩保留原始的值
                aug_mask = np.where(
                    (aug_mask > 0) & (aug_mask <= 3),
                    aug_mask,
                    mask.flatten()[0]  # 使用原始遮罩中的默認值
                )
                
                # 檢查前景占比
                foreground_ratio = np.sum((aug_mask == 1) | (aug_mask == 3)) / aug_mask.size
                if foreground_ratio < 0.5:  # 如果前景占比太小，則跳過
                    continue
                
                # 保存增強後的圖像和遮罩
                aug_image_filename = f"{base_name}_aug{i}.jpg"
                aug_mask_filename = f"{base_name}_aug{i}.png"
                
                # 將RGB圖像轉換回BGR後保存
                cv2.imwrite(
                    os.path.join(aug_images_dir, aug_image_filename),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )
                Image.fromarray(aug_mask.astype(np.uint8)).save(
                    os.path.join(aug_masks_dir, aug_mask_filename)
                )
            except Exception as e:
                print(f"Error augmenting {filename}: {e}")
                continue
    
    # 計算增強後的數據集大小
    final_count = len(glob.glob(aug_images_dir + "/*.jpg"))
    print(f"Augmented dataset created with {final_count} images")
    
    return aug_dir

def visualize_samples(aug_dir, num_samples=5):
    """可視化幾個樣本圖像以檢查增強效果"""
    images_dir = os.path.join(aug_dir, "images")
    masks_dir = os.path.join(aug_dir, "annotations", "trimaps")
    
    # 隨機選擇幾個圖像
    image_files = glob.glob(images_dir + "/*.jpg")
    sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    
    for i, image_path in enumerate(sample_files):
        filename = os.path.basename(image_path)
        mask_path = os.path.join(masks_dir, filename.replace(".jpg", ".png"))
        
        # 讀取圖像和遮罩
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(mask_path))
        
        # 可視化遮罩 (1: 前景, 2: 背景, 3: 未分類)
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask == 1] = [0, 255, 0]    # 前景為綠色
        colored_mask[mask == 2] = [0, 0, 255]    # 背景為藍色
        colored_mask[mask == 3] = [255, 0, 0]    # 未分類為紅色
        
        # 顯示圖像和遮罩
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(colored_mask)
        axes[i, 1].set_title(f"Mask: Green=Foreground, Blue=Background, Red=Unclassified")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(aug_dir, "sample_visualizations.png"))
    plt.close()
    print(f"Sample visualizations saved to {os.path.join(aug_dir, 'sample_visualizations.png')}")

def modify_original_dataset_loader(filtered_or_augmented_dir):
    """修改提供的OxfordPetDataset類以使用我們的過濾/增強數據集"""
    # 這裡我們將返回一個修改後的OxfordPetDataset類
    class ModifiedOxfordPetDataset(torch.utils.data.Dataset):
        def __init__(self, root, mode="train", transform=None):
            assert mode in {"train", "valid", "test"}
            
            self.root = filtered_or_augmented_dir  # 使用我們的過濾/增強數據集
            self.mode = mode
            self.t

    
def load_dataset(data_path, mode):
    # 檢查 mode 是否有效
    assert mode in {"train", "valid", "test"}, "Mode must be one of 'train', 'valid', 'test'"
    
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
    
    # 使用 SimpleOxfordPetDataset 創建數據集對象
    dataset = SimpleOxfordPetDataset(root=data_path, mode=mode)
    
    # 應用數據類型轉換
    transformed_dataset = FloatTransformedDataset(dataset)
    
    return transformed_dataset
    # assert False, "Not implemented yet!"