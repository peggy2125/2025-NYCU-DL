''' original code'''
'''import argparse

def train(args):
    # implement the training function here
    
    assert False, "Not implemented yet!"

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()'''
    
#test code
import argparse
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.oxford_pet_test import load_dataset # 导入加载数据集的函数
import torch.nn.functional as F
import os

''' model structure '''
class DoubleConv(nn.Module):
    """ 兩層連續卷積 + 批次正規化 + ReLU """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # 編碼（Encoder）：每層深度加倍
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.encoder5 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解碼（Decoder）：使用轉置卷積（上採樣）
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512) #input: 1024 = skip connection from encoder4 + 512 from upconv4

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256) #input: 512 = skip connection from encoder3 + 256 from upconv3

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128) #input: 256 = skip connection from encoder2 + 128 from upconv2

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64) #input: 128 = skip connection from encoder1 + 64 from upconv1

        # 輸出層：1x1 卷積轉換為 1 個通道
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        #print(f"encoder1尺寸: {enc1.shape}")
        enc2 = self.encoder2(self.pool(enc1))
        #print(f"encoder2尺寸: {enc2.shape}")
        enc3 = self.encoder3(self.pool(enc2))
        #print(f"encoder3尺寸: {enc3.shape}")
        enc4 = self.encoder4(self.pool(enc3))
        #print(f"encoder4尺寸: {enc4.shape}")
        enc5 = self.encoder5(self.pool(enc4))
        #print(f"encoder5尺寸: {enc5.shape}")
        
        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        #print(f"decoder4尺寸: {dec4.shape}")

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        #print(f"decoder3尺寸: {dec3.shape}")
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        #print(f"decoder2尺寸: {dec2.shape}")

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        #print(f"decoder1尺寸: {dec1.shape}")

        # 輸出影像分割結果
        out = self.final_conv(dec1)
        return out

def dice_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold  # 使用sigmoid处理输出
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return 2.0 * intersection / (union + intersection + 1e-6)


###到這裡為止
import os  # 添加此import以使用路徑功能

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据 - 直接使用load_dataset而不是random split
    train_dataset = load_dataset(args.data_path, mode="train")
    valid_dataset = load_dataset(args.data_path, mode="valid")
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. 创建模型
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)

    # 3. 设置损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 创建保存模型的目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 跟踪最佳验证指标
    best_dice = 0.0

    # 4. 训练过程
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        epoch_dice_score = 0.0

        for batch in tqdm(train_loader):
            images, masks = batch['image'], batch['mask']
            
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算 Dice Score
            dice = dice_score(outputs, masks)
            epoch_dice_score += dice.item()

        # 每个 epoch 结束后打印训练损失
        train_loss = running_loss / len(train_loader)
        train_dice = epoch_dice_score / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.4f}, "
              f"Dice Score: {train_dice:.4f}")

        # 验证
        val_loss, val_dice = validate(model, valid_loader, criterion, device)
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(args.save_path, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
            }, save_path)
            print(f"Best model saved to {save_path}")
        
        # 每个epoch结束后保存检查点
        checkpoint_path = os.path.join(args.save_path, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    epoch_dice_score = 0.0

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            images, masks = batch['image'], batch['mask']
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

            # 计算 Dice Score
            dice = dice_score(outputs, masks)
            epoch_dice_score += dice.item()

    val_loss = running_loss / len(valid_loader)
    val_dice = epoch_dice_score / len(valid_loader)
    print(f"Validation Loss: {val_loss:.4f}, "
          f"Dice Score: {val_dice:.4f}")
    
    return val_loss, val_dice

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, required=True, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--save_path', type=str, default='D:\\PUPU\\2025 NYCU DL\\Lab2\\Lab2_Binary_Semantic_Segmentation_2025\\Lab2_Binary_Semantic_Segmentation_2025\\saved_models', 
                        help='path to save the model')

    return parser.parse_args()
        


if __name__ == "__main__":
    args = get_args()
    train(args)


    