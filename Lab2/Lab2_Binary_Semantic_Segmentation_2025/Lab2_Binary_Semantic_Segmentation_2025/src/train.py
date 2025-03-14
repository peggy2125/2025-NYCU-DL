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
from src.oxford_pet import load_dataset # 导入加载数据集的函数
import torch.nn.functional as F

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

######用於測試而已]
from torch.utils.data import random_split

def load_and_split_data(data_path, batch_size):
    """ 加载数据集并随机拆分 """
    full_train_dataset = load_dataset(data_path, mode="train")
    full_valid_dataset = load_dataset(data_path, mode="valid")

    train_size = int(0.2 * len(full_train_dataset))
    valid_size = int(0.2 * len(full_valid_dataset))

    train_subset, _ = random_split(full_train_dataset, [train_size, len(full_train_dataset) - train_size])
    valid_subset, _ = random_split(full_valid_dataset, [valid_size, len(full_valid_dataset) - valid_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

    # **打印调试信息**
    print(f"Train Loader Type: {type(train_loader)}")
    print(f"Valid Loader Type: {type(valid_loader)}")
    print(f"Train Subset Length: {len(train_subset)}")
    print(f"Valid Subset Length: {len(valid_subset)}")

    return train_loader, valid_loader


###到這裡

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    #train_loader = load_dataset(args.data_path, mode="train")
    #valid_loader = load_dataset(args.data_path, mode="valid")
    train_loader, valid_loader = load_and_split_data(args.data_path, args.batch_size)
    
    # 2. 创建模型
    # 在模型初始化時設置 in_channels=3
    model = UNet(in_channels=3, out_channels=1)# 替换为你自己的模型结构
    model = model.to(device)   # 如果使用GPU, 否则删除此行

    # 3. 设置损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 适用于二分类问题
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # 优化器

    # 4. 训练过程
    for epoch in range(args.epochs):
        model.train()  # 设为训练模式
        running_loss = 0.0
        epoch_dice_score = 0.0

        # 使用 tqdm 进度条显示训练进度
        for batch in train_loader:
            images, masks = batch['image'], batch['mask']

            # 在train函數中修改
            images, masks = images.to(device), masks.to(device) # 如果使用GPU, 否则删除此行

            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 模型前向传播

            loss = criterion(outputs, masks)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

            # 计算 Dice Score
            dice = dice_score(outputs, masks)
            epoch_dice_score += dice.item()

        # 每个 epoch 结束后打印训练损失
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss / len(train_loader):.4f}, Dice Score: {epoch_dice_score / len(train_loader):.4f}")

        # 验证
        # 在訓練循環中修改為
        validate(model, valid_loader, criterion, device)

def validate(model, valid_loader, criterion, device):
    """
    验证模型，并计算 Dice Score
    """
    model.eval()  # 设为评估模式
    running_loss = 0.0
    epoch_dice_score = 0.0

    with torch.no_grad():  # 禁用梯度计算
        for batch in (valid_loader):
            images, masks = batch['image'], batch['mask']

            images, masks = images.to(device), masks.to(device)  # 如果使用GPU, 否则删除此行

            outputs = model(images)  # 模型前向传播
            loss = criterion(outputs, masks)  # 计算损失

            running_loss += loss.item()

            # 计算 Dice Score
            dice = dice_score(outputs, masks)
            epoch_dice_score += dice.item()

    print(f"Validation Loss: {running_loss / len(valid_loader):.4f}, Dice Score: {epoch_dice_score / len(valid_loader):.4f}")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, required=True, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)


    