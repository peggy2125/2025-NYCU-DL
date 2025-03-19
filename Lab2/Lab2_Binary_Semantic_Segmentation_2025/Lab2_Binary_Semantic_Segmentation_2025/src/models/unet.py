import argparse
from tqdm import tqdm 
import torch
import torch.nn as nn



# Implement your UNet model here
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
# assert False, "Not implemented yet!"