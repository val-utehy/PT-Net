import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5)) #創建一個大小為 5x5 的模糊核，並將每個元素初始化為 1/25 這裡使用了 NumPy 函數ones()
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])#將模糊核的維度轉換為1x1x5x5，這樣它就可以被當作卷積層的卷積核使用。這裡使用了NumPy 函數 reshape()。
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)#創建一個PyTorch 的參數，將模糊核轉換為PyTorch的張量格式
        image = np.zeros((5, 5))#創建了一個大小為 5x5 的 NumPy 數組，並且將所有元素初始化為0
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1]) #NumPy數組 image 的維度轉換為 1x1x5x5，這樣它就可以被當作卷積層的輸入
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False) #將 image 轉換為 PyTorch 的張量格式
        self.gray_scale = GrayscaleLayer() #將彩色圖片轉換為灰度圖片

    def forward(self, x):
        x = self.gray_scale(x)#將輸入圖片 x 通過灰度轉換層（GrayscaleLayer）轉換為灰度圖片
        return self.mse(F.conv2d(x, self.image), F.conv2d(x, self.blur))

class GrayscaleLayer(nn.Module):
    def __init__(self):
        super(GrayscaleLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, keepdim=True)
