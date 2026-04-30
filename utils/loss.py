import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from utils.DCP import get_atmosphere
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_var is None or self.real_label_var.shape != input.shape:
                self.real_label_var = torch.full_like(input, self.real_label.item())
            return self.real_label_var
        else:
            if self.fake_label_var is None or self.fake_label_var.shape != input.shape:
                self.fake_label_var = torch.full_like(input, self.fake_label.item())
            return self.fake_label_var

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        # x, y: (B, 3, H, W)
        # So sánh Mean (giá trị trung bình màu)
        mu_x = x.mean([2, 3])
        mu_y = y.mean([2, 3])

        # So sánh Std (độ lệch chuẩn - độ tương phản/phân bố màu)
        std_x = x.view(x.size(0), x.size(1), -1).std(2)
        std_y = y.view(y.size(0), y.size(1), -1).std(2)

        return self.l1(mu_x, mu_y) + self.l1(std_x, std_y)


def compute_dcp_loss(pred_a, hazy_img):
    """
    Tính loss giữa A dự đoán và A tính theo DCP.
    pred_a: (B, 3, 1, 1)
    hazy_img: (B, 3, H, W)
    """
    target_a = get_atmosphere(hazy_img)
    target_a = target_a.detach()

    return mse_loss(pred_a, target_a)

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.style_layers = {'0': 'relu1_1', '5': 'relu2_1', '10': 'relu3_1', '19': 'relu4_1', '28': 'relu5_1'}
        self.content_layers = {'22': 'relu4_2'}
        self.vgg_layers = vgg
        self.l1_loss = nn.L1Loss()

    def forward(self, generated, real_img, is_style=True):
        gen_features = self.get_features(generated)
        real_features = self.get_features(real_img)
        loss = 0.0

        layers = self.style_layers if is_style else self.content_layers
        for name in layers.keys():
            loss += self.l1_loss(gen_features[name], real_features[name])

        return loss

    def get_features(self, x):
        features = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.style_layers or name in self.content_layers:
                features[name] = x
        return features

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def get_disparity_smoothness(depth_map, ref_img):
    disp_gradients_x = gradient_x(depth_map)
    disp_gradients_y = gradient_y(depth_map)
    image_gradients_x = gradient_x(ref_img)
    image_gradients_y = gradient_y(ref_img)
    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y
    return torch.mean(torch.abs(smoothness_x)) / 2.0 + torch.mean(torch.abs(smoothness_y)) / 2.0

def get_t_loss(pred_t, ref_img_gray):
    l1 = F.smooth_l1_loss(pred_t, ref_img_gray)
    ssim_loss = 1 - ssim(pred_t, ref_img_gray)
    disp_loss = get_disparity_smoothness(pred_t, ref_img_gray)
    t_loss = 0.5 * (0.5 * l1 + 0.5 * ssim_loss) + 1.0 * disp_loss
    return t_loss