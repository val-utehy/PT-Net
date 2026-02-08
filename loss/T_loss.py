import torch.nn as nn
from utils.pytorch_ssim import ssim as ssim
from loss.blur_loss import StdLoss
from loss.disp_loss import get_disparity_loss
from torch.nn.functional import smooth_l1_loss


def get_t_loss(pred_img, ref_img):
    l1 = smooth_l1_loss(pred_img, ref_img)
    ssim_loss = 1 - ssim(pred_img, ref_img)
    disp_loss = get_disparity_loss(pred_img, ref_img)

    t_loss = 0.5 * (0.5 * l1 + 0.5 * ssim_loss) + 1.0 * disp_loss
    # t_loss = 0.4 * (0.5 * l1 + 0.5 * ssim_loss) + 1.0 * disp_loss

    return t_loss

