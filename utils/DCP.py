import numpy as np
import torch
from torch.nn.functional import pad

def get_dark_channel(image, w=15):
    """
    Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    B, M, N, _ = image.shape
    # padded = pad(image, ((0, 0), (w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    # padded = pad(image, (0, 0, w // 2, w // 2, w // 2, w // 2, 0, 0), 'edge')
    padded = pad(image, (0, 0, w // 2, w // 2, w // 2, w // 2))
    darkch = torch.zeros((B, M, N))
    for b, i, j in np.ndindex(darkch.shape):
        darkch[b, i, j] = torch.min(padded[b, i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
    w:      window for dark channel
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    image = image.permute(0, 2, 3, 1)
    # reference CVPR09, 4.4
    darkch = get_dark_channel(image, w)
    B, M, N = darkch.shape
    flatI = image.reshape(B, M * N, 3)
    # flatdark = darkch.ravel()
    flatdark = darkch.view(B, -1)
    # searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
    searchidx = torch.argsort(-flatdark)[:int(M * N * p)]  # find top M * N * p indexes
    # return the highest intensity for each channel
    return torch.max(torch.take(flatI, searchidx), axis=0)
