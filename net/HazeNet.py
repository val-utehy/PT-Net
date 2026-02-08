import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthAnythingModule(nn.Module):
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", device='cuda'):
        super().__init__()
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.processor.do_rescale = False
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)

        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=x.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        prediction = torch.clamp(prediction, min=0.0)

        if torch.isnan(prediction).any() or torch.isinf(prediction).any():
            prediction = torch.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0)

        flat_pred = prediction.view(prediction.size(0), -1)

        min_val = torch.min(flat_pred, dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_val = torch.max(flat_pred, dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)

        denominator = max_val - min_val
        denominator = torch.clamp(denominator, min=1e-5)

        normalized_depth = (prediction - min_val) / denominator

        normalized_depth = torch.clamp(normalized_depth, 0.0, 1.0)

        return normalized_depth


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            raise ValueError(f"ERROR MODE {idx}")
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]
        H = W = self.resolution
        B, L, C = q.shape
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)
        return x


class CSWinBlock(nn.Module):
    def __init__(self, dim, reso, num_heads, split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 1 if last_stage or self.patches_resolution == split_size else 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(dim, resolution=self.patches_resolution, idx=-1, split_size=split_size,
                              num_heads=num_heads, dim_out=dim, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(dim // 2, resolution=self.patches_resolution, idx=i, split_size=split_size,
                              num_heads=num_heads // 2, dim_out=dim // 2, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        H = W = self.patches_resolution
        B, L, C = x.shape
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x


class CSWinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, depth=[2, 4, 32, 2],
                 split_size=[1, 2, 7, 7],
                 num_heads=[4, 8, 16, 32], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )
        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=num_heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.stage1_conv_embed(x)
        for blk in self.stage1:
            x = blk(x)

        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale=2):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.block(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim, context_dim, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Conv2d(query_dim, embed_dim, 1, bias=False)
        self.to_k = nn.Conv2d(context_dim, embed_dim, 1, bias=False)
        self.to_v = nn.Conv2d(context_dim, embed_dim, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(embed_dim, context_dim, 1),
            nn.BatchNorm2d(context_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, query, context):
        B, _, H, W = query.shape

        q = self.to_q(query)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)

        attn_scores = torch.matmul(q, k) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, v)
        output = output.transpose(-1, -2).reshape(B, self.num_heads * self.head_dim, H, W)

        output = self.to_out(output)

        return context + output


class DepthFormer_TEN(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.depth_anything = DepthAnythingModule()
        for param in self.depth_anything.parameters():
            param.requires_grad = False

        self.cswin = CSWinTransformer(
            img_size=img_size,
            embed_dim=96,
            depth=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            split_size=[1, 2, 7, 7]
        )

        cswin_feature_channels = 96
        depth_feature_channels = 1

        self.fusion_module = CrossAttentionFusion(
            query_dim=depth_feature_channels,
            context_dim=cswin_feature_channels,
            embed_dim=cswin_feature_channels
        )

        self.decoder = nn.Sequential(
            DecoderBlock(cswin_feature_channels, cswin_feature_channels // 2),
            DecoderBlock(cswin_feature_channels // 2, cswin_feature_channels // 4),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(cswin_feature_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        initial_depth_map = self.depth_anything(x)

        cswin_features = self.cswin.forward_features(x)

        depth_map_resized = F.interpolate(
            initial_depth_map,
            size=cswin_features.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        fused_features = self.fusion_module(depth_map_resized, cswin_features)

        decoded_features = self.decoder(fused_features)

        final_transmission_map = self.final_conv(decoded_features)

        final_transmission_map = F.interpolate(
            final_transmission_map,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )

        return final_transmission_map, initial_depth_map, cswin_features


class AdvancedAirlightNet(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.cswin_encoder = CSWinTransformer(
            img_size=img_size,
            embed_dim=96,
            depth=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            split_size=[1, 2, 7, 7]
        )

        cswin_feature_channels = 96

        self.sky_mask_head = nn.Sequential(
            nn.Conv2d(cswin_feature_channels, cswin_feature_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(cswin_feature_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, hazy_img):
        features = self.cswin_encoder.forward_features(hazy_img)

        object_mask = self.sky_mask_head(features)
        object_mask_resized = F.interpolate(
            object_mask,
            size=hazy_img.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        sky_mask_real = 1.0 - object_mask_resized

        combined_weight = sky_mask_real
        masked_hazy = hazy_img * combined_weight

        sum_pixels = torch.sum(masked_hazy, dim=[2, 3])
        count_pixels = torch.sum(combined_weight, dim=[2, 3])
        count_pixels = torch.clamp(count_pixels, min=1e-5)

        airlight = sum_pixels / count_pixels
        airlight = torch.clamp(airlight, 0.05, 1.0)

        return airlight.view(hazy_img.size(0), 3, 1, 1)


class HazeNet(nn.Module):
    def __init__(self, in_dim=3, img_size=224):
        super(HazeNet, self).__init__()
        self.T_net = DepthFormer_TEN(img_size=img_size)
        self.A_net = AdvancedAirlightNet(img_size=img_size)

    def forward(self, hazy_img, clean_img, alpha=1.0, max_beta=3.0, uniform_haze=0.5):
        final_T, initial_depth, features = self.T_net(clean_img)

        current_alpha = alpha
        if not isinstance(current_alpha, torch.Tensor):
            current_alpha = torch.tensor(
                current_alpha,
                device=clean_img.device,
                dtype=clean_img.dtype
            )

        if current_alpha.ndim == 0:
            current_alpha = current_alpha.view(1, 1, 1, 1)
        elif current_alpha.ndim == 1:
            current_alpha = current_alpha.view(-1, 1, 1, 1)

        beta = current_alpha * max_beta
        beta_float = torch.clamp(beta.float(), min=0.0, max=3.0)

        A = self.A_net(hazy_img)
        A = torch.clamp(A, 0.05, 1.0)

        final_T_safe = torch.clamp(final_T * 0.95, min=0.05, max=0.99)
        final_T_safe_float = final_T_safe.float()

        final_T_powered = torch.pow(final_T_safe_float, beta_float)

        max_transmission = 1.0 - (uniform_haze * current_alpha)
        if not isinstance(max_transmission, torch.Tensor):
            max_transmission = torch.tensor(
                max_transmission,
                device=clean_img.device,
                dtype=clean_img.dtype
            )

        max_transmission = torch.clamp(max_transmission, min=0.2, max=1.0)
        final_T_powered = final_T_powered * max_transmission
        final_T_powered = torch.clamp(final_T_powered, min=0.05, max=1.0)

        term_J = clean_img.float() * final_T_powered
        term_A = A.float() * (1 - final_T_powered)

        out = term_J + term_A
        out = torch.clamp(out, 0.0, 1.0)

        return out, final_T, A, initial_depth