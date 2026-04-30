import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
from net.HazeNet import HazeNet

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'])

def inference_images(opt):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")

    netG = HazeNet(img_size=opt.image_size).to(device)

    if os.path.isfile(opt.ckpt_path):
        checkpoint = torch.load(opt.ckpt_path, map_location=device)
        netG.load_state_dict(checkpoint)
    else:
        return

    netG.eval()
    netG.T_net.depth_anything.processor.do_rescale = False

    img_to_tensor = ToTensor()
    tensor_to_img = ToPILImage()

    content_images = [x for x in os.listdir(opt.content_dir) if is_image_file(x)]
    style_images = [x for x in os.listdir(opt.style_dir) if is_image_file(x)]

    content_images.sort()
    style_images.sort()

    alpha_values = [round(x * 0.1, 1) for x in range(11)]

    for content_name in tqdm(content_images):
        content_path = os.path.join(opt.content_dir, content_name)
        try:
            content_pil = Image.open(content_path).convert('RGB')
        except Exception:
            continue

        content_tensor = img_to_tensor(content_pil).unsqueeze(0).to(device)
        content_tensor = F.interpolate(content_tensor, size=(opt.image_size, opt.image_size), mode='bilinear', align_corners=False)
        content_tensor = torch.clamp(content_tensor, 0.0, 1.0)

        for style_name in style_images:
            style_path = os.path.join(opt.style_dir, style_name)
            try:
                style_pil = Image.open(style_path).convert('RGB')
            except Exception:
                continue

            style_tensor = img_to_tensor(style_pil).unsqueeze(0).to(device)
            style_tensor = F.interpolate(style_tensor, size=(opt.image_size, opt.image_size), mode='bilinear', align_corners=False)
            style_tensor = torch.clamp(style_tensor, 0.0, 1.0)

            for alpha in alpha_values:
                with torch.no_grad():
                    hazy_fake, _, _, _ = netG(
                        style_tensor,
                        content_tensor,
                        alpha=alpha,
                        max_beta=opt.max_beta,
                        uniform_haze=opt.uniform_haze_strength
                    )

                hazy_fake = torch.clamp(hazy_fake, 0.0, 1.0)
                out_img_pil = tensor_to_img(hazy_fake.squeeze(0).cpu())

                c_name = os.path.splitext(content_name)[0]
                s_name = os.path.splitext(style_name)[0]

                out_filename = f"{c_name}_style_{s_name}_alpha_{alpha:.1f}.png"
                out_path = os.path.join(opt.output_dir, out_filename)

                out_img_pil.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--style_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--max_beta', type=float, default=3.5)
    parser.add_argument('--uniform_haze_strength', type=float, default=0.5)

    args = parser.parse_args()
    inference_images(args)