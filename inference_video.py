import argparse
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
from net.HazeNet import HazeNet


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'])


def resize_keep_aspect(image, target_width):
    w, h = image.size
    ratio = h / w
    new_h = int(target_width * ratio)
    return image.resize((target_width, new_h), Image.Resampling.LANCZOS)


def inference_video(opt):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    netG = HazeNet(img_size=opt.image_size).to(device)

    if os.path.isfile(opt.ckpt_path):
        checkpoint = torch.load(opt.ckpt_path, map_location=device)
        netG.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {opt.ckpt_path}")
    else:
        print(f"Checkpoint not found: {opt.ckpt_path}")
        return

    netG.eval()
    netG.T_net.depth_anything.processor.do_rescale = False

    img_to_tensor = ToTensor()
    tensor_to_img = ToPILImage()

    style_files = sorted([f for f in os.listdir(opt.style_dir) if is_image_file(f)])
    if len(style_files) == 0:
        print(f"No style images found in {opt.style_dir}")
        return

    print(f"Found {len(style_files)} styles. Loading...")

    loaded_styles = []
    for s_file in style_files:
        try:
            s_path = os.path.join(opt.style_dir, s_file)
            s_pil = Image.open(s_path).convert('RGB')

            s_tensor = img_to_tensor(s_pil).unsqueeze(0).to(device)
            s_tensor = F.interpolate(s_tensor, size=(opt.image_size, opt.image_size), mode='bilinear',
                                     align_corners=False)
            s_tensor = torch.clamp(s_tensor, 0.0, 1.0)

            loaded_styles.append({
                'name': s_file,
                'tensor': s_tensor,
                'pil_thumb': resize_keep_aspect(s_pil, target_width=150)
            })
        except Exception as e:
            print(f"Skipping {s_file}: {e}")

    if not loaded_styles:
        return

    cap = cv2.VideoCapture(opt.input_video)
    if not cap.isOpened():
        print(f"Error opening video file: {opt.input_video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.splitext(os.path.basename(opt.input_video))[0]
    output_filename = f"{video_name}_multi_style.mp4"
    output_path = os.path.join(opt.output_dir, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (original_w, original_h))

    num_styles = len(loaded_styles)
    frames_per_style = total_frames / num_styles

    print(f"Processing video: {total_frames} frames with {num_styles} styles.")
    print(f"Approx {int(frames_per_style)} frames per style.")

    pbar = tqdm(total=total_frames)
    frame_idx = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        content_tensor = img_to_tensor(frame_pil).unsqueeze(0).to(device)
        content_tensor = F.interpolate(content_tensor, size=(opt.image_size, opt.image_size), mode='bilinear',
                                       align_corners=False)
        content_tensor = torch.clamp(content_tensor, 0.0, 1.0)

        style_idx = int(frame_idx / frames_per_style)
        if style_idx >= num_styles:
            style_idx = num_styles - 1

        current_style_data = loaded_styles[style_idx]

        start_frame_of_style = style_idx * frames_per_style
        progress = (frame_idx - start_frame_of_style) / frames_per_style
        progress = max(0.0, min(1.0, progress))

        alpha_val = progress

        with torch.no_grad():
            hazy_fake, _, _, _ = netG(
                current_style_data['tensor'],
                content_tensor,
                alpha=alpha_val,
                max_beta=opt.max_beta,
                uniform_haze=opt.uniform_haze_strength
            )

        hazy_fake = torch.clamp(hazy_fake, 0.0, 1.0)

        hazy_fake_resized = F.interpolate(hazy_fake, size=(original_h, original_w), mode='bilinear',
                                          align_corners=False)

        out_pil = tensor_to_img(hazy_fake_resized.squeeze(0).cpu())
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

        thumb = current_style_data['pil_thumb']
        thumb_w, thumb_h = thumb.size
        thumb_np = np.array(thumb)
        thumb_bgr = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2BGR)

        y_offset = 20
        x_offset = 20

        if y_offset + thumb_h < original_h and x_offset + thumb_w < original_w:
            out_bgr[y_offset:y_offset + thumb_h, x_offset:x_offset + thumb_w] = thumb_bgr

            cv2.rectangle(out_bgr, (x_offset, y_offset), (x_offset + thumb_w, y_offset + thumb_h), (0, 255, 0), 2)
            cv2.putText(out_bgr, f"Alpha: {alpha_val:.2f}", (x_offset, y_offset + thumb_h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out_writer.write(out_bgr)

        pbar.update(1)
        frame_idx += 1

        if frame_idx % 50 == 0:
            torch.cuda.empty_cache()

    cap.release()
    out_writer.release()
    pbar.close()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True)
    parser.add_argument('--style_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='inference_videos')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--max_beta', type=float, default=4)
    parser.add_argument('--uniform_haze_strength', type=float, default=0.5)

    args = parser.parse_args()
    inference_video(args)
