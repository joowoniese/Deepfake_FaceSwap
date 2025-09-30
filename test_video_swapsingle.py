import os
import glob
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap


def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if __name__ == '__main__':
    opt = TestOptions().parse()
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'

    # SimSwap ?? ??
    model = create_model(opt)
    model.eval()

    # ?? ??? ?? - buffalo_l ?? ?? ??
    app = Face_detect_crop(name='buffalo_l')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

    with torch.no_grad():
        pic_a_path = opt.pic_a_path.strip()
        img_id_list = []

        # ? ????? ?? ?? ??? ?? ?? ??
        if os.path.isdir(pic_a_path):
            image_exts = ['*.jpg', '*.jpeg', '*.png']
            pic_a_paths = []
            for ext in image_exts:
                pic_a_paths.extend(glob.glob(os.path.join(pic_a_path, ext)))
            pic_a_paths.sort()
        else:
            # ??? ?? ?? ??? ??? ?? ??
            pic_a_paths = pic_a_path.split(',')

        if not pic_a_paths:
            raise ValueError(f"No valid reference images found in: {pic_a_path}")

        for img_path in pic_a_paths:
            img_path = img_path.strip()
            img_a_whole = cv2.imread(img_path)
            if img_a_whole is None:
                print(f"?? Warning: Failed to load image: {img_path}")
                continue

            try:
                img_a_align_crop, _ = app.get(img_a_whole, crop_size)
                img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
                img_a = transformer_Arcface(img_a_align_crop_pil)
                img_id = img_a.unsqueeze(0).cuda()
                img_id_downsample = F.interpolate(img_id, size=(112, 112))
                latent_id = model.netArc(img_id_downsample)
                latent_id = F.normalize(latent_id, p=2, dim=1)
                img_id_list.append(latent_id)
            except Exception as e:
                print(f"? Failed to process {img_path}: {e}")
                continue

        if not img_id_list:
            raise ValueError("No valid face embeddings generated from reference images.")

        # ? ?? ??? ??
        latend_id = torch.stack(img_id_list, dim=0).mean(dim=0, keepdim=True)

        # ?? ?? ??? ??
        video_swap(
            opt.video_path,
            latend_id,
            model,
            app,
            opt.output_path,
            temp_results_dir=opt.temp_path,
            no_simswaplogo=opt.no_simswaplogo,
            use_mask=opt.use_mask,
            crop_size=crop_size
        )
