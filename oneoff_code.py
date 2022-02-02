import os
from PIL import Image
from glob import glob
import cv2
import numpy as np
import fiftyone.brain as fob
import fiftyone as fo

def downsize_images(source_dir: str, dest_dir: str, target_size: tuple[int, int]):

    img_paths = glob(source_dir)
    n_imgs = len(img_paths)
    for i, p in enumerate(img_paths):
        print(f'{i}/{n_imgs}')
        img = Image.open(p)
        img = img.resize(target_size, Image.LANCZOS)
        img.save(os.path.join(dest_dir, os.path.split(p)[1]), optimize=True, quality=85)
