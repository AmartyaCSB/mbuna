import os
from PIL import Image
from glob import glob


def downsize_images(source_dir: str, dest_dir: str, target_size: tuple[int, int]):
    """

    :param source_dir:
    :param dest_dir:
    :param target_size:
    """
    img_paths = glob(source_dir)
    n_imgs = len(img_paths)
    for i, p in enumerate(img_paths):
        print(f'{i}/{n_imgs}')
        img = Image.open(p)
        img = img.resize(target_size, Image.LANCZOS)
        img.save(os.path.join(dest_dir, os.path.split(p)[1]), optimize=True, quality=85)
