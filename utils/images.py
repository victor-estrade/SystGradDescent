# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def gather_images(model_directory):
    cv_dirs = [os.path.join(model_directory, d) for d in sorted(os.listdir(model_directory))]
    cv_dirs = [d for d in cv_dirs if os.path.isdir(d)]
    all_png_files = [os.path.join(d, file) 
                        for d in cv_dirs for file in os.listdir(d) 
                        if file.endswith(".png")]
    unique_png_names = set([os.path.basename(img) for img in all_png_files])
    same_images = [[img for img in all_png_files if os.path.basename(img) == unique_name] 
                for unique_name in unique_png_names]
    for same_img, name in zip(same_images, unique_png_names):
        concat_img = concat_images(same_img)
        name = "".join(name[:-4]+"_concat.png")
        fname = os.path.join(model_directory, name)
        concat_img.save(fname)


def concat_images(images):
    img_per_row = 3
    n_rows = len(images) // img_per_row
    n_rows = n_rows + 1 if len(images) % img_per_row != 0 else n_rows
    images = [draw_number(Image.open(img), i) for i, img in enumerate(images)]
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    total_width = max_width * img_per_row
    total_height = max_height * n_rows
    new_img = Image.new('RGB', (total_width, total_height))
    for i, img in enumerate(images):
        row = i // img_per_row
        col = i % img_per_row
        new_img.paste(img, (col * max_width, row * max_height))
    return new_img


def draw_number(img, i):
    draw = ImageDraw.Draw(img)
    font = load_ImageFont()
    draw.text((10, 10),"{:d}".format(i),(150,20,20),font=font)
    return img


def load_ImageFont():
    try :
        font = ImageFont.truetype("FiraSans-Regular.ttf", 24)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
    return font


