import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
from PIL import Image
import random
import imageio
import utils3d

from trellis.pipelines import TrellisVideoTo4DPipeline
from trellis.renderers import OctreeRenderer
from trellis.representations.octree import DfsOctree as Octree
from trellis.utils.general_utils import save_images_as_video
from trellis.utils import render_utils


parser = argparse.ArgumentParser(description="SS4D pipeline")
parser.add_argument('--output_dir', type=str, default="outputs")
opt = parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)           # For single GPU
    torch.cuda.manual_seed_all(seed)       # For multi-GPU

set_seed(42)


def render_coords(coords, exts, ints):
    renderer = OctreeRenderer()
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 4
    renderer.pipe.primitive = 'voxel'
    resolution = 64

    images = []
    for i, coord in enumerate(coords):
        representation = Octree(
            depth=10,
            aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
            device='cuda',
            primitive='voxel',
            sh_degree=0,
            primitive_config={'solid': True},
        )
        representation.position = coord.float() / resolution
        representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

        image = torch.zeros(3, 1024, 1024).cuda()
        tile = [2, 2]
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
            image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
        images.append(image)

    images = torch.stack(images)
    return images


def render_samples(samples, exts, ints):
    images = []
    for i, sample in enumerate(samples['gaussian']):
        frames = render_utils.render_frames(sample, exts, ints, {'resolution': 512, 'bg_color': (1, 1, 1), }, verbose=False)['color']
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        tile = [2, 2]
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            image[512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1), :] = frames[j]
        images.append(image)
    images = np.stack(images)
    return images


# pipeline = TrellisVideoTo4DPipeline.from_pretrained("lizb6626/SS4D")
pipeline = TrellisVideoTo4DPipeline.from_pretrained("/mnt/petrelfs/zhangmengchen/lizb/SS4D_ckpt") # load locally
pipeline.cuda()

output_dir = opt.output_dir
os.makedirs(output_dir, exist_ok=True)

yaws = [0 + np.pi / 6, np.pi / 2 + np.pi / 6, np.pi + np.pi / 6, 3 * np.pi / 2 + np.pi / 6]
pitch = [np.random.uniform(0, np.pi / 6) for _ in range(4)]

exts, ints = [], []
for yaw, pitch in zip(yaws, pitch):
    orig = torch.tensor([
        np.sin(yaw) * np.cos(pitch),
        np.cos(yaw) * np.cos(pitch),
        np.sin(pitch),
    ]).float().cuda() * 2
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    exts.append(extrinsics)
    ints.append(intrinsics)

for case_name in os.listdir("examples"):
    files = [f'{i}.png' for i in range(32)]

    images = [Image.open(os.path.join("examples", case_name, f)) for f in files]

    coords, samples, images_cond = pipeline.run(images, return_images=True)
    images_cond = np.array(images_cond)
    save_images_as_video(images_cond, os.path.join(output_dir, f"{case_name}_cond.mp4"))

    images = render_samples(samples, exts, ints)
    save_images_as_video(images, os.path.join(output_dir, f"{case_name}.mp4"))

    images = render_coords(coords, exts, ints)
    save_images_as_video(images, os.path.join(output_dir, f"{case_name}_coords.mp4"))
