import os
import json
from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisVideoTo4DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisVideoTo4DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisVideoTo4DPipeline, TrellisVideoTo4DPipeline).from_pretrained(path, skip=False)
        new_pipeline = TrellisVideoTo4DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        if os.getenv("EXP_PLATFORM", "other") == "huoshanyun":
            dinov2_model = torch.hub.load(
                '/fs-computility/mllm/lizhibing/.cache/torch/hub/facebookresearch_dinov2_main',
                'dinov2_vitl14_reg', source="local", pretrained=True)
        else:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_images(self, inputs: list[Image.Image]) -> list[Image.Image]:
        for image in inputs:
            assert image.mode == 'RGBA', "Input image should be RGBA"
        alphas = np.array([image.getchannel(3) for image in inputs])
        bboxes = alphas.nonzero()
        bboxes = [bboxes[2].min(), bboxes[1].min(), bboxes[2].max(), bboxes[1].max()]
        center = [(bboxes[0] + bboxes[2]) / 2, (bboxes[1] + bboxes[3]) / 2]
        hsize = max(bboxes[2] - bboxes[0], bboxes[3] - bboxes[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        outputs = [image.crop(aug_bbox) for image in inputs]

        outputs = [image.resize((518, 518), Image.Resampling.LANCZOS) for image in outputs]
        outputs = np.array(outputs).astype(np.float32) / 255
        outputs = outputs[:, :, :, :3] * outputs[:, :, :, 3:4]
        outputs = [Image.fromarray((output * 255).astype(np.uint8)) for output in outputs]
        return outputs

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        num_frames: int,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            raise NotImplementedError("Mesh decoding is not implemented yet.")
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat, num_frames)
        if 'radiance_field' in formats:
            raise NotImplementedError("Radiance field decoding is not implemented yet.")
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decompose_coords(self, coords, num_samples):
        seq_len = torch.bincount(coords[:, 0], minlength=num_samples)
        offset = torch.cumsum(seq_len, dim=0)
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(num_samples)]
        coords = coords[:, 1:]
        coords = [coords[layout[i]] for i in range(num_samples)]
        return coords

    @torch.no_grad()
    def run(
        self,
        images: List[Image.Image],
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['gaussian'],
        preprocess_image: bool = True,
        coords: Optional[torch.Tensor] = None,
        return_images: bool = False,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = self.preprocess_images(images)
        cond = self.get_cond(images)
        torch.manual_seed(seed)
        num_frames = len(images)
        if coords is None:
            sparse_structure_sampler_params["num_frames"] = len(images)
            coords = self.sample_sparse_structure(cond, num_frames, sparse_structure_sampler_params)

        slat_sampler_params["num_frames"] = num_frames
        slat = self.sample_slat(cond, coords, slat_sampler_params)

        samples = self.decode_slat(slat, num_frames, formats)
        coords = self.decompose_coords(coords, num_frames)
        if return_images:
            return coords, samples, images
        return coords, samples

    @torch.no_grad()
    def run_ss(
        self,
        images: List[Image.Image],
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_images: bool = False,
    ) -> list[torch.Tensor]:
        if preprocess_image:
            images = self.preprocess_images(images)
        cond = self.get_cond(images)
        torch.manual_seed(seed)
        num_samples = len(images)
        sparse_structure_sampler_params["num_frames"] = len(images)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)

        coords = self.decompose_coords(coords, num_samples)
        if return_images:
            return coords, images
        return coords
