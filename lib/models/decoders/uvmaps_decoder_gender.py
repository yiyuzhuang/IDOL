import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from pytorch3d import ops

from lib.mmutils import xavier_init, constant_init
import numpy as np
import time
import cv2
import math
from simple_knn._C import distCUDA2
from pytorch3d.transforms import quaternion_to_matrix

from ..deformers import SMPLXDeformer_gender
from ..renderers import GRenderer, get_covariance, batch_rodrigues 
from lib.ops import TruncExp
import torchvision

from lib.utils.train_util import main_print

def ensure_dtype(input_tensor, target_dtype=torch.float32):
    """
    Ensure tensor dtype matches target dtype.
    If not, convert it.
    """
    if input_tensor.dtype != target_dtype:
        input_tensor = input_tensor.to(dtype=target_dtype)
    return input_tensor


class UVNDecoder_gender(nn.Module):

    activation_dict = {
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'trunc_exp': TruncExp,
        'sigmoid': nn.Sigmoid}

    def __init__(self,
                 *args,
                 interp_mode='bilinear',
                 base_layers=[3 * 32, 128],
                 density_layers=[128, 1],
                 color_layers=[128, 128, 3],
                 offset_layers=[128, 3],
                 scale_layers=[128, 3],
                 radius_layers=[128, 3],
                 use_dir_enc=True,
                 dir_layers=None,
                 scene_base_size=None,
                 scene_rand_dims=(0, 1),
                 activation='silu',
                 sigma_activation='sigmoid',
                 sigmoid_saturation=0.001,
                 code_dropout=0.0,
                 flip_z=False,
                 extend_z=False,
                 gender='neutral',
                 multires=0,
                 bg_color=0,
                 image_size=1024,
                 superres=False,
                 focal = 1280, # the default focal defination
                 reshape_type=None, # if true, it will create a cnn layers to upsample the uv features
                 fix_sigma=False, #  if true, the density of GS will be fixed
                 up_cnn_in_channels = None, #  the channel number of the upsample cnn
                 vithead_param=None, # the vit head for decode to uv features
                 is_sub2=False, # if true, will use the sub2 uv map
                 **kwargs):
        super().__init__()
        self.interp_mode = interp_mode
        self.in_chn = base_layers[0]
        self.use_dir_enc = use_dir_enc
        if scene_base_size is None:
            self.scene_base = None
        else:
            rand_size = [1 for _ in scene_base_size]
            for dim in scene_rand_dims:
                rand_size[dim] = scene_base_size[dim]
            init_base = torch.randn(rand_size).expand(scene_base_size).clone()
            self.scene_base = nn.Parameter(init_base)
        self.dir_encoder = None
        self.sigmoid_saturation = sigmoid_saturation
        self.deformer = SMPLXDeformer_gender(gender, is_sub2=is_sub2)

        self.renderer = GRenderer(image_size=image_size, bg_color=bg_color, f=focal)
        if superres:
            self.superres = None
        else:
            self.superres = None
        self.gender= gender
        self.reshape_type = reshape_type
        if reshape_type=='cnn':
            self.upsample_conv = torch.nn.ConvTranspose2d(512, 32, kernel_size=4, stride=4,).cuda()
                                                 
        elif reshape_type == 'VitHead': # changes the up block's layernorm into the feature channel norm instead of the full image norm
            from lib.models.decoders.vit_head import VitHead
            self.upsample_conv = VitHead(**vithead_param)
            # 256, 128, 128 -> 128, 256, 256 -> 64, 512, 512, ->32, 1024, 1024
        
        base_cache_dir = 'work_dirs/cache'   
        if is_sub2:
            base_cache_dir = 'work_dirs/cache_sub2'
            # main_print("!!!!!!!!!!!!!!!!!!! using the sub2 uv map !!!!!!!!!!!!!!!!!!!")
        if gender == 'neutral':
            select_uv = torch.as_tensor(np.load(base_cache_dir+'/init_uv_smplx_newNeutral.npy'))
            self.register_buffer('select_coord', select_uv.unsqueeze(0)*2.-1.)

            init_pcd = torch.as_tensor(np.load(base_cache_dir+'/init_pcd_smplx_newNeutral.npy'))
            self.register_buffer('init_pcd', init_pcd.unsqueeze(0), persistent=False) # 0.9-- -1
        elif gender == 'male':
            assert NotImplementedError("Haven't create the init_uv_smplx_thu in v_template")
            select_uv = torch.as_tensor(np.load(base_cache_dir+'/init_uv_smplx_thu.npy'))
            self.register_buffer('select_coord', select_uv.unsqueeze(0)*2.-1.)

            init_pcd = torch.as_tensor(np.load(base_cache_dir+'/init_pcd_smplx_thu.npy'))
            self.register_buffer('init_pcd', init_pcd.unsqueeze(0), persistent=False) # 0.9-- -1
        self.num_init = self.init_pcd.shape[1]
        main_print(f"!!!!!!!!!!!!!!!!!!! cur points number are {self.num_init} !!!!!!!!!!!!!!!!!!!")

        self.init_pcd = self.init_pcd 

        self.multires = multires # 0 Haven't 
        if multires > 0:
            uv_map = torch.as_tensor(np.load(base_cache_dir+'/init_uvmap_smplx_thu.npy'))
            pcd_map = torch.as_tensor(np.load(base_cache_dir+'/init_posmap_smplx_thu.npy'))
            input_coord = torch.cat([pcd_map, uv_map], dim=1)
            self.register_buffer('input_freq', input_coord, persistent=False)
            base_layers[0] += 5
            color_layers[0] += 5
        else:
            self.init_uv = None

        activation_layer = self.activation_dict[activation.lower()]


        base_net = [] # linear (in=18, out=64, bias=True)
        for i in range(len(base_layers) - 1):
            base_net.append(nn.Conv2d(base_layers[i], base_layers[i + 1], 3, padding=1))
            if i != len(base_layers) - 2:
                base_net.append(nn.BatchNorm2d(base_layers[i+1]))
                base_net.append(activation_layer())
        self.base_net = nn.Sequential(*base_net)
        self.base_bn = nn.BatchNorm2d(base_layers[-1])
        self.base_activation = activation_layer()

        density_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(density_layers) - 1):
            density_net.append(nn.Conv2d(density_layers[i], density_layers[i + 1], 1))
            if i != len(density_layers) - 2:
                density_net.append(nn.BatchNorm2d(density_layers[i+1]))
                density_net.append(activation_layer())
        density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.density_net = nn.Sequential(*density_net)

        offset_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(offset_layers) - 1):
            offset_net.append(nn.Conv2d(offset_layers[i], offset_layers[i + 1], 1))
            if i != len(offset_layers) - 2:
                offset_net.append(nn.BatchNorm2d(offset_layers[i+1]))
                offset_net.append(activation_layer())
        self.offset_net = nn.Sequential(*offset_net)

        self.dir_net = None
        color_net = [] # linear(in=64, out=3, bias=True), sigmoid
        for i in range(len(color_layers) - 2):
            color_net.append(nn.Conv2d(color_layers[i], color_layers[i + 1], kernel_size=3, padding=1))
            color_net.append(nn.BatchNorm2d(color_layers[i+1]))
            color_net.append(activation_layer())
        color_net.append(nn.Conv2d(color_layers[-2], color_layers[-1], kernel_size=1))
        color_net.append(nn.Sigmoid())
        self.color_net = nn.Sequential(*color_net)
        self.code_dropout = nn.Dropout2d(code_dropout) if code_dropout > 0 else None

        self.flip_z = flip_z
        self.extend_z = extend_z

        if self.gender == 'neutral':
            init_rot = torch.as_tensor(np.load(base_cache_dir+'/init_rot_smplx_newNeutral.npy'))
            self.register_buffer('init_rot', init_rot, persistent=False)

            face_mask = torch.as_tensor(np.load(base_cache_dir+'/face_mask_thu_newNeutral.npy'))
            self.register_buffer('face_mask', face_mask.unsqueeze(0), persistent=False)

            hands_mask = torch.as_tensor(np.load(base_cache_dir+'/hands_mask_thu_newNeutral.npy'))
            self.register_buffer('hands_mask', hands_mask.unsqueeze(0), persistent=False)

            outside_mask = torch.as_tensor(np.load(base_cache_dir+'/outside_mask_thu_newNeutral.npy'))
            self.register_buffer('outside_mask', outside_mask.unsqueeze(0), persistent=False)
        else:
            assert NotImplementedError("Haven't create the init_rot in v_template")
            init_rot = torch.as_tensor(np.load(base_cache_dir+'/init_rot_smplx_thu.npy'))
            self.register_buffer('init_rot', init_rot, persistent=False)

            face_mask = torch.as_tensor(np.load(base_cache_dir+'/face_mask_thu.npy'))
            self.register_buffer('face_mask', face_mask.unsqueeze(0), persistent=False)

            hands_mask = torch.as_tensor(np.load(base_cache_dir+'/hands_mask_thu.npy'))
            self.register_buffer('hands_mask', hands_mask.unsqueeze(0), persistent=False)

            outside_mask = torch.as_tensor(np.load(base_cache_dir+'/outside_mask_thu.npy'))
            self.register_buffer('outside_mask', outside_mask.unsqueeze(0), persistent=False)

        self.iter = 0
        # self.init_weights()
        self.if_rotate_gaussian = False
        self.fix_sigma = fix_sigma
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        if self.dir_net is not None:
            constant_init(self.dir_net[-1], 0)
        if self.offset_net is not None:
            self.offset_net[-1].weight.data.uniform_(-1e-5, 1e-5)
            self.offset_net[-1].bias.data.zero_()


    def extract_pcd(self, code, smpl_params, init=False, zeros_hands_off=False):
        '''
        Args:
            B == num_scenes
            code (tensor): latent code. shape: [B, C, H, W]
            smpl_params (tensor): SMPL parameters. shape: [B_pose, 189]
            init (bool): Not used
        Returns:
            defm_pcd (tensor): deformed point cloud. shape: [B, N, B_pose, 3]
            sigmas, rgbs, offset, radius, rot(tensor): GS attributes. shape: [B, N, C]
            tfs(tensor): deformation matrics. shape: [B, N, C]
        '''
        if isinstance(code, list):
            num_scenes, _, h, w = code[0].size()
        else:
            num_scenes, n_channels, h, w = code.size()
        init_pcd = self.init_pcd.repeat(num_scenes, 1, 1)  # T-posed space points, for computing the skinning weights
        
        sigmas, rgbs, radius, rot, offset = self._decode(code, init=init) #  the person-specify attributes of GS
        if self.fix_sigma:
            sigmas = torch.ones_like(sigmas)
        if zeros_hands_off:
            offset[self.hands_mask[...,None].expand(num_scenes, -1, 3)] = 0
        canon_pcd = init_pcd + offset
        
        self.deformer.prepare_deformer(smpl_params, num_scenes, device=canon_pcd.device)
        defm_pcd, tfs = self.deformer(canon_pcd, rot, mask=(self.face_mask+self.hands_mask+self.outside_mask), cano=False, if_rotate_gaussian=self.if_rotate_gaussian)
        return defm_pcd, sigmas, rgbs, offset, radius, tfs, rot

    def deform_pcd(self, code, smpl_params, init=False, zeros_hands_off=False, value=0.1):
        '''
        Args:
            B == num_scenes
            code (List): list of data
            smpl_params (tensor): SMPL parameters. shape: [B_pose, 189]
            init (bool): Not used
        Returns:
            defm_pcd (tensor): deformed point cloud. shape: [B, N, B_pose, 3]
            sigmas, rgbs, offset, radius, rot(tensor): GS attributes. shape: [B, N, C]
            tfs(tensor): deformation matrics. shape: [B, N, C]
        '''
        sigmas, rgbs, radius, rot, offset = code
        num_scenes = sigmas.shape[0]
        init_pcd = self.init_pcd.repeat(num_scenes, 1, 1)  #T-posed space points, for computing the skinning weights

        if self.fix_sigma:
            sigmas = torch.ones_like(sigmas)
        if zeros_hands_off:
            offset[self.hands_mask[...,None].expand(num_scenes, -1, 3)] = torch.clamp(offset[self.hands_mask[...,None].expand(num_scenes, -1, 3)], -value, value)
        canon_pcd = init_pcd + offset
        self.deformer.prepare_deformer(smpl_params, num_scenes, device=canon_pcd.device)
        defm_pcd, tfs = self.deformer(canon_pcd, rot, mask=(self.face_mask+self.hands_mask+self.outside_mask), cano=False, if_rotate_gaussian=self.if_rotate_gaussian)
        return defm_pcd, sigmas, rgbs, offset, radius, tfs, rot


        
    def _sample_feature(self,results,):
        # outputs, sigma_uv, offset_uv, rgbs_uv, radius_uv, rot_uv = results['output'], results['sigma'], results['offset'], results['rgbs'], results['radius'], results['rot']
        sigma = results['sigma']
        outputs = results['output']
        if isinstance(sigma, list):
            num_scenes, _, h, w = sigma[0].shape
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
        elif sigma.dim() == 4:
            num_scenes, n_channels, h, w = sigma.shape
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
        else:
            assert False
        output_attr = F.grid_sample(outputs, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, 13, -1).permute(0, 2, 1)
        sigma, offset, rgbs, radius, rot = output_attr.split([1, 3, 3, 3, 3], dim=2)

        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        
        radius = (radius - 0.5) * 2
        rot = (rot - 0.5) * np.pi

        return sigma, rgbs, radius, rot, offset

    def _decode_feature(self, point_code, low_ram=False):
        if isinstance(point_code, list):
            num_scenes, _, h, w = point_code[0].shape
            geo_code, tex_code = point_code
            # select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
            if self.multires != 0:
                input_freq = self.input_freq.repeat(num_scenes, 1, 1, 1)
        elif point_code.dim() == 4:
            num_scenes, n_channels, h, w = point_code.shape
            # select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
            if self.multires != 0:
                input_freq = self.input_freq.repeat(num_scenes, 1, 1, 1)
            geo_code, tex_code = point_code.split(16, dim=1)
        else:
            assert False

        base_in = geo_code if self.multires == 0 else torch.cat([geo_code, input_freq], dim=1)
        base_x = self.base_net(base_in)
        base_bn = self.base_bn(base_x)
        
        if low_ram:
            del base_x
            torch.cuda.empty_cache()

        base_x_act = self.base_activation(base_bn)

        if low_ram:
            del base_bn
            torch.cuda.empty_cache()
    
        sigma = self.density_net(base_x_act)
        offset = self.offset_net(base_x_act)

        if low_ram:
            del base_x_act
            torch.cuda.empty_cache()

        color_in = tex_code if self.multires == 0 else torch.cat([tex_code, input_freq], dim=1)
        rgbs_radius_rot = self.color_net(color_in)
        
        outputs = torch.cat([sigma, offset, rgbs_radius_rot], dim=1)

        if low_ram:
            del color_in
            del rgbs_radius_rot
            torch.cuda.empty_cache()
        
        main_print(outputs.shape)
        sigma, offset, rgbs, radius, rot = outputs.split([1, 3, 3, 3, 3], dim=1)
        results = {'output':outputs, 'sigma': sigma, 'offset': offset, 'rgbs': rgbs, 'radius': radius, 'rot': rot}

        return results
    def _decode(self, point_code, init=False):
        if isinstance(point_code, list):
            num_scenes, _, h, w = point_code[0].shape
            geo_code, tex_code = point_code
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
            if self.multires != 0:
                input_freq = self.input_freq.repeat(num_scenes, 1, 1, 1)
        elif point_code.dim() == 4:
            num_scenes, n_channels, h, w = point_code.shape
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
            if self.multires != 0:
                input_freq = self.input_freq.repeat(num_scenes, 1, 1, 1)
            geo_code, tex_code = point_code.split(16, dim=1)
        else:
            assert False

        base_in = geo_code if self.multires == 0 else torch.cat([geo_code, input_freq], dim=1)
        base_x = self.base_net(base_in)
        base_x_act = self.base_activation(self.base_bn(base_x))
     
        sigma = self.density_net(base_x_act)
        offset = self.offset_net(base_x_act)
        color_in = tex_code if self.multires == 0 else torch.cat([tex_code, input_freq], dim=1)
        rgbs_radius_rot = self.color_net(color_in)
        
        outputs = torch.cat([sigma, offset, rgbs_radius_rot], dim=1)
        output_attr = F.grid_sample(outputs, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, 13, -1).permute(0, 2, 1)
        sigma, offset, rgbs, radius, rot = output_attr.split([1, 3, 3, 3, 3], dim=2)

        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        
        radius = (radius - 0.5) * 2
        rot = (rot - 0.5) * np.pi

        return sigma, rgbs, radius, rot, offset

    def gaussian_render(self, pcd, sigmas, rgbs, normals, rot, num_scenes, num_imgs, cameras, use_scale=False, radius=None, \
                         return_norm=False, return_viz=False, mask=None):
        # add mask or visible points to images or select ind to images
        '''
           render the gaussian to images
           return_norm: return the normals of the gaussian (haven't been used)
           return_viz: return the mask of the gaussian
           mask: the mask of the gaussian
        '''
        assert num_scenes == 1
        
        pcd = pcd.reshape(-1, 3)
        if use_scale: 
            dist2 = distCUDA2(pcd)
            dist2 = torch.clamp_min((dist2), 0.0000001)
            scales = torch.sqrt(dist2)[...,None].repeat(1, 3).detach() # distence between different points
            scale = (radius+1)*scales  # scaling_modifier # radius[-1--1], scale of GS
            cov3D = get_covariance(scale, rot).reshape(-1, 6) # inputs rot is the rotations
       
        images_all = []
        viz_masks = [] if return_viz else None
        norm_all = [] if return_norm else None

        if mask != None:
            pcd = pcd[mask]
            rgbs = rgbs[mask]
            sigmas = sigmas[mask]
            cov3D = cov3D[mask]
            normals = normals[mask]
        if 1:
            for i in range(num_imgs):
                self.renderer.prepare(cameras[i])

                image = self.renderer.render_gaussian(means3D=pcd, colors_precomp=rgbs, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                images_all.append(image)
                if return_viz:
                    viz_mask = self.renderer.render_gaussian(means3D=pcd, colors_precomp=pcd.clone(), 
                        rotations=None, opacities=sigmas*0+1, scales=None, cov3D_precomp=cov3D)
                    viz_masks.append(viz_mask)
          

        images_all = torch.stack(images_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        if return_viz:
            viz_masks = torch.stack(viz_masks, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2).reshape(1, -1, 3)
            dist_sq, idx, neighbors = ops.knn_points(pcd.unsqueeze(0), viz_masks[:, ::10], K=1, return_nn=True)
            viz_masks = (dist_sq < 0.0001)[0]
         # ===== END the original code for batch size = 1 =====
        if use_scale:
            return images_all, norm_all, viz_masks, scale
        else:
            return images_all, norm_all, viz_masks, None

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        num_scenes, num_chn, h, w = code.size()
        code_viz = code.reshape(num_scenes, 4, 8, h, w).to(torch.float32).cpu().numpy()
        if not self.flip_z:
            code_viz = code_viz[..., ::-1, :]
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 4 * h, 8 * w)
        for code_single, code_viz_single, scene_name_single in zip(code, code_viz, scene_name):
            plt.imsave(os.path.join(viz_dir, 'a_scene_' + scene_name_single + '.png'), code_viz_single,
                       vmin=code_range[0], vmax=code_range[1])

    def forward(self, code, smpl_params, cameras, num_imgs,
                return_loss=False, return_norm=False, init=False, mask=None, zeros_hands_off=False):
        """
        Args:

          
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
            YY:
            grid_size, dt_gamma, perturb, T_thresh are deleted
            code: Shape (num_scenes, *code_size)
            cameras: Shape (num_scenes, num_imgs, 19(3+16))
            smpl_params: Shape (num_scenes, 189)

        """
        # import ipdb; ipdb.set_trace()
        if isinstance(code, list):
            num_scenes = len(code[0])
        else:
            num_scenes = len(code)
        assert num_scenes > 0
        self.iter+=1

        image = []
        scales = []
        norm = [] if return_norm else None
        viz_masks = [] if not self.training else None

        xyzs, sigmas, rgbs, offsets, radius, tfs, rot = self.extract_pcd(code, smpl_params, init=init, zeros_hands_off=zeros_hands_off)

        if zeros_hands_off:
            main_print('zeros_hands_off is on!')
            main_print('zeros_hands_off is on!')
            offsets[self.hands_mask[...,None].repeat(num_scenes, 1, 3)] = 0
            rgbs[self.hands_mask[...,None].repeat(num_scenes, 1, 3)] = 0
            rgbs[self.hands_mask[...,None].repeat(num_scenes, 1, 3)] = 0
        R_delta = batch_rodrigues(rot.reshape(-1, 3))
        R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
        R_def = torch.bmm(tfs.flatten(0, 1)[:, :3, :3], R)
        normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
        R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
      
        return_to_bfloat16 = True if xyzs.dtype==torch.bfloat16 else False ####### ============ translate the output to BF16 =================
        # return_to_bfloat16 = False # I don't want to trans it back to bf16
        if return_to_bfloat16:
              main_print("changes the return_to_bfloat16")
              cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius = [ensure_dtype(item, torch.float32) for item in (cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius)]
        # with torch.amp.autocast(enabled=False, device_type='cuda'):
        if 1:
            for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                image_single, norm_single,  viz_mask, scale = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, \
                                                                        radius=radius_single, return_norm=return_norm, return_viz=not self.training)
                image.append(image_single)
                scales.append(scale.unsqueeze(0))
                if return_norm:
                    norm.append(norm_single)
                if not self.training:
                    viz_masks.append(viz_mask)
        image = torch.cat(image, dim=0)
        scales = torch.cat(scales, dim=0)

        norm = torch.cat(norm, dim=0) if return_norm else None
        viz_masks = torch.cat(viz_masks, dim=0) if  (not self.training) and viz_masks else None

      
        main_print("not trans the rendered results to float16")
        if False:
            image = image.to(torch.bfloat16)
            scales = scales.to(torch.bfloat16)
            if return_norm:
                norm = norm.to(torch.bfloat16)
            if viz_masks is not None:
                viz_masks = viz_masks.to(torch.bfloat16)
            offsets = offsets.to(torch.bfloat16)

        if self.training:
            offset_dist = offsets ** 2
            weighted_offset = torch.mean(offset_dist) + torch.mean(offset_dist[self.hands_mask.repeat(num_scenes, 1)]) #+ torch.mean(offset_dist[self.face_mask.repeat(num_scenes, 1)])
        else:
            weighted_offset = offsets
        

        results = dict(
            viz_masks=viz_masks,
            scales=scales,
            norm=norm,
            image=image,
            offset=weighted_offset)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results
    
    def forward_render(self, code, cameras, num_imgs,
                return_loss=False, return_norm=False, init=False, mask=None, zeros_hands_off=False):
        """
        Args:

        
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
            YY:
            grid_size, dt_gamma, perturb, T_thresh are deleted
            code: Shape (num_scenes, *code_size)
            cameras: Shape (num_scenes, num_imgs, 19(3+16))
            smpl_params: Shape (num_scenes, 189)

        """
        image = []
        scales = []
        norm = [] if return_norm else None
        viz_masks = [] if not self.training else None

        xyzs, sigmas, rgbs, offsets, radius, tfs, rot =  code
        num_scenes = xyzs.shape[0]
        if zeros_hands_off:
            main_print('zeros_hands_off is on!')
            main_print('zeros_hands_off is on!')
            offsets[self.hands_mask[...,None].repeat(num_scenes, 1, 3)] = 0
            rgbs[self.hands_mask[...,None].repeat(num_scenes, 1, 3)] = 0
            rgbs[self.hands_mask[...,None].repeat(num_scenes, 1, 3)] = 0
        R_delta = batch_rodrigues(rot.reshape(-1, 3))
        R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
        R_def = torch.bmm(tfs.flatten(0, 1)[:, :3, :3], R)
        normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
        R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
        # import ipdb; ipdb.set_trace()
    
        return_to_bfloat16 = True if xyzs.dtype==torch.bfloat16 else False ####### ============ translate the output to BF16 =================
        # return_to_bfloat16 = False # I don't want to trans it back to bf16
        if return_to_bfloat16:
            main_print("changes the return_to_bfloat16")
            cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius = [ensure_dtype(item, torch.float32) for item in (cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius)]

        if 1:
            for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                image_single, norm_single,  viz_mask, scale = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, \
                                                                        radius=radius_single, return_norm=return_norm, return_viz=not self.training)
                image.append(image_single)
                scales.append(scale.unsqueeze(0))
                if return_norm:
                    norm.append(norm_single)
                if not self.training:
                    viz_masks.append(viz_mask)
        image = torch.cat(image, dim=0)
        scales = torch.cat(scales, dim=0)

        norm = torch.cat(norm, dim=0) if return_norm else None
        viz_masks = torch.cat(viz_masks, dim=0) if  (not self.training) and viz_masks else None


        main_print("not trans the rendered results to float16")
        if False:
            image = image.to(torch.bfloat16)
            scales = scales.to(torch.bfloat16)
            if return_norm:
                norm = norm.to(torch.bfloat16)
            if viz_masks is not None:
                viz_masks = viz_masks.to(torch.bfloat16)
            offsets = offsets.to(torch.bfloat16)

        if self.training:
            offset_dist = offsets ** 2
            weighted_offset = torch.mean(offset_dist) + torch.mean(offset_dist[self.hands_mask.repeat(num_scenes, 1)]) #+ torch.mean(offset_dist[self.face_mask.repeat(num_scenes, 1)])
        else:
            weighted_offset = offsets
        

        results = dict(
            viz_masks=viz_masks,
            scales=scales,
            norm=norm,
            image=image,
            offset=weighted_offset)
        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results
    
    
    def forward_testing_time(self, code, smpl_params, cameras, num_imgs,
                return_loss=False, return_norm=False, init=False, mask=None, zeros_hands_off=False):
        """
        Args:

          
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
            YY:
            grid_size, dt_gamma, perturb, T_thresh are deleted
            code: Shape (num_scenes, *code_size)
            cameras: Shape (num_scenes, num_imgs, 19(3+16))
            smpl_params: Shape (num_scenes, 189)

        """
        if isinstance(code, list):
            num_scenes = len(code[0])
        else:
            num_scenes = len(code)
        assert num_scenes > 0
        self.iter+=1

        image = []
        scales = []
        norm = [] if return_norm else None
        viz_masks = [] if not self.training else None
        start_time = time.time()
        xyzs, sigmas, rgbs, offsets, radius, tfs, rot = self.extract_pcd(code, smpl_params, init=init, zeros_hands_off=zeros_hands_off)
        end_time_to_3D = time.time()
        time_code_to_3d = end_time_to_3D- start_time 

        R_delta = batch_rodrigues(rot.reshape(-1, 3))
        R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
        R_def = torch.bmm(tfs.flatten(0, 1)[:, :3, :3], R)
        normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
        R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
        if 1:
            for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                image_single, norm_single,  viz_mask, scale = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, \
                                                                        radius=radius_single, return_norm=False, return_viz=not self.training)
                image.append(image_single)
                scales.append(scale.unsqueeze(0))
                if return_norm:
                    norm.append(norm_single)
                if not self.training:
                    viz_masks.append(viz_mask)
        image = torch.cat(image, dim=0)
        scales = torch.cat(scales, dim=0)

        norm = torch.cat(norm, dim=0) if return_norm else None
        viz_masks = torch.cat(viz_masks, dim=0) if  (not self.training) and viz_masks else None

        time_3D_to_img = time.time() - end_time_to_3D


        if False:
            image = image.to(torch.bfloat16)
            scales = scales.to(torch.bfloat16)
            if return_norm:
                norm = norm.to(torch.bfloat16)
            if viz_masks is not None:
                viz_masks = viz_masks.to(torch.bfloat16)
            offsets = offsets.to(torch.bfloat16)

        results = dict(
            image=image)

        return results, time_code_to_3d, time_3D_to_img