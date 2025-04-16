
import os
import math
import json
from torch.optim import Adam
from torch.nn.parallel.distributed import DistributedDataParallel
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import pytorch_lightning as pl
from pytorch_lightning.utilities.grads import grad_norm
from einops import rearrange, repeat

from lib.utils.train_util import instantiate_from_config
from lib.ops.activation import TruncExp
import time
import matplotlib.pyplot as plt

from PIL import Image

import numpy as np 
from lib.utils.train_util import main_print

from typing import List, Optional, Tuple, Union
def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.Tensor, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`torch.Tensor` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    theta = theta * ntk_factor
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) / linear_factor  # [D/2]
    t = pos # torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
    freqs = freqs.to(device=t.device, dtype=t.dtype)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis
class FluxPosEmbed(torch.nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: [int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        freqs_dtype = torch.float32 if is_mps else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i], repeat_interleave_real=True, use_real=True#, freqs_dtype=freqs_dtype
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin
    
class SapiensGS_SA_v1(pl.LightningModule):
    
    def __init__(
        self,
        encoder=dict(type='mmpretrain.VisionTransformer'),
        neck=dict(type='mmpretrain.VisionTransformer'),
        decoder=dict(),
        diffusion_use_ema=True,
        freeze_decoder=False,
        image_cond=False,
        code_permute=None,
        code_reshape=None,
        autocast_dtype=None,
        ortho=True,
        return_norm=False,
        # reshape_type='reshape', # 'cnn'
        code_size=None,
        decoder_use_ema=None,
        bg_color=1,
        training_mode=None, # stage2's flag, default None for stage1 

        patch_size: int = 4,
            
        warmup_steps: int = 12_000,
        use_checkpoint: bool = True,
        lambda_depth_tv: float = 0.05,
        lambda_lpips: float = 2.0,
        lambda_mse: float = 1.0,
        lambda_l1: float=0, 
        lambda_ssim: float=0, 
        neck_learning_rate: float = 5e-4,
        decoder_learning_rate: float = 1e-3,
        encoder_learning_rate: float=0,
        max_steps: int = 100_000,
        loss_coef: float = 0.5,
        init_iter: int = 500,
        lambda_offset: int = 50,  # offset_weight: 50
        scale_weight: float = 0.01,
        is_debug: bool = False, # if debug, then it will not returns lpips
        code_activation: dict=None,
        output_hidden_states: bool=False, # if True, will output the hidden states from sapiens shallow layer, for the neck decoder
        loss_weights_views: List = [], # the loss weights for the views, if empty, will use the same weights for all the views
        **kwargs
    ):
        super(SapiensGS_SA_v1, self).__init__()
        ## ========== part -- Add the code to save this parameters for optimizers ========
        self.warmup_steps = warmup_steps
        self.use_checkpoint = use_checkpoint
        self.lambda_depth_tv = lambda_depth_tv
        self.lambda_lpips = lambda_lpips
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim  
        self.neck_learning_rate = neck_learning_rate
        self.decoder_learning_rate = decoder_learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.max_steps = max_steps

        self.loss_coef = loss_coef
        self.init_iter = init_iter
        self.lambda_offset = lambda_offset
        self.scale_weight = scale_weight

        self.is_debug = is_debug
        ## ========== end part ========

     
        
        self.code_size = code_size
        if code_activation['type'] == 'tanh':
            self.code_activation = torch.nn.Tanh()
        else:
            self.code_activation = TruncExp() #build_module(code_activation)
        # self.grid_size = grid_size
        self.decoder = instantiate_from_config(decoder)
        self.decoder_use_ema = decoder_use_ema
        if decoder_use_ema:
            raise NotImplementedError("decoder_use_ema has not been implemented")
            if self.decoder_use_ema:
                self.decoder_ema = deepcopy(self.decoder)
        self.encoder = instantiate_from_config(encoder)
        # get_obj_from_str(config["target"])

        self.code_size = code_reshape
        self.code_clip_range = [-1,1]
        
        # ============= begin config  =============
        # transformer from class MAEPretrainDecoder(BaseModule):
         # compress the token number of the uv code
        self.patch_size = patch_size
        self.code_patch_size = self.patch_size
        self.num_patches_axis = code_reshape[-1]//self.patch_size # reshape it for the upsampling
        self.num_patches = self.num_patches_axis ** 2
        self.code_feat_dims = code_reshape[0] # only used for the upsampling of 'reshape' type 
        self.code_resolution = code_reshape[-1] # only used for the upsampling of 'reshape' type 

        self.reshape_type = self.decoder.reshape_type

        
        self.inputs_front_only = True
        self.render_loss_all_view = True
        self.if_include_video_ref_img = True
        
        self.training_mode = training_mode

        self.loss_weights_views = torch.Tensor(loss_weights_views).reshape(-1) / sum(loss_weights_views)  # normalize the weights
        
        

        # ========== config meaning ===========
        self.neck =  instantiate_from_config(neck)

        self.ids_restore = torch.arange(0, self.num_patches).unsqueeze(0)
        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            self.decoder.requires_grad_(False)
            if self.decoder_use_ema:
                self.decoder_ema.requires_grad_(False)
        self.image_cond = image_cond
        self.code_permute = code_permute
        self.code_reshape = code_reshape
        self.code_reshape_inv = [self.code_size[axis] for axis in self.code_permute] if code_permute is not None \
            else self.code_size
        self.code_permute_inv = [self.code_permute.index(axis) for axis in range(len(self.code_permute))] \
            if code_permute is not None else None

        self.autocast_dtype = autocast_dtype
        self.ortho = ortho
        self.return_norm = return_norm

        '''add a flag for the skip connection from sapiens shallow layer'''
        self.output_hidden_states = output_hidden_states

        ''' add the in-the-wild images visualization'''
        if self.lambda_lpips > 0:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        else:
            self.lpips = None

        self.ssim = StructuralSimilarityIndexMeasure()
        
        
        self.validation_step_outputs = []
        self.validation_step_code_outputs = [] # saving the code
        self.validation_step_nvPose_outputs = []
        self.validation_metrics = []

        
        # loading the smplx for the nv pose
        import json
        import numpy as np
        # evaluate the animation
        smplx_path = './work_dirs/demo_data/Ways_to_Catch_360_clip1.json'
        with open(smplx_path, 'r') as f:
            smplx_pose_param = json.load(f)
        smplx_param_list = []
        for par in smplx_pose_param['annotations']:
            k = par['smplx_params']
            for i in k.keys():
                k[i] = np.array(k[i])
            left_hands = np.array([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
                -0.6652, -0.7290,  0.0084, -0.4818])
            betas = torch.zeros((10))
            smplx_param = \
                np.concatenate([np.array([1]), np.array([0,0.,0]),  np.array([0, -1, 0])*k['root_orient'], \
                                k['pose_body'],betas, \
                                    k['pose_hand'], k['pose_jaw'], np.zeros(6), k['face_expr'][:10]], axis=0).reshape(1,-1)
            # print(smplx_param.shape)
            smplx_param_list.append(smplx_param)
        smplx_params = np.concatenate(smplx_param_list, 0)
        self.smplx_params = torch.Tensor(smplx_params).cuda()
    def get_default_smplx_params(self):
        A_pose = torch.Tensor([[ 1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.1047,  0.0000,  0.0000, -0.1047,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.7854,  0.0000,
            0.0000,  0.7854,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.7470,  1.0966,
            0.0169, -0.0534, -0.0212,  0.0782, -0.0348,  0.0260,  0.0060,  0.0118,
            -0.1117, -0.0429,  0.4164, -0.1088,  0.0660,  0.7562,  0.0964,  0.0909,
            0.1885,  0.1181, -0.0509,  0.5296,  0.1437, -0.0552,  0.7049,  0.0192,
            0.0923,  0.3379,  0.4570,  0.1963,  0.6255,  0.2147,  0.0660,  0.5069,
            0.3697,  0.0603,  0.0795,  0.1419,  0.0859,  0.6355,  0.3033,  0.0579,
            0.6314,  0.1761,  0.1321,  0.3734, -0.8510, -0.2769,  0.0915,  0.4998,
            -0.0266, -0.0529, -0.5356, -0.0460,  0.2774, -0.1117,  0.0429, -0.4164,
            -0.1088, -0.0660, -0.7562,  0.0964, -0.0909, -0.1885,  0.1181,  0.0509,
            -0.5296,  0.1437,  0.0552, -0.7049,  0.0192, -0.0923, -0.3379,  0.4570,
            -0.1963, -0.6255,  0.2147, -0.0660, -0.5069,  0.3697, -0.0603, -0.0795,
            0.1419, -0.0859, -0.6355,  0.3033, -0.0579, -0.6314,  0.1761, -0.1321,
            -0.3734, -0.8510,  0.2769, -0.0915,  0.4998,  0.0266,  0.0529, -0.5356,
            0.0460, -0.2774,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
        return A_pose
    def forward_decoder(self, decoder, code, target_rgbs, cameras,   
            smpl_params=None, return_decoder_loss=False, init=False):
        decoder = self.decoder_ema if self.freeze_decoder and self.decoder_use_ema else self.decoder
        num_imgs = target_rgbs.shape[1]
        outputs = decoder(
            code, smpl_params, cameras,
            num_imgs, return_loss=return_decoder_loss, init=init, return_norm=False)
        return outputs

    def on_fit_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val_code'), exist_ok=True)
    


    def forward(self, data):
        # print("iter")
        
        num_scenes = len(data['scene_id'])  # 8
        if 'cond_imgs' in data:
            cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
            cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            cond_poses = data['cond_poses'] # (num_scenes, num_imgs, 4, 4)
            smpl_params = data['cond_smpl_param']  # (num_scenes, c)
            # if 'cond_norm' in data:cond_norm = data['cond_norm'] else:  cond_norm = None
            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            cameras = torch.cat([cond_intrinsics, cond_poses.reshape(num_scenes, num_imgs, -1)], dim=-1)
            if self.if_include_video_ref_img: # new! we render to all view, including the input image; # don't compute this loss
                cameras = cameras[:,1:]
                num_imgs = num_imgs - 1


        if self.inputs_front_only: # default setting to use the first image as input
            inputs_img_idx = [0]
        else:
            raise NotImplementedError("inputs_front_only is False")
        inputs_img = cond_imgs[:,inputs_img_idx[0],...].permute([0,3,1,2]) # 
        
        target_imgs = cond_imgs[:, 1:]
        assert cameras.shape[1] == target_imgs.shape[1]


        if self.is_debug:
            try:
                code = self.forward_image_to_uv(inputs_img, is_training=self.training) #TODO check where the validation
            except Exception as e: # OOM
                main_print(e)
                code = torch.zeros([num_scenes, 32, 256, 256]).to(inputs_img.dtype).to(inputs_img.device)
        else:
            code = self.forward_image_to_uv(inputs_img, is_training=self.training) #TODO check where the validation

        decoder = self.decoder_ema if self.freeze_decoder and self.decoder_use_ema else self.decoder
        # uvmaps_decoder_gender's forward
        output = decoder(
            code, smpl_params, cameras,
            num_imgs, return_loss=False, init=(self.global_step < self.init_iter), return_norm=False) #(['scales', 'norm', 'image', 'offset'])
     
        output['code'] = code
        output['target_imgs'] = target_imgs
        output['inputs_img'] = cond_imgs[:,[0],...]
       
        # for visualization
        if self.global_rank == 0 and self.global_step % 200 == 0 and self.is_debug:
            overlay_imgs = 0.5 * target_imgs + 0.5 * output['image']
            overlay_imgs = rearrange(overlay_imgs, 'b n h w c -> b h n w c')
            overlay_imgs = rearrange(overlay_imgs, ' b h n w c -> (b h) (n w) c')
            overlay_imgs = overlay_imgs.to(torch.float32).detach().cpu().numpy()
            overlay_imgs = (overlay_imgs * 255).astype(np.uint8)
            Image.fromarray(overlay_imgs).save(f'debug_{self.global_step}.jpg')
        
        return output

    def forward_image_to_uv(self, inputs_img, is_training=True, low_ram=False):
        '''
            inputs_img: torch.Tensor, bs, 3, H, W
            return
            code : bs, 256, 256, 32
        '''
        if self.decoder_learning_rate <= 0:
            with torch.no_grad():
                features_flatten =  self.encoder(inputs_img, use_my_proces=True, output_hidden_states=self.output_hidden_states) 
        else:
            features_flatten =  self.encoder(inputs_img, use_my_proces=True, output_hidden_states=self.output_hidden_states) 
        
        if low_ram:
            del self.encoder
            torch.cuda.empty_cache()

        if self.ids_restore.device !=features_flatten.device:
            self.ids_restore = self.ids_restore.to(features_flatten.device)
        ids_restore = self.ids_restore.expand([features_flatten.shape[0], -1])
        uv_code =  self.neck(features_flatten, ids_restore)

        if low_ram:
            del self.neck
            torch.cuda.empty_cache()

        batch_size, token_num, dims_feature = uv_code.shape
        
        if self.reshape_type=='reshape':
            feature_map = uv_code.reshape(batch_size, self.num_patches_axis, self.num_patches_axis,\
                            self.code_feat_dims, self.code_patch_size, self.code_patch_size) # torch.Size([1, 64, 64, 32, 4, 4, ])  
            feature_map = feature_map.permute(0, 3, 1, 4, 2, 5)   # ([1, 32, 64, 4, 64, 4])
            feature_map = feature_map.reshape(batch_size, self.code_feat_dims,  self.code_resolution, self.code_resolution) # torch.Size([1, 32, 256, 256])
            code = feature_map # [1, 32, 256, 256]
        else:
            feature_map = uv_code.reshape(batch_size, self.num_patches_axis, self.num_patches_axis,dims_feature) # torch.Size([1, 64, 64, 512, ])  
            if isinstance(self.decoder, DistributedDataParallel):
                code = self.decoder.module.upsample_conv(feature_map.permute([0,3,1,2])) # torch.Size([1, 32, 256, 256])
            else:
                code = self.decoder.upsample_conv(feature_map.permute([0,3,1,2])) # torch.Size([1, 32, 256, 256])

        code = self.code_activation(code)
        return code

    def compute_loss(self, render_out):
        render_images = render_out['image'] # .Size([1, 5, 896, 640, 3]), range [0, 1]
        target_images = render_out['target_imgs']
        target_images  =target_images.to(render_images)
        if self.is_debug:
            render_images_tmp= rearrange(render_images, 'b n h w c -> (b n) c h w')
            target_images_tmp = rearrange(target_images, 'b n h w c -> (b n) c h w')
            all_images = torch.cat([render_images_tmp, target_images_tmp], dim=2)
            all_images = render_images_tmp*0.5 + target_images_tmp*0.5
            grid = make_grid(all_images, nrow=4, normalize=True, value_range=(0, 1))
            save_image(grid, "./debug.png")
            main_print("saving into ./debug.png")
           

        render_images = rearrange(render_images, 'b n h w c -> (b n) c h w') * 2.0 - 1.0
        target_images = rearrange(target_images, 'b n h w c -> (b n) c h w') * 2.0 - 1.0
        if self.lambda_mse<=0:
            loss_mse = 0
        else:
            if self.loss_weights_views.numel() != 0:
                b, n, _, _, _ = render_out['image'].shape
                loss_weights_views = self.loss_weights_views.unsqueeze(0).to(render_images.device)
                loss_weights_views = loss_weights_views.repeat(b,1).reshape(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                loss_mse = weighted_mse_loss(render_images, target_images, loss_weights_views)
                main_print("weighted sum mse")
            else:
                loss_mse = F.mse_loss(render_images, target_images)

        if self.lambda_l1<=0:
            loss_l1 = 0
        else:
            loss_l1 = F.l1_loss(render_images, target_images)

        if self.lambda_ssim <= 0:
            loss_ssim = 0
        else:
            loss_ssim = 1 - self.ssim(render_images, target_images)
        if not self.is_debug:
            if self.lambda_lpips<=0:
                loss_lpips = 0
            else:
                if self.loss_weights_views.numel() != 0:
                    with torch.cuda.amp.autocast():
                        loss_lpips = self.lpips(render_images.clamp(-1, 1), target_images)
                else:
                    loss_lpips = 0
                    with torch.cuda.amp.autocast():
                        for img_idx in range(render_images.shape[0]):
                            loss_lpips += self.lpips(render_images[[img_idx]].clamp(-1, 1), target_images[[img_idx]])
                    loss_lpips /= render_images.shape[0]
                    
        else:
            loss_lpips = 0
        loss_gs_offset = render_out['offset']
        loss = loss_mse * self.lambda_mse \
            + loss_l1 * self.lambda_l1 \
            + loss_ssim * self.lambda_ssim \
            + loss_lpips * self.lambda_lpips \
            + loss_gs_offset * self.lambda_offset
        
        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_mse': loss_mse})
        loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
        loss_dict.update({f'{prefix}/loss_gs_offset': loss_gs_offset})
        loss_dict.update({f'{prefix}/loss_ssim': loss_ssim})
        loss_dict.update({f'{prefix}/loss_l1': loss_l1})
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def compute_metrics(self, render_out):
        # NOTE: all the rgb value range  is [0, 1]
        # render_out.keys = (['scales', 'norm', 'image', 'offset', 'code', 'target_imgs'])
        render_images = render_out['image'].clamp(0, 1) # .Size([1, 5, 896, 640, 3]), range [0, 1]
        target_images = render_out['target_imgs']
        if target_images.dtype!=render_images.dtype:
            target_images = target_images.to(render_images.dtype)

        render_images = rearrange(render_images, 'b n h w c -> (b n) c h w')
        target_images = rearrange(target_images, 'b n h w c -> (b n) c h w').to(render_images)

        mse = F.mse_loss(render_images, target_images).mean()
        psnr = 10 * torch.log10(1.0 / mse)
        ssim = self.ssim(render_images, target_images)
        
        render_images = render_images * 2.0 - 1.0
        target_images = target_images * 2.0 - 1.0

        if self.lambda_lpips<=0:
            lpips = torch.Tensor([0]).to(render_images.device).to(render_images.dtype)
        else:
            with torch.cuda.amp.autocast():
                lpips = self.lpips(render_images, target_images)

        metrics = {
            'val/mse': mse,
            'val/pnsr': psnr,
            'val/ssim': ssim,
            'val/lpips': lpips,
        }
        return metrics

    def new_on_before_optimizer_step(self):
        norms = grad_norm(self.neck, norm_type=2)
        if 'grad_2.0_norm_total' in norms:
            self.log_dict({'grad_norm/lrm_generator': norms['grad_2.0_norm_total']})

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        render_out = self.forward(batch)

        metrics = self.compute_metrics(render_out)
        self.validation_metrics.append(metrics)
        render_images = render_out['image']
        render_images = rearrange(render_images, 'b n h w c -> b c h (n w)')
        gt_images = render_out['target_imgs']
        gt_images = rearrange(gt_images, 'b n h w c-> b c h (n w)')
        log_images = torch.cat([render_images, gt_images], dim=-2)
        self.validation_step_outputs.append(log_images)

        self.validation_step_code_outputs.append( render_out['code'])

        render_out_comb = self.forward_nvPose(batch, smplx_given=None)
        self.validation_step_nvPose_outputs.append(render_out_comb)
       
      
    def forward_nvPose(self, batch, smplx_given):
        '''
            smplx_given: torch.Tensor, bs, 189
            it will returns images with cameras_num * poses_num
        '''
        _, num_img, _,_ = batch['cond_poses'].shape
        # write a code to seperately input the smplx_params
        if smplx_given == None:
            step_pose = self.smplx_params.shape[0] // num_img
            smplx_given = self.smplx_params
        else:
            step_pose = 1
        render_out_list = []
        for i in range(num_img):  
            target_pose = smplx_given[[i*step_pose]]
            bk = batch['cond_smpl_param'].clone()
            batch['cond_smpl_param'][:, 7:70] = target_pose[:, 7:70] # copy body_pose
            batch['cond_smpl_param'][:, 80:80+93] = target_pose[:, 80:80+93]# copy pose_hand + pose_jaw
            batch['cond_smpl_param'][:, 179:189] = target_pose[:, 179:189]# copy face expression
            render_out_new = self.forward(batch)
            render_out_list.append(render_out_new['image'])
        render_out_comb = torch.cat(render_out_list, dim=2) # stack in the H axis
        render_out_comb = rearrange(render_out_comb, 'b n h w c -> b c h (n w)')
        return render_out_comb

    
    def on_validation_epoch_end(self): #
        images = torch.cat(self.validation_step_outputs, dim=-1)
        all_images = self.all_gather(images).cpu()
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        # nv pose
        images_pose = torch.cat(self.validation_step_nvPose_outputs, dim=-1)
        all_images_pose = self.all_gather(images_pose).cpu()
        all_images_pose = rearrange(all_images_pose, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            image_path = os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png')

            grid = make_grid(all_images, nrow=1, normalize=True, value_range=(0, 1))
            save_image(grid, image_path)
            main_print(f"Saved image to {image_path}")

            metrics = {}
            for key in self.validation_metrics[0].keys():
                metrics[key] = torch.stack([m[key] for m in self.validation_metrics]).mean()
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)


            # code for saving the nvPose images
            image_path_nvPose = os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}_nvPose.png')
            grid_nvPose = make_grid(all_images_pose, nrow=1, normalize=True, value_range=(0, 1))
            save_image(grid_nvPose, image_path_nvPose)
            main_print(f"Saved image to {image_path_nvPose}")


            # code for saving the code images
            for i, code in enumerate(self.validation_step_code_outputs):
                image_path = os.path.join(self.logdir, 'images_val_code')
                
                num_scenes, num_chn, h, w = code.size()
                code_viz = code.reshape(num_scenes, 4, 8, h, w).to(torch.float32).cpu().numpy()
                code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 4 * h, 8 * w)
                for j, code_viz_single in enumerate(code_viz):
                    plt.imsave(os.path.join(image_path, f'val_{self.global_step:07d}_{i*num_scenes+j:04d}' + '.png'), code_viz_single,
                        vmin=self.code_clip_range[0], vmax=self.code_clip_range[1])
        self.validation_step_outputs.clear()
        self.validation_step_nvPose_outputs.clear()
        self.validation_metrics.clear()
        self.validation_step_code_outputs.clear()
    
    def on_test_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images_test'), exist_ok=True)
    
    def on_test_epoch_end(self):
        metrics = {}
        metrics_mean = {}
        metrics_var = {}
        for key in self.validation_metrics[0].keys():
            tmp = torch.stack([m[key] for m in self.validation_metrics]).cpu().numpy()
            metrics_mean[key] = tmp.mean()
            metrics_var[key] = tmp.var()

        formatted_metrics = {}
        for key in metrics_mean.keys():
            formatted_metrics[key] = f"{metrics_mean[key]:.4f}±{metrics_var[key]:.4f}"

        for key in self.validation_metrics[0].keys():
            metrics[key] = torch.stack([m[key] for m in self.validation_metrics]).cpu().numpy().tolist()
        

        final_dict = {"average": formatted_metrics,
                      'details': metrics}

        metric_path = os.path.join(self.logdir, f'metrics.json')
        with open(metric_path, 'w') as f:
            json.dump(final_dict, f, indent=4)
        main_print(f"Saved metrics to {metric_path}")
        
        for key in metrics.keys():
            metrics[key] = torch.stack([m[key] for m in self.validation_metrics]).mean()
        main_print(metrics)
        
        self.validation_metrics.clear()
    
    def configure_optimizers(self):
        # define the optimizer and the scheduler for neck and decoder 
        main_print("WARNING currently, we only support the single optimizer for both neck and decoder")
        
        learning_rate = self.neck_learning_rate
        params= [
            {'params': self.neck.parameters(), 'lr': self.neck_learning_rate, },
            {'params': self.decoder.parameters(), 'lr': self.decoder_learning_rate},
        ]
        if hasattr(self, "encoder_learning_rate") and self.encoder_learning_rate>0:
            params.append({'params': self.encoder.parameters(), 'lr': self.encoder_learning_rate})
            main_print("============add the encoder into the optimizer============")
        optimizer = torch.optim.Adam(
            params
        )
        T_warmup, T_max, eta_min = self.warmup_steps, self.max_steps, 0.001
        lr_lambda = lambda step: \
            eta_min + (1 - math.cos(math.pi * step / T_warmup)) * (1 - eta_min) * 0.5 if step < T_warmup else \
            eta_min + (1 + math.cos(math.pi * (step - T_warmup) / (T_max - T_warmup))) * (1 - eta_min) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return  {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        scheduler.step()
        render_gt = None #? 
        render_out = self.forward(batch)
        loss, loss_dict = self.compute_loss(render_out)


        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
       
        if self.global_step % 200 == 0 and self.global_rank == 0:
            self.new_on_before_optimizer_step() # log the norm
        if self.global_step % 200 == 0 and self.global_rank == 0:
            if self.if_include_video_ref_img and self.training:
                render_images = torch.cat([ torch.ones_like(render_out['image'][:,0:1]), render_out['image']], dim=1)
                target_images = torch.cat([ render_out['inputs_img'], render_out['target_imgs']], dim=1)

            target_images = rearrange(
                target_images, 'b n h w c -> b c h (n w)')
            render_images = rearrange(
                render_images, 'b n  h w c-> b c h (n w)')
            

            grid = torch.cat([
                target_images, render_images, 0.5*render_images + 0.5*target_images,
               
            ], dim=-2)
            grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))
           
            image_path = os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.jpg')
            save_image(grid, image_path)
            main_print(f"Saved image to {image_path}")

        return loss
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # input_dict, render_gt = self.prepare_validation_batch_data(batch)
        render_out = self.forward(batch)
        render_gt = render_out['target_imgs']
        render_img = render_out['image']
        # Compute metrics
        metrics = self.compute_metrics(render_out)
        self.validation_metrics.append(metrics)
        
        # Save images
        target_images = rearrange(
            render_gt, 'b n h w c -> b c h (n w)')
        render_images = rearrange(
            render_img, 'b n h w c -> b c h (n w)')


        grid = torch.cat([
            target_images, render_images, 
        ], dim=-2)
        grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))
        # self.logger.log_image('train/render', [grid], step=self.global_step)
        image_path = os.path.join(self.logdir, 'images_test', f'{batch_idx:07d}.png')
        save_image(grid, image_path)

        # code visualize
        code = render_out['code']
        self.decoder.visualize(code, batch['scene_name'],
                        os.path.dirname(image_path), code_range=self.code_clip_range)

        print(f"Saved image to {image_path}")
    
    def on_test_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images_test'), exist_ok=True)
    
    def on_test_epoch_end(self):
        metrics = {}
        metrics_mean = {}
        metrics_var = {}
        for key in self.validation_metrics[0].keys():
            tmp = torch.stack([m[key] for m in self.validation_metrics]).cpu().numpy()
            metrics_mean[key] = tmp.mean()
            metrics_var[key] = tmp.var()

        # trans format into "mean±var" 
        formatted_metrics = {}
        for key in metrics_mean.keys():
            formatted_metrics[key] = f"{metrics_mean[key]:.4f}±{metrics_var[key]:.4f}"

        for key in self.validation_metrics[0].keys():
            metrics[key] = torch.stack([m[key] for m in self.validation_metrics]).cpu().numpy().tolist()
        

        # saving into a dictionary
        final_dict = {"average": formatted_metrics,
                      'details': metrics}

        metric_path = os.path.join(self.logdir, f'metrics.json')
        with open(metric_path, 'w') as f:
            json.dump(final_dict, f, indent=4)
        print(f"Saved metrics to {metric_path}")
        
        for key in metrics.keys():
            metrics[key] = torch.stack([m[key] for m in self.validation_metrics]).mean()
        print(metrics)
        
        self.validation_metrics.clear()
    
def weighted_mse_loss(render_images, target_images, weights):
    squared_diff = (render_images - target_images) ** 2
    main_print(squared_diff.shape, weights.shape)
    weighted_squared_diff = squared_diff * weights
    loss_mse_weighted = weighted_squared_diff.mean()
    return loss_mse_weighted