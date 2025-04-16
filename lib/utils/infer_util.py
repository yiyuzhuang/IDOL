import os
import imageio
import rembg
import torch
import numpy as np
import PIL.Image
from PIL import Image
from typing import Any
import json

from pathlib import Path
from torchvision.transforms import ToTensor
from rembg import remove  # For background removal
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from lib.models.deformers.smplx.lbs import batch_rodrigues
import cv2
from PIL import Image
import numpy as np

import json
# import random
import math
# import av


def reset_first_frame_rotation(root_orient, trans):
    """
    Set the root_orient rotation matrix of the first frame to the identity matrix (no rotation),
    keep the relative rotation relationships of other frames, and adjust trans accordingly.

    Parameters:
        root_orient: Tensor of shape (N, 3), representing the axis-angle parameters for N frames.
        trans: Tensor of shape (N, 3), representing the translation parameters for N frames.

    Returns:
        new_root_orient: Tensor of shape (N, 3), adjusted axis-angle parameters.
        new_trans: Tensor of shape (N, 3), adjusted translation parameters.
    """
    # Convert the root_orient of the first frame to a rotation matrix
    R_0 = axis_angle_to_matrix(root_orient[0:1])  # Shape: (1, 3, 3)

    # Compute the inverse of the first frame's rotation matrix
    R_0_inv = torch.inverse(R_0)  # Shape: (1, 3, 3)

    # Initialize lists for new root_orient and trans
    new_root_orient = []
    new_trans = []

    for i in range(root_orient.shape[0]):
        # Rotation matrix of the current frame
        R_i = axis_angle_to_matrix(root_orient[i:i+1])  # Shape: (1, 3, 3)
        R_new = torch.matmul(R_0_inv, R_i)  # Shape: (1, 3, 3)
        
        # Convert the rotation matrix back to axis-angle representation
        axis_angle_new = matrix_to_axis_angle(R_new)  # Shape: (1, 3)
        new_root_orient.append(axis_angle_new)
        
        # Adjust the translation for the current frame
        trans_i = trans[i:i+1]  # Shape: (1, 3)
        trans_new = torch.matmul(R_0_inv, trans_i.T).T  # Shape: (1, 3)
        new_trans.append(trans_new)
    
    # Stack the results of new_root_orient and new_trans
    new_root_orient = torch.cat(new_root_orient, dim=0)  # Shape: (N, 3)
    new_trans = torch.cat(new_trans, dim=0)  # Shape: (N, 3)
    
    # Adjust the new translations relative to the first frame
    new_trans = new_trans - new_trans[[0], :]  

    return new_root_orient, new_trans 

from scipy.spatial.transform import Rotation
def rotation_matrix_to_rodrigues(rotation_matrices):
    # reshape rotation_matrices to (-1, 3, 3)
    reshaped_matrices = rotation_matrices.reshape(-1, 3, 3)
    rotation = Rotation.from_matrix(reshaped_matrices)
    rodrigues_vectors = rotation.as_rotvec()
    return rodrigues_vectors



def get_hand_pose_mean():
    import numpy as np
    hand_pose_mean=  np.array([[ 0.11167871,  0.04289218, -0.41644183,  0.10881133, -0.06598568,
        -0.75622   , -0.09639297, -0.09091566, -0.18845929, -0.11809504,
         0.05094385, -0.5295845 , -0.14369841,  0.0552417 , -0.7048571 ,
        -0.01918292, -0.09233685, -0.3379135 , -0.45703298, -0.19628395,
        -0.6254575 , -0.21465237, -0.06599829, -0.50689423, -0.36972436,
        -0.06034463, -0.07949023, -0.1418697 , -0.08585263, -0.63552827,
        -0.3033416 , -0.05788098, -0.6313892 , -0.17612089, -0.13209307,
        -0.37335458,  0.8509643 ,  0.27692273, -0.09154807, -0.49983943,
         0.02655647,  0.05288088,  0.5355592 ,  0.04596104, -0.27735803,
         0.11167871, -0.04289218,  0.41644183,  0.10881133,  0.06598568,
         0.75622   , -0.09639297,  0.09091566,  0.18845929, -0.11809504,
        -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.7048571 ,
        -0.01918292,  0.09233685,  0.3379135 , -0.45703298,  0.19628395,
         0.6254575 , -0.21465237,  0.06599829,  0.50689423, -0.36972436,
         0.06034463,  0.07949023, -0.1418697 ,  0.08585263,  0.63552827,
        -0.3033416 ,  0.05788098,  0.6313892 , -0.17612089,  0.13209307,
         0.37335458,  0.8509643 , -0.27692273,  0.09154807, -0.49983943,
        -0.02655647, -0.05288088,  0.5355592 , -0.04596104,  0.27735803]])
    return hand_pose_mean


def load_smplify_json(smplx_smplify_path):
    with open(smplx_smplify_path) as f:
        data = json.load(f)
    
    # Prepare camera transformation matrix (R | t)
    RT = torch.concatenate([torch.Tensor(data['camera']['R']), torch.Tensor(data['camera']['t']).reshape(3, 1) * 2], dim=1)
    RT = torch.cat([RT, torch.Tensor([[0, 0, 0, 1]])], dim=0)

    # Create intrinsic parameters tensor
    intri = torch.Tensor(data['camera']['focal'] + data['camera']['princpt'])
    # intri[[3, 2]] = intri[[2, 3]]

    # # Set default focal length and image resolution
    # default_focal = 1120  # Default focal length
    # img_res = [640, 896]
    # default_fxy_cxy = torch.tensor([default_focal, default_focal, img_res[1] // 2, img_res[0] // 2]).reshape(1, 4)

    # # Adjust intrinsic parameters based on default focal and resolution
    # intri = intri * default_fxy_cxy[0, -2] / intri[-2]  
    # intri[-2:] = default_fxy_cxy[0, -2:]  # Force consistent image width and height

    # Extract SMPL parameters from data
    smpl_param_data = data 
    global_orient = np.array(smpl_param_data['root_pose']).reshape(1, -1)
    body_pose = np.array(smpl_param_data['body_pose']).reshape(1, -1)
    shape = np.array(smpl_param_data['betas_save']).reshape(1, -1)[:, :10]
    left_hand_pose = np.array(smpl_param_data['lhand_pose']).reshape(1, -1)
    right_hand_pose = np.array(smpl_param_data['rhand_pose']).reshape(1, -1)

    # Concatenate all parameters into a single tensor for SMPL model
    smpl_param_ref = np.concatenate([np.array([[1.]]), np.array(smpl_param_data['trans']).reshape(1, 3),
        global_orient, body_pose, shape, left_hand_pose, right_hand_pose,
        np.array(smpl_param_data['jaw_pose']).reshape(1, -1),
        np.zeros_like(np.array(smpl_param_data['leye_pose']).reshape(1, -1)),
        np.zeros_like(np.array(smpl_param_data['reye_pose']).reshape(1, -1)),
        np.zeros_like(np.array(smpl_param_data['expr']).reshape(1, -1)[:, :10])], axis=1)

    return RT, intri, torch.Tensor(smpl_param_ref).reshape(-1)  # Return transformation, intrinsic, and SMPL parameters

def load_image(input_path, output_folder, image_frame_ratio=None):
    input_img_path = Path(input_path)

    vids = []
    save_path = os.path.join(output_folder, f"{input_img_path.name}")
    print(f"Processing: {save_path}")
    image = Image.open(input_img_path)

    if image.mode == "RGBA":
        pass
    else:
        # remove bg
        image = remove(image.convert("RGBA"), alpha_matting=True)

    # resize object in frame
    image_arr = np.array(image)
    in_w, in_h = image_arr.shape[:2]
    ret, mask = cv2.threshold(
        np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = (
        int(max_size / image_frame_ratio)
        if image_frame_ratio is not None
        else int(max_size / 0.85)
    )
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h,
        center - w // 2 : center - w // 2 + w,
    ] = image_arr[y : y + h, x : x + w]
    rgba = Image.fromarray(padded_image).resize((896, 896), Image.LANCZOS)
    # crop the width into 640 in the center
    rgba = rgba.crop([128, 0, 640+128, 896])
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = Image.fromarray((rgb * 255).astype(np.uint8))

    image = ToTensor()(input_image)

    return image



def prepare_camera( resolution_x = 640, resolution_y = 640, focal_length = 600,sensor_width = 32,  camera_dist = 20, num_views=1, stides=1):
        
    def look_at(camera_position, target_position, up_vector):  # colmap +z forward, +y down
        forward = -(camera_position - target_position) / np.linalg.norm(camera_position - target_position)
        right = np.cross(up_vector, forward)
        up = np.cross(forward, right)
        return np.column_stack((right, up, forward))

    # set the intrisics
    focal_length = focal_length * (resolution_y/sensor_width)

    K = np.array(
        [[focal_length, 0, resolution_x//2],
        [0, focal_length, resolution_y//2],
        [0, 0, 1]]
    )

    # set the extrisics
    camera_pose_list = []
    for frame_idx in range(0, num_views, stides):

        phi = math.radians(90)
        theta = (3 / 4) * math.pi * 2
        camera_location = np.array(
            [camera_dist * math.sin(phi) * math.cos(theta),
            
            camera_dist * math.cos(phi),
            -camera_dist * math.sin(phi) * math.sin(theta),]
            )
        # print(camera_location)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_location

        # Set camera position and target position
        camera_position = camera_location
        target_position = np.array([0.0, 0.0, 0.0])

        # Compute the camera's rotation matrix to look at the target
        up_vector = np.array([0.0, -1.0, 0.0]) # colmap
        rotation_matrix = look_at(camera_position, target_position, up_vector)

        # Update camera position and rotation
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = camera_position
        camera_pose_list.append(camera_pose)
    return K, camera_pose_list


def construct_camera(K, cam_list, device='cuda'):
    num_imgs = len(cam_list)
    front_idx = num_imgs//4*3
    cam_list = cam_list[front_idx:] + cam_list[:front_idx] 
    cam_raw = np.array(cam_list)
    cam_raw[:, :3, 3] = cam_raw[:, :3, 3] 
    cam = np.linalg.inv(cam_raw)
    cam = torch.Tensor(cam)
    intrics = torch.Tensor([K[0,0],K[1,1], K[0,2], K[1,2]]).reshape(-1)
    scale = 0.5
    # diffrent from the synthetic data, the scale is process first
    # trans from (3,) to (batch_size, 3,1)
    trans = [0, 0.2, 0] #in the center
    trans_bt = torch.Tensor(trans).reshape(1, 3, 1).expand(cam.shape[0], 3, 1)
    cam[:,:3,3] = cam[:,:3,3] + torch.bmm(cam[:,:3,:3], trans_bt).reshape(-1, 3) # T = Rt+T torch.Size([24, 3, 1])
    cam[:,:3,:3] = cam[:,:3,:3] * scale  # R = sR
    cam_c2w = torch.inverse(cam)
    cam_w2c = cam
    poses = []
    for i_cam in range(cam.shape[0]):
        poses.append( torch.concat([
            (intrics.reshape(-1)).to(torch.float32), #C ! # C ? T 理论上要给C
            (cam_w2c[i_cam]).to(torch.float32).reshape(-1), # RT  #Rt|C ? RT 理论上要给RT
        ], dim=0))
    cameras = torch.stack(poses).to(device) # [N, 19]
    return cameras

def get_name_str(name):
    path_ = os.path.basename(os.path.dirname(name)) + os.path.basename(name)
    return path_



def load_smplx_from_npy(smplx_path, device='cuda'):
    hand_mean = get_hand_pose_mean().reshape(-1)
    smplx_pose_param = np.load(smplx_path, allow_pickle=True)
    # if "person1" in smplx_pose_param:
    #     smplx_pose_param = smplx_pose_param['person1']
    smplx_pose_param = {
        'root_orient': smplx_pose_param[:, :3],  # controls the global root orientation
        'pose_body': smplx_pose_param[:, 3:3+63],  # controls the body
        'pose_hand': smplx_pose_param[:, 66:66+90],  # controls the finger articulation
        'pose_jaw': smplx_pose_param[:, 66+90:66+93],  # controls the yaw pose
        'face_expr': smplx_pose_param[:, 159:159+50],  # controls the face expression
        'face_shape': smplx_pose_param[:, 209:209+100],  # controls the face shape
        'trans': smplx_pose_param[:, 309:309+3],  # controls the global body position
        'betas': smplx_pose_param[:, 312:],  # controls the body shape. Body shape is static
    }

    smplx_param_list = []
    for i in range(1, 1800, 1):
        # for i in k.keys():
        #     k[i] = np.array(k[i])
        left_hands = np.array([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
            -0.6652, -0.7290,  0.0084, -0.4818])
        betas = torch.zeros((10))
        smplx_param = \
            np.concatenate([np.array([1]), smplx_pose_param['trans'][i], smplx_pose_param['root_orient'][i], \
                            smplx_pose_param['pose_body'][i],betas, \
                                smplx_pose_param['pose_hand'][i]-hand_mean, smplx_pose_param['pose_jaw'][i], np.zeros(6), smplx_pose_param['face_expr'][i][:10]], axis=0).reshape(1,-1)
        smplx_param_list.append(smplx_param)
    smplx_params = np.concatenate(smplx_param_list, 0)
    smpl_params = torch.Tensor(smplx_params).to(device)
    return smpl_params
def add_root_rotate_to_smplx(smpl_tmp, frames_num=180, device='cuda'):
    from cv2 import Rodrigues
    initial_matrix = batch_rodrigues(smpl_tmp.reshape(1,189)[:, 4:7]).cpu().numpy().copy()
    # Rotate a rotation matrix by 360 degrees around the y-axis.
    # frames_num = 180
    all_smpl = []
    # Combine the rotations
    all_smpl = []
    for idx_f in range(frames_num):
        new_smpl = smpl_tmp.clone() 
        angle = 360//frames_num * idx_f
        y_angle = np.radians(angle)
        y_rotation_matrix = np.array([
            [ np.cos(y_angle),0,  np.sin(y_angle)],
            [0,  1, 0],
            [-np.sin(y_angle), 0, np.cos(y_angle)],
        ])
        final_matrix = y_rotation_matrix[None] @ initial_matrix
        
        new_smpl[4:7] = torch.Tensor(rotation_matrix_to_rodrigues(torch.Tensor(final_matrix))).to(device)
        all_smpl.append(new_smpl)
    all_smpl = torch.stack(all_smpl, 0)
    smpl_params = all_smpl.to(device)
    return smpl_params

def load_smplx_from_json(smplx_path, device='cuda'):
    # format of motion-x
    hand_mean = get_hand_pose_mean().reshape(-1)
    with open(smplx_path, 'r') as f:
        smplx_pose_param = json.load(f)
    smplx_param_list = []
    for par in smplx_pose_param['annotations']:
        k = par['smplx_params']
        for i in k.keys():
            k[i] = np.array(k[i])

        betas = torch.zeros((10))
        # #########   wrist pose fix ################
        smplx_param = \
            np.concatenate([np.array([1]), k['trans'], 
                            k['root_orient']*np.array([1, 1, 1]), \
                            k['pose_body'],betas, \
                            k['pose_hand']-hand_mean, k['pose_jaw'], np.zeros(6), np.zeros_like(k['face_expr'][:10])], axis=0).reshape(1,-1)
        smplx_param_list.append(smplx_param)


    smplx_params = np.concatenate(smplx_param_list, 0)
    print(smplx_params.shape)
    smpl_params = torch.Tensor(smplx_params).to(device)
    return smpl_params

def get_image_dimensions(input_path):
    with Image.open(input_path) as img:
        return img.height, img.width  

def construct_camera_from_motionx(smplx_path, device='cuda'):
    with open(smplx_path, 'r') as f:
        smplx_pose_param = json.load(f)
    cam_exts = []
    cam_ints = []
    for par in smplx_pose_param['annotations']:
        cam = par['cam_params']
        R = np.array(cam['cam_R'])
        K = np.array(cam['intrins'])
        T = np.array(cam['cam_T']) 
        cam['cam_T'][1] = -cam['cam_T'][1]
        cam['cam_T'][2] = -cam['cam_T'][2]
        extrix = np.eye(4)
        extrix[:3, :3] = R
        extrix[:3,3] = T
        cam_exts.append(extrix)
        intrix = K
        cam_ints.append(intrix)

    # target N,20
    cam_exts_array = np.array(cam_exts)

    cam_exts_stack = torch.Tensor(cam_exts_array).to(device).reshape(-1, 16)
    cam_ints_stack = torch.Tensor(cam_ints).to(device).reshape(-1, 4)
    cameras = torch.cat([cam_ints_stack, cam_exts_stack], dim=-1).reshape(-1,1, 20)
    return cameras

def remove_background(image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = PIL.Image.fromarray(new_image)
    return new_image


def images_to_video(
    images: torch.Tensor, 
    output_path: str, 
    fps: int = 30,
) -> None:
    # images: (N, C, H, W)
    video_dir = os.path.dirname(output_path)
    video_name = os.path.basename(output_path)
    os.makedirs(video_dir, exist_ok=True)

    frames = []
    for i in range(len(images)):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, quality=10)


def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 30,
) -> None:
    """
    使用OpenCV保存视频（自动处理BGR到RGB转换）
    
    参数:
        frames: 输入帧序列，形状为 (N, C, H, W) 的torch.Tensor
        output_path: 输出视频路径
        fps: 帧率（默认30）
    """
    # 检查输入张量维度
    if frames.dim() != 4:
        raise ValueError(f"输入frames应该是4D张量 (N, C, H, W)，但得到的是 {frames.dim()}D")
    
    # 转换为numpy数组并调整格式
    frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)
    frames_np = (frames_np * 255).astype(np.uint8)  # 转换为0-255范围
    
    # 获取视频尺寸
    height, width = frames_np.shape[1:3]
    
    # 创建VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames_np:
        # OpenCV使用BGR格式，如果输入是RGB需要转换
        if frame.shape[2] == 3:  # 仅当是彩色图像时转换
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame  # 灰度图像不转换
        
        writer.write(frame_bgr)
    
    writer.release()