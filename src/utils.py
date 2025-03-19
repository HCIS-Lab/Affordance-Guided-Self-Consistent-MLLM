import base64
import mimetypes
import math
import torch

from isaacgym import gymapi
from isaacgym.torch_utils import quat_mul, quat_conjugate

def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

def decode_image(encoded_string: str, save_path: str):
    """Decodes an image from a base64 string."""
    encoded_string = encoded_string.split(',')[1]
    with open(save_path, "wb") as image_file:
        image_file.write(base64.b64decode(encoded_string.encode('utf-8')))

def sort_scores_dict(scores_dict):
    return dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=1))


def euler_to_quaternion(roll, pitch, yaw, use_tensor=False):
    # Abbreviations for the various angular functions
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    q = gymapi.Quat(0,0,0,0)
    q.w = cy * cr * cp + sy * sr * sp
    q.x = cy * sr * cp - sy * cr * sp
    q.y = cy * cr * sp + sy * sr * cp
    q.z = sy * cr * cp - cy * sr * sp

    return q if not use_tensor else torch.tensor([[q.x, q.y, q.z, q.w]])

# Input [x,y,z,w] list with quat
def quaternion_to_euler(q, log=False):
    # roll (x-axis rotation)
    sinr_cosp = +2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = +2.0 * (q.w * q.y - q.z * q.x)
    if (math.fabs(sinp) >= 1):
        pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = +2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)  
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll,pitch,yaw]

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)