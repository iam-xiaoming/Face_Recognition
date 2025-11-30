import cv2
import numpy as np

_BLUR_THR = 40

def is_blurry(img):
    if img is None or img.size == 0:
        return True 
    return cv2.Laplacian(img, cv2.CV_64F).var() < _BLUR_THR


def estimate_pose_from_kps(kps):
    """
    Tính yaw (nghiêng trái/phải) + pitch (cúi/ngước nhẹ)
    kps: (5,2)
    Format insightface: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]

    eye_dx = abs(right_eye[0] - left_eye[0])
    if eye_dx < 1:
        return 0, 0

    # yaw (nghiêng trái/phải)
    mid_x = (left_eye[0] + right_eye[0]) / 2
    yaw = abs(nose[0] - mid_x) / eye_dx * 100

    # pitch (cúi/ngước)
    eye_dy = abs(right_eye[1] - left_eye[1])
    pitch = eye_dy / eye_dx * 100

    return yaw, pitch

