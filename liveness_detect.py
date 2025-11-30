import numpy as np

def is_movement_kps(kps, prev_kps):
    try:
        delta = np.linalg.norm(kps - prev_kps)
    except Exception as e:
        return True
    if delta < 2:
        return False
    return True
