import cv2

def draw_bbox(img, x, y, color=(0, 255, 0), linewidth=2):
    cv2.rectangle(img, x, y, color, linewidth)
    return img


def draw_text(img, text, position, color=(0,255,0), scale=0.8, thickness=2):
    x, y = position
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    return img
