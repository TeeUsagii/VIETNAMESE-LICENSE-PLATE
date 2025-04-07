import cv2
import numpy as np

def convert2Square(img, size=28):
    h, w = img.shape[:2]
    max_dim = max(h, w)
    square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return cv2.resize(square_img, (size, size), interpolation=cv2.INTER_AREA)
