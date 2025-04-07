import cv2
from ultralytics import YOLO
import os
import numpy as np
from .recognizer import segment_characters, LicensePlateRecognizer

# Load mô hình YOLO
model = YOLO("./models/yolo-11-biensoxe.pt")
recognizer = LicensePlateRecognizer()


def resize_image_if_larger(image, target_width=500):
    if image is None:
        print("Không thể đọc ảnh")
        return None

    height, width = image.shape[:2]
    if width > target_width:
        aspect_ratio = height / width
        new_height = int(target_width * aspect_ratio)
        resized_img = cv2.resize(image, (target_width, new_height))
        return resized_img
    return image


def align_license_plate(plates):
    gray = cv2.cvtColor(plates, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        angles = [(line[0][1] * 180 / np.pi) - 90 for line in lines]
        valid_angles = [angle for angle in angles if -15 < angle < 15]
        median_angle = np.median(valid_angles) if valid_angles else 0

        h, w = plates.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        aligned_plate = cv2.warpAffine(plates, rotation_matrix, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
    else:
        aligned_plate = plates.copy()

    return aligned_plate


def process_image(image_path):
    img = cv2.imread(image_path)
    resized_image = resize_image_if_larger(img)
    results = model(resized_image)

    result_data = {}
    for result in results:
        image_and_boxes = result.plot()
        result_path = os.path.join('static/results', f"result_{os.path.basename(image_path)}")
        cv2.imwrite(result_path, image_and_boxes)
        result_data['boxed_image'] = os.path.basename(result_path)

        plates_data = []
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            plate = resized_image[y1:y2, x1:x2]
            aligned_plate = align_license_plate(plate)

            aligned_path = os.path.join('static/results', f"aligned_plate_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(aligned_path, aligned_plate)

            candidates = segment_characters(aligned_plate)
            recognizer.recognize_characters(candidates)
            plate_text = recognizer.format()

            plates_data.append({
                'aligned_image': os.path.basename(aligned_path),
                'text': plate_text
            })

        result_data['plates'] = plates_data

    return result_data