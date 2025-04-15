import cv2
import numpy as np
from skimage import measure
from skimage.filters import threshold_local
import imutils
from tensorflow.keras.models import load_model

# Từ điển ánh xạ ký tự
ALPHA_DICT = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'Background', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H',
    19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'R', 25: 'S', 26: 'T', 27: 'U',
    28: 'V', 29: 'X', 30: 'Y', 31: 'Z'
}

def convert2Square(img, size=28):
    h, w = img.shape[:2]
    max_dim = max(h, w)

    # Tạo ảnh vuông có nền đen
    square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)

    # Canh giữa ký tự trong ảnh vuông
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img

    # Resize về 28x28
    final_img = cv2.resize(square_img, (size, size), interpolation=cv2.INTER_AREA)
    return final_img

def segment_characters(aligned_plate):
    # Chuyển sang không gian màu HSV và lấy kênh V (độ sáng)
    V = cv2.split(cv2.cvtColor(aligned_plate, cv2.COLOR_BGR2HSV))[2]

    # Áp dụng adaptive threshold
    T = threshold_local(V, 15, offset=10, method="gaussian")
    thresh = (V > T).astype("uint8") * 255

    # Đảo màu: chữ trắng, nền đen
    thresh = cv2.bitwise_not(thresh)

    # Resize ảnh để thống nhất kích thước xử lý
    thresh = imutils.resize(thresh, width=400)

    # Làm mờ ảnh để giảm nhiễu
    thresh = cv2.medianBlur(thresh, 5)

    # Phân tích thành phần liên kết
    labels = measure.label(thresh, connectivity=2, background=0)
    candidates = []

    for label in np.unique(labels):
        if label == 0:
            continue

        # Tạo mask chỉ chứa thành phần hiện tại
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

        # Tìm contour của thành phần
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(contour)

            # Lọc bỏ các vùng không phải ký tự
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            heightRatio = h / float(aligned_plate.shape[0])

            if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                char_img = mask[y:y + h, x:x + w]
                char_img = convert2Square(char_img, size=28)
                char_img = char_img.reshape((28, 28, 1))
                candidates.append((char_img, (y, x)))

    return sorted(candidates, key=lambda c: c[1][1])

class LicensePlateRecognizer:
    def __init__(self):
        self.recogChar = load_model("./models/license_plate_char.h5")  # Load mô hình trong class
        # self.recogChar = load_model("D:/Tien/Project/Python/VIETNAMESE-LICENSE-PLATE/Traning-cnn/cnn1/model/license_plate_char_best.h5")  # Load mô hình trong class
        self.candidates = []

    def recognize_characters(self, candidates):
        if not candidates:
            return []

        characters = np.array([char for char, _ in candidates]) / 255.0  # Chuẩn hóa pixel
        coordinates = [coord for _, coord in candidates]

        # Nhận diện ký tự
        results = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(results, axis=1)

        # Lưu lại các ký tự nhận diện được
        recognized_chars = [(ALPHA_DICT[idx], coord) for idx, coord in zip(result_idx, coordinates) if idx != 12]
        self.candidates = recognized_chars

    def format(self):
        if not self.candidates:
            return ""

        # Lấy toàn bộ giá trị y
        y_values = [coord[0] for _, coord in self.candidates]
        y_mean = np.mean(y_values)
        y_std = np.std(y_values)

        # Ngưỡng lệch chuẩn để quyết định 1 hay 2 dòng
        threshold_std = 12

        if y_std < threshold_std:
            # Một dòng
            sorted_chars = sorted(self.candidates, key=lambda x: x[1][1])  # theo x
            return "".join([char for char, _ in sorted_chars])
        else:
            # Hai dòng
            first_line, second_line = [], []
            for char, coord in self.candidates:
                if coord[0] < y_mean:
                    first_line.append((char, coord[1]))  # y < mean → dòng trên
                else:
                    second_line.append((char, coord[1]))  # y >= mean → dòng dưới

            first_line.sort(key=lambda s: s[1])
            second_line.sort(key=lambda s: s[1])
            return "".join([char for char, _ in first_line]) + "".join([char for char, _ in second_line])