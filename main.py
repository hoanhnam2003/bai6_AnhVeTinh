import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào dưới dạng ma trận (chuyển sang grayscale nếu là ảnh màu)
def load_image(path):
    img = Image.open(path).convert("L")  # Chuyển ảnh sang grayscale
    return np.array(img)

# Bộ lọc Gaussian nhẹ để làm mịn ảnh trước khi phát hiện cạnh
def apply_gaussian(image, sigma=1):
    return ndimage.gaussian_filter(image, sigma=sigma)

# Toán tử Sobel (Tăng cường độ sắc nét)
def sobel_operator(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobel_x, sobel_y)
    sobel = (sobel / sobel.max() * 255).astype(np.uint8)
    return cv2.convertScaleAbs(sobel, alpha=2, beta=0)  # Tăng cường độ nét

# Toán tử Prewitt (Tăng cường rõ nét tối đa)
def prewitt_operator(image):
    Kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Ix = ndimage.convolve(image, Kx)
    Iy = ndimage.convolve(image, Ky)
    G = np.hypot(Ix, Iy)  # Tính cường độ biên độ
    G = (G / G.max() * 255).astype(np.uint8)  # Chuẩn hóa kết quả
    _, thresholded = cv2.threshold(G, 150, 255, cv2.THRESH_BINARY)  # Tăng ngưỡng lên 150
    enhanced = cv2.convertScaleAbs(thresholded, alpha=5, beta=0)  # Tăng cường rõ nét mạnh mẽ
    return enhanced

# Toán tử Roberts (Tăng cường rõ nét tối đa)
def roberts_operator(image):
    Kx = np.array([[1, 0], [0, -1]])
    Ky = np.array([[0, 1], [-1, 0]])
    Ix = ndimage.convolve(image, Kx)
    Iy = ndimage.convolve(image, Ky)
    G = np.hypot(Ix, Iy)  # Tính cường độ biên độ
    G = (G / G.max() * 255).astype(np.uint8)  # Chuẩn hóa kết quả
    _, thresholded = cv2.threshold(G, 150, 255, cv2.THRESH_BINARY)  # Tăng ngưỡng lên 150
    enhanced = cv2.convertScaleAbs(thresholded, alpha=5, beta=0)  # Tăng cường rõ nét mạnh mẽ
    return enhanced

# Toán tử Canny
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

# Bộ lọc Laplacian
def laplacian_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.uint8(np.absolute(laplacian))

# Đọc và xử lý ảnh
image_path =  r"C:\Users\Admin\Downloads\z6012553531934_fa9e5ee21ce66baa92d94d47b7ae1330.jpg"  # Đường dẫn tới ảnh
image = load_image(image_path)

# Tiền xử lý bằng Gaussian
gaussian_blurred = apply_gaussian(image, sigma=1)

# Áp dụng các toán tử cạnh
sobel_image = sobel_operator(gaussian_blurred)
prewitt_image = prewitt_operator(gaussian_blurred)
roberts_image = roberts_operator(gaussian_blurred)
canny_image = canny_edge_detection(gaussian_blurred, low_threshold=30, high_threshold=100)
laplacian_image = laplacian_filter(gaussian_blurred)

# Hiển thị kết quả Sobel
plt.figure(figsize=(18, 14))  # Điều chỉnh kích thước ảnh để có không gian cho các ảnh
plt.subplot(3, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(image, cmap='gray')

plt.subplot(3, 2, 2)
plt.title("Ảnh sau khi áp dụng Sobel")
plt.imshow(sobel_image, cmap='gray')

# Hiển thị kết quả Prewitt
plt.subplot(3, 2, 3)
plt.title("Ảnh sau khi áp dụng Prewitt")
plt.imshow(prewitt_image, cmap='gray')

# Hiển thị kết quả Roberts
plt.subplot(3, 2, 4)
plt.title("Ảnh sau khi áp dụng Roberts")
plt.imshow(roberts_image, cmap='gray')

# Hiển thị kết quả Canny
plt.subplot(3, 2, 5)
plt.title("Ảnh sau khi áp dụng Canny")
plt.imshow(canny_image, cmap='gray')

# Hiển thị kết quả Gaussian + Threshold
plt.subplot(3, 2, 6)
plt.title("Ảnh sau khi phân đoạn ngưỡng")
plt.imshow(laplacian_image, cmap='gray')

# Điều chỉnh khoảng cách giữa các ảnh
plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Giảm khoảng cách ngang và dọc nếu cần

# Hiển thị ảnh
plt.show()



