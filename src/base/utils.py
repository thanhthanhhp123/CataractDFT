import numpy as np
import cv2
import matplotlib.pyplot as plt

class FrequencyPatchMasking:
    def __init__(self, mask_size=(10, 10)):
        """
        Frequency Patch Masking (FPM) Module.
        Args:
            mask_size: Tuple (h, w), kích thước của mặt nạ được áp dụng trên phổ tần số.
        """
        self.mask_size = mask_size
    
    def fft_2d(self, image):
        """Biến đổi FFT-2D và chuyển dịch phổ tần số."""
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        return dft_shift

    def ifft_2d(self, dft_shift):
        """Biến đổi ngược IFFT-2D."""
        dft_ishift = np.fft.ifftshift(dft_shift)
        img_reconstructed = cv2.idft(dft_ishift)
        img_reconstructed = cv2.magnitude(img_reconstructed[:, :, 0], img_reconstructed[:, :, 1])
        return img_reconstructed

    def apply_mask(self, dft_shift):
        """Áp dụng mặt nạ trên phổ tần số."""
        rows, cols = dft_shift.shape[:2]
        center_row, center_col = rows // 2, cols // 2

        mask = np.ones((rows, cols, 2), np.uint8) 
        h, w = self.mask_size
        mask[center_row - h // 2:center_row + h // 2, center_col - w // 2:center_col + w // 2] = 0

        masked_dft = dft_shift * mask
        return masked_dft

    def process(self, image):
        """
        Thực hiện FPM.
        Args:
            image: Ảnh đầu vào, dạng grayscale.
        Returns:
            Augmented image: Ảnh tăng cường sau khi áp dụng FPM.
        """
        dft_shift = self.fft_2d(image)

        masked_dft = self.apply_mask(dft_shift)

        augmented_image = self.ifft_2d(masked_dft)
        
        return augmented_image

if __name__ == "_main_":
    # Đọc ảnh đầu vào (grayscale)
    img = cv2.imread('/Users/tranthanh/Documents/Projects/CataractDFT/cat_0_1000_jpg.rf.7459ebf91d974d53a62a330633556239.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Tạo module FPM
    fpm = FrequencyPatchMasking(mask_size=(5, 5))
    
    # Áp dụng FPM
    augmented_img = fpm.process(img)
    
    # Hiển thị kết quả
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Augmented Image (FPM)")
    plt.imshow(augmented_img, cmap='gray')

    plt.show()