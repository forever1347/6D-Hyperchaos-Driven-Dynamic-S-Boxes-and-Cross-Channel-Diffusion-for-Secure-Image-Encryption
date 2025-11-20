import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def arnold_cat_map(image):
    num_iter = 20
    N = image.shape[0]  # 假设图像是方形
    transformed_image = np.copy(image)

    # 预计算所有坐标
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

    for _ in range(num_iter):
        # 计算新坐标
        new_x = (x + y) % N
        new_y = (x + 2 * y) % N

        # 使用高级索引进行像素重排
        if len(transformed_image.shape) == 2:  # 灰度图像
            transformed_image = transformed_image[new_x, new_y]
        else:  # 彩色图像
            transformed_image = transformed_image[new_x, new_y, :]

    return transformed_image

def inverse_arnold_cat_map(image):
    num_iter = 20
    N = image.shape[0]  # 假设图像是方形
    transformed_image = np.copy(image)

    # 预计算所有坐标
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

    for _ in range(num_iter):
        # 计算逆变换坐标
        new_x = (2 * x - y) % N
        new_y = (-x + y) % N

        # 使用高级索引进行像素重排
        if len(transformed_image.shape) == 2:  # 灰度图像
            transformed_image = transformed_image[new_x, new_y]
        else:  # 彩色图像
            transformed_image = transformed_image[new_x, new_y, :]

    return transformed_image

def main_Arnold(image, state):
    # 加载图像并转换为numpy数组
    img_array = np.array(image)

    # 检查图像是否为方形，若不是，则进行填充
    height, width = img_array.shape[:2]
    if height != width:
        size = max(height, width)
        # 创建新数组并填充
        if len(img_array.shape) == 2:  # 灰度图像
            new_array = np.zeros((size, size), dtype=img_array.dtype)
        else:  # 彩色图像
            new_array = np.zeros((size, size, 3), dtype=img_array.dtype)
        new_array[:height, :width] = img_array
        img_array = new_array

    # 进行 Arnold Cat Map 变换
    if state == 'encryption':
        transformed_image = arnold_cat_map(img_array)
    elif state == 'decryption':
        transformed_image = inverse_arnold_cat_map(img_array)
    else:
        raise ValueError("state参数必须是'encryption'或'decryption'")

    # 如果原始图像不是方形，裁剪回原始尺寸
    if height != width:
        transformed_image = transformed_image[:height, :width]

    return transformed_image

'''
image_path = 'peppers.png'  # 替换为你的图像路径
image = Image.open(image_path)  # 转换为灰度
img_array = np.array(image)

# 进行 Arnold Cat Map 变换
transformed_image = main_Arnold(img_array,'encryption')

# 进行 Arnold Cat Map 的逆变换
recovered_image = main_Arnold(transformed_image,'decryption')
'''
'''
plt.subplot(1, 2, 1)
plt.title("Transformed Image")
plt.imshow(transformed_image, cmap='gray')

plt.show()
plt.subplot(1, 2, 2)
plt.title("Transformed Image")
plt.imshow(recovered_image, cmap='gray')

plt.show()
'''
