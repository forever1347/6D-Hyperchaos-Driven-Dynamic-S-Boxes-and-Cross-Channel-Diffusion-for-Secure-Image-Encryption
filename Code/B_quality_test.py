import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D

#-----------------------------------------------------噪声攻击--------------------------------------------
def gauss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

#默认10%的椒盐噪声
def salt_and_pepper_noise(noise_img, proportion=0.1):
    height, width, _ = noise_img.shape
    num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img

#-----------------------------------------------直方图分析----------------------------------------

def histogram_test(image):
    # 读取彩色图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以正确显示

    # 分离图像的 R、G、B 通道
    b_channel, g_channel, r_channel = cv2.split(image_rgb)

    # 计算每个通道的直方图
    hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])
    return hist_r,hist_g,hist_b

def compare_histograms(encrypted1, encrypted2):
    print("---histogram start---")
    hist_r1,hist_g1,hist_b1= histogram_test(encrypted1)
    hist_r2,hist_g2,hist_b2 = histogram_test(encrypted2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    x = np.arange(256)  # x轴表示像素强度
    plt.fill_between(x, hist_r1.flatten(), color='r', alpha=0.5, label='Red Channel')

    plt.ylim(0, np.max(hist_r1) * 1.1)  # 动态设置纵坐标范围

    plt.fill_between(x, hist_g1.flatten(), color='g', alpha=0.5, label='Green Channel')

    plt.ylim(0, np.max(hist_g1) * 1.1)

    plt.fill_between(x, hist_b1.flatten(), color='b', alpha=0.5, label='Blue Channel')

    plt.ylim(0, np.max(hist_b1) * 1.1)

    # 显示结果
    plt.tight_layout()


    plt.subplot(1, 2, 2)
    x = np.arange(256)  # x轴表示像素强度

    plt.fill_between(x, hist_r2.flatten(), color='r', alpha=0.5, label='Red Channel')

    plt.ylim(0, np.max(hist_r2-hist_r1) * 1.1)  # 动态设置纵坐标范围



    plt.fill_between(x, hist_g2.flatten(), color='g', alpha=0.5, label='Green Channel')

    plt.ylim(0, np.max(hist_g2) * 1.1)
    plt.fill_between(x, hist_b2.flatten(), color='b', alpha=0.5, label='Blue Channel')


    plt.ylim(0, np.max(hist_b2) * 1.1)

    # 显示结果
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6, 5))
    x = np.arange(256)  # x轴表示像素强度
    plt.fill_between(x, np.abs(hist_r1-hist_r2).flatten(), color='r', alpha=0.5, label='Red Channel')

    plt.ylim(0, np.max(np.abs(hist_r1-hist_r2)) * 1.1)  # 动态设置纵坐标范围



    plt.fill_between(x, np.abs(hist_g1-hist_g2).flatten(), color='g', alpha=0.5, label='Green Channel')

    plt.ylim(0, np.max(np.abs(hist_g1-hist_g2)) * 1.1)



    plt.fill_between(x, np.abs(hist_b1-hist_b2).flatten(), color='b', alpha=0.5, label='Blue Channel')

    plt.ylim(0, np.max(np.abs(hist_b1-hist_b2)) * 1.1)
    # 显示结果
    plt.tight_layout()
    plt.show()
    print("---histogram end---")


# 示例使用
encrypted1 = cv2.imread("Airplane.png_encrypted_key1.png")  # 灰度图
encrypted2 = cv2.imread("Airplane.png_encrypted_key3.png")  # 灰度图
compare_histograms(encrypted1, encrypted2)

#-------------------------------------信息熵和相关性分析---------------------------------------
def calc_entropy(channel):
    hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256], density=True)
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
    return entropy

def correlation_coefficient(x, y):
    return np.corrcoef(x, y)[0, 1]

def init_correlation(image1, image2):
    # 转为 RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # 分离通道
    R1, G1, B1 = cv2.split(image1)
    R2, G2, B2 = cv2.split(image2)

    # 图像准备
    original_channels = [R1, G1, B1]
    encrypted_channels = [R2, G2, B2]

    # 打印熵
    print('\n--- Entropy ---')
    for name, ch in zip(['R1', 'G1', 'B1', 'R2', 'G2', 'B2'], [R1, G1, B1, R2, G2, B2]):
        print(f'{name}: {calc_entropy(ch):.4f}')

    # 打印相关性
    print('\n--- Correlation Coefficients ---')
    for name, ch in zip(['R1', 'G1', 'B1', 'R2', 'G2', 'B2'], [R1, G1, B1, R2, G2, B2]):
        h1, h2, v1, v2, d1, d2 = collect_pairs_individual(ch)
        print(f'{name}: H={correlation_coefficient(h1, h2):.4f}, V={correlation_coefficient(v1, v2):.4f}, D={correlation_coefficient(d1, d2):.4f}')
    return original_channels,encrypted_channels
def collect_pairs_individual(channel):
    M, N = channel.shape
    x = np.random.randint(0, M - 1, 5000)
    y = np.random.randint(0, N - 1, 5000)

    # H
    h1 = channel[x, y]
    h2 = channel[x, y + 1]
    # V
    v1 = channel[x, y]
    v2 = channel[x + 1, y]
    # D
    d1 = channel[x, y]
    d2 = channel[x + 1, y + 1]
    return h1, h2, v1, v2, d1, d2

img1 = cv2.imread('peppers.png')
img2 = cv2.imread('peppers.png_encrypted.png')

