import numpy as np
import cv2
from PIL import Image  # 新增导入
import A_6D_hyperchaotic_system
import A_Sbox
import A_diffusion
import A_blcok_DNA_AES
import A_DNA
import A_Arnold
import A_cross
import B_quality_test
import A_DNA_AES
from hashlib import sha256
import Sbox_test
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def save_with_dpi(filename, image_array):
    """保存图像并设置DPI为300"""
    # OpenCV使用BGR通道，PIL需要RGB
    if len(image_array.shape) == 3:  # 彩色图像
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:  # 灰度图像
        image_rgb = image_array
    pil_image = Image.fromarray(image_rgb)
    pil_image.save(filename, dpi=(300, 300))  # 设置DPI为300

# 混沌序列以及块大小
def pre_en_and_de(seq_len):
    chaos_sequence,initial_state,iv = A_6D_hyperchaotic_system.init_system_parms(seq_len)
    seq1 = (chaos_sequence.y[0] * 1e5 % 256).astype(np.uint8)
    seq2 = (chaos_sequence.y[1] * 1e5 % 256).astype(np.uint8)
    seq3 = (chaos_sequence.y[2] * 1e5 % 256).astype(np.uint8)
    seq4 = (chaos_sequence.y[3] * 1e5 % 256).astype(np.uint8)
    seq5 = (chaos_sequence.y[4] * 1e5 % 256).astype(np.uint8)
    seq6 = (chaos_sequence.y[5] * 1e5 % 256).astype(np.uint8)
    seqs = [seq1, seq2, seq3, seq4, seq5, seq6]
    return seqs,initial_state,iv

def encryption(image, file):
    g, b, r = cv2.split(image)
    w, h, _ = image.shape
    seq_len = w * h
    state = 'encryption'
    block = 512
    seqs,initial_state,iv = pre_en_and_de(seq_len)


    # 假设 iv_key 是整数数组（0-255）
    iv_key = seqs[5][:256]  # 取前 256 个整数
    assert len(iv_key) == 256

    # 转换为十六进制字符串（小写）
    hex_str = ''.join([f"{x:02x}" for x in iv_key])
    hex_str = hex_str[:64]
    print("hex_str--------------",hex_str)
    for i in range(1):

        g, b, r = A_cross.main_Cross(g, b, r, state, seqs,iv[0])
        #g, b, r = [A_blcok_DNA_AES.confusion_main(channel, state, block, processor) for channel in (g, b, r)]
        '''
        encryptor = A_DNA_AES.DNAImageEncryptor(key='4a1b3f8c9d2e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1', mode='CFB')
        g, b, r = [encryptor.encrypt_channel(channel) for channel in (g, b, r)]
        '''
        encryptor = A_Sbox.ChaosSBoxEncryptor(hex_str, 'CFB',initial_state,iv[1])
        g,b,r = [encryptor.encrypt_image(channel) for channel in (g,b,r)]
        g, b, r = [A_Arnold.main_Arnold(channel, state) for channel in (g, b, r)]


    encrypted_image = cv2.merge([g, b, r])
    save_with_dpi(file, encrypted_image)  # 替换 cv2.imwrite
    return encrypted_image

def decryption(image, file):
    g, b, r = cv2.split(image)
    w, h, _ = image.shape
    state = 'decryption'
    block = 512
    seq_len = w * h
    seqs,initial_state,iv = pre_en_and_de(seq_len)

    iv_key = seqs[5][:256]  # 取前 256 个整数
    assert len(iv_key) == 256
    hex_str = ''.join([f"{x:02x}" for x in iv_key])
    hex_str = hex_str[:64]
    for i in range(1):
        g, b, r = [A_Arnold.main_Arnold(channel, state) for channel in (g, b, r)]
        '''
        decryptor = A_DNA_AES.DNAImageEncryptor(key='4a1b3f8c9d2e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1', mode='CFB')
        g, b, r = [decryptor.decrypt_channel(channel) for channel in (g, b, r)]
        '''
        decryptor = A_Sbox.ChaosSBoxEncryptor(hex_str, 'CFB',initial_state,iv[1])
        g,b,r = [decryptor.decrypt_image(channel) for channel in (g,b,r)]

        g, b, r = A_cross.main_Cross(g, b, r, state, seqs,iv[0])
    decrypted_image = cv2.merge([g, b, r])
    save_with_dpi(file, decrypted_image)  # 替换 cv2.imwrite
    return decrypted_image

def main_function(image,image_name):
    file1 = image_name +'_encrypted'+ '.png'
    file2 = image_name +'_decrypted'+'.png'

    import time
    large_matrix = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    start = time.time()
    encrypted_image = encryption(image, file1)
    encrypt_time = time.time() - start
    start = time.time()
    decrypted_image = decryption(encrypted_image, file2)
    decrypt_time = time.time() - start
    print(f"加密512x512图像耗时: {encrypt_time:.4f}秒")
    print(f"解密512x512图像耗时: {decrypt_time:.4f}秒")
    print(f"加解密总耗时: {encrypt_time+decrypt_time:.4f}秒")

    flag = images_identical(image, decrypted_image)
    print("是否还原成功:", flag)

def uaci_and_npcr_rgb(image):
    file1 = 'uaci_and_npcr_plain.png'
    file2 = 'uaci_and_npcr_modify.png'
    w,h= image.shape[:2]
    for data in [1, 5, 10, 50, 100, 200]:
        # Test position [1,1]
        modified_image = image.copy()
        original = encryption(modified_image, file1)
        modified_image[1, 1] = (modified_image[1, 1] + data) % 256
        encrypted = encryption(modified_image, file2)

        # Calculate for each channel
        for channel in range(3):
            diff = original[..., channel] != encrypted[..., channel]
            npcr = np.sum(diff) / original[..., channel].size * 100
            uaci = np.mean(np.abs(original[..., channel].astype(int) - encrypted[..., channel].astype(int))) / 255 * 100
            print(f"data {data:3d} [1,1]    Channel {['R','G','B'][channel]} - NPCR: {npcr:.4f}% UACI: {uaci:.4f}%")

        # Test position [255,255]
        modified_image = image.copy()
        modified_image[255, 255] = (modified_image[255, 255] + data) % 256
        encrypted = encryption(modified_image, file2)

        for channel in range(3):
            diff = original[..., channel] != encrypted[..., channel]
            npcr = np.sum(diff) / original[..., channel].size * 100
            uaci = np.mean(np.abs(original[..., channel].astype(int) - encrypted[..., channel].astype(int))) / 255 * 100
            print(f"data {data:3d} [255,255] Channel {['R','G','B'][channel]} - NPCR: {npcr:.4f}% UACI: {uaci:.4f}%")
        if(w == 512):
            # Test position [511,511]
            modified_image = image.copy()
            modified_image[511, 511] = (modified_image[511, 511] + data) % 256
            encrypted = encryption(modified_image, file2)

            for channel in range(3):
                diff = original[..., channel] != encrypted[..., channel]
                npcr = np.sum(diff) / original[..., channel].size * 100
                uaci = np.mean(np.abs(original[..., channel].astype(int) - encrypted[..., channel].astype(int))) / 255 * 100
                print(f"data {data:3d} [511,511] Channel {['R','G','B'][channel]} - NPCR: {npcr:.4f}% UACI: {uaci:.4f}%")

def crop_attack(image,image_name):
    w,h,_ = image.shape
    image1 = image.copy()
    image2 = image.copy()
    image3 = image.copy()
    image4 = image.copy()
    image5 = image.copy()
    for i in range(h):
        for j in range(w):
            if j>=200 and j<=250 and i>=200 and i<=250:
                image1[i,j] = 0
            if j>=0 and j<=h and i>=200 and i<=250:
                image1[i,j] = 0
            if j>=200 and j<=250 and i>=200 and i<=250:
                image1[j,i] = 0
            if j>=0 and j<=h and i>=200 and i<=250:
                image1[j,i] = 0

    for i in range(h):
        for j in range(w):
            if j>=200 and j<=250 and i>=200 and i<=250:
                image2[i,j] = 0
            if j>=0 and j<=h and i>=200 and i<=250:
                image2[i,j] = 0

    for i in range(h):
        for j in range(w):
            if j>=200 and j<=250 and i>=200 and i<=250:
                image3[j,i] = 0
            if j>=0 and j<=h and i>=200 and i<=250:
                image3[j,i] = 0

    for i in range(h):
        for j in range(w):
            if j>=0 and j<=256 and i>=0 and i<=256:
                image4[j,i] = 0

    for i in range(h):
        for j in range(w):
            if j>=0 and j<=256 and i>=0 and i<=512:
                image5[j,i] = 0
    print(' crop start')
    '''
    save_with_dpi(image_name+"_cropped_image1.png", image1)  # 替换 cv2.imwrite
    decryption(image1, image_name+"decrypt_cropped_image1.png")
    save_with_dpi(image_name+"_cropped_image2.png", image2)  # 替换 cv2.imwrite
    decryption(image2, image_name+"decrypt_cropped_image2.png")
    save_with_dpi(image_name+"_cropped_image3.png", image3)  # 替换 cv2.imwrite
    decryption(image3, image_name+"decrypt_cropped_cropped_image3.png")
    
    save_with_dpi(image_name+"_cropped_image4.png", image4)  # 替换 cv2.imwrite
    decryption(image4, image_name+"decrypt_cropped_cropped_image4.png")
    print("crop_image_finished")
    '''
    save_with_dpi(image_name+"_cropped_image5.png", image5)  # 替换 cv2.imwrite
    decryption(image5, image_name+"decrypt_cropped_cropped_image5.png")
    print("crop_image_finished")


def images_identical(img1, img2):
    if img1.shape != img2.shape:
        return False
    return np.array_equal(img1, img2)

def collect_pairs_all(channel):
    M, N = channel.shape
    x = np.random.randint(0, M - 1, 5000)
    y = np.random.randint(0, N - 1, 5000)

    # Horizontal: y vs y+1
    h1 = channel[x, y]
    h2 = channel[x, y + 1]
    h_dir = np.zeros_like(h1)

    # Vertical: x vs x+1
    v1 = channel[x, y]
    v2 = channel[x + 1, y]
    v_dir = np.ones_like(v1)

    # Diagonal: x,y vs x+1, y+1
    d1 = channel[x, y]
    d2 = channel[x + 1, y + 1]
    d_dir = np.full_like(d1, 2)

    x_all = np.concatenate([h1, v1, d1])
    y_all = np.concatenate([h_dir, v_dir, d_dir])
    z_all = np.concatenate([h2, v2, d2])

    return x_all, y_all, z_all

def plot_3d_channel(ax, channel, title, color):
    x, y, z = collect_pairs_all(channel)
    ax.scatter(y, x, z, c=color, marker='.', s=1)

    ax.set_ylabel('Pixel value')
    ax.set_zlabel('Adjacent pixel value')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['H', 'V', 'D'])
    ax.invert_yaxis()
    ax.grid(False)
    ax.view_init(elev=30, azim=45)


def quality_test(image_name):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    en_image = cv2.imread(image_name +'_encrypted.png', cv2.IMREAD_COLOR)
    '''

    #---------------------------计算差分攻击------------------------
    uaci_and_npcr_image = en_image.copy()
    #uaci_and_npcr_rgb(uaci_and_npcr_image)



    #---------------------------裁剪攻击----------------------------
    crop_image = en_image.copy()
    #crop_attack(crop_image,image_name)




    #---------------------------直方图--------------------------
    print("---histogram start---")
    his1 = cv2.imread(image_name)
    his2 = cv2.imread(image_name +'_encrypted'+ '.png')
    hist_r1,hist_g1,hist_b1= B_quality_test.histogram_test(his1)
    hist_r2,hist_g2,hist_b2 = B_quality_test.histogram_test(his2)

    plt.figure(figsize=(6, 5))
    x = np.arange(256)  # x轴表示像素强度
    plt.subplot(3, 1, 1)
    plt.fill_between(x, hist_r1.flatten(), color='r', alpha=0.5, label='Red Channel')

    plt.ylim(0, np.max(hist_r1) * 1.1)  # 动态设置纵坐标范围
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.fill_between(x, hist_g1.flatten(), color='g', alpha=0.5, label='Green Channel')

    plt.ylim(0, np.max(hist_g1) * 1.1)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.fill_between(x, hist_b1.flatten(), color='b', alpha=0.5, label='Blue Channel')

    plt.ylim(0, np.max(hist_b1) * 1.1)
    plt.legend()
    plt.savefig(image_name+"_his_plain.png", dpi=300, bbox_inches='tight',pad_inches=0.5)
    # 显示结果
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    x = np.arange(256)  # x轴表示像素强度
    plt.subplot(3, 1, 1)
    plt.fill_between(x, hist_r2.flatten(), color='r', alpha=0.5, label='Red Channel')

    plt.ylim(0, np.max(hist_r2) * 1.1)  # 动态设置纵坐标范围
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.fill_between(x, hist_g2.flatten(), color='g', alpha=0.5, label='Green Channel')

    plt.ylim(0, np.max(hist_g2) * 1.1)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.fill_between(x, hist_b2.flatten(), color='b', alpha=0.5, label='Blue Channel')


    plt.ylim(0, np.max(hist_b2) * 1.1)
    plt.legend()
    plt.savefig(image_name+"_his_encrypted.png", dpi=300, bbox_inches='tight', pad_inches=0.5)
    # 显示结果
    plt.tight_layout()
    plt.show()

    print("---histogram end---")


    
    
    #----------------------相关性分析和信息熵分析---------------------
    img1 = cv2.imread(image_name)
    img2 = cv2.imread(image_name +'_encrypted'+ '.png')
    original_channels,encrypted_channels = B_quality_test.init_correlation(img1, img2)
    titles = ['R Channel', 'G Channel', 'B Channel']

    colors = ['red', 'green', 'blue']  # 给 R/G/B 通道用的颜色

    fig = plt.figure(figsize=(14, 8))

    # 原图
    for i, (ch, t, color) in enumerate(zip(original_channels, titles, colors)):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        plot_3d_channel(ax, ch, f'Original {t}', color)

    # 加密图
    for i, (ch, t, color) in enumerate(zip(encrypted_channels, titles, colors)):
        ax = fig.add_subplot(2, 3, i + 4, projection='3d')
        plot_3d_channel(ax, ch, f'Encrypted {t}', color)
    plt.savefig(image_name+"_correlation.png", dpi=300 ,bbox_inches='tight', pad_inches=0.5)
    print('R1 原始图像， R2 加密图像')
    '''
    '''
    #-------------------------噪声攻击--------------------------------
    #-------------------------椒盐噪声--------------------------------
    en_image = cv2.imread(image_name +'_encrypted.png', cv2.IMREAD_COLOR)

    print("----noisy test start----")
    nosisy_image = B_quality_test.salt_and_pepper_noise(en_image, proportion=0.01)
    file1 = image_name +'_0.01_noisy_decrypted_image.png'
    decryption(nosisy_image, file1)
    print("25%")
    nosisy_image = B_quality_test.salt_and_pepper_noise(en_image, proportion=0.05)
    save_with_dpi(image_name +'_0.05_noisy_encrypted_image.png', nosisy_image)
    file1 = image_name +'_0.05_noisy_decrypted_image.png'
    decryption(nosisy_image, file1)
    print("50%")
    nosisy_image = B_quality_test.salt_and_pepper_noise(en_image, proportion=0.1)
    file1 = image_name +'_0.1_noisy_decrypted_image.png'
    decryption(nosisy_image, file1)
    print("75%")
    nosisy_image = B_quality_test.salt_and_pepper_noise(en_image, proportion=0.2)

    file1 = image_name +'_0.2_noisy_decrypted_image.png'
    decryption(nosisy_image, file1)
    print("100%")
    print("noisy attack finished")
    '''

def plaintext_attack_test(image, image_name):
    """
    文本攻击测试：改变原始图像的一个像素点，加密并保存结果进行对比
    """
    print("=== 文本攻击测试开始 ===")

    # 创建原始图像的副本
    original_image = image.copy()

    # 选择要修改的像素位置（例如中心点）
    height, width = image.shape[:2]
    center_x, center_y = height // 2, width // 2

    # 保存原始像素值
    original_pixel = original_image[center_x, center_y].copy()
    print(f"原始像素值 (位置 [{center_x}, {center_y}]): {original_pixel}")

    # 修改像素值（将RGB值都增加50）
    modified_image = image.copy()
    modified_pixel = (original_pixel + 50) % 256  # 确保值在0-255范围内
    modified_image[center_x, center_y] = modified_pixel
    print(f"修改后像素值: {modified_pixel}")

    # 加密原始图像
    original_encrypted_file = image_name + 'plaintext_attack_original_encrypted.png'
    original_encrypted = encryption(original_image, original_encrypted_file)

    # 加密修改后的图像
    modified_encrypted_file = image_name + 'plaintext_attack_modified_encrypted.png'
    modified_encrypted = encryption(modified_image, modified_encrypted_file)

    # 计算差异
    diff = np.sum(original_encrypted != modified_encrypted)
    total_pixels = original_encrypted.size
    change_percentage = (diff / total_pixels) * 100

    print(f"加密图像差异像素数: {diff}")
    print(f"总像素数: {total_pixels}")
    print(f"变化百分比: {change_percentage:.6f}%")

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始图像和加密图像
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(original_encrypted, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('原始图像加密结果')
    axes[0, 1].axis('off')

def test_image_all_0(image):
    main_function(image,"all_0_test2.png")



if __name__ == "__main__":

    #--------------------------------------------------更换图片的时候还需要修改A_6D_hyuperchaotic_system文件中的图片------------------------------------------------------------------
    image_name = "peppers.png"
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #main_function(image,image_name)



    #选择文本攻击
    image_0 =  np.zeros_like(image)
    image_0[0,0] = 255
    test_image_all_0(image_0)

    #plaintext_attack_test(image,image_name)
    print("------------50%--------------")
    #quality_test(image_name)
    print("------------100%--------------")
