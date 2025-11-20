import numpy as np
import A_DNA
import A_DNA_AES
from hashlib import sha256
'''
def block_encryption(image_block, encryptor):
    """加密 2x2 图像块"""
    encryptor = A_DNA_AES.DNAImageEncryptor("My_Secret_Key_123")
    encrypted_img = encryptor.encrypt_image(image_block)
    return encrypted_img

def block_decryption(encrypted_block, encryptor, original_shape):
    """解密 2x2 图像块"""
    decrypted_img = encryptor.decrypt_image(encrypted_block)
    return decrypted_img

def confusion_operator(image, block_size,encryptor ):
    """加密整个图像"""
    h, w = image.shape
    encrypted_image = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            encrypted_block = block_encryption(block, encryptor)
            encrypted_image[i:i+block_size, j:j+block_size] = encrypted_block
    return encrypted_image

def de_confusion_operator(encrypted_image, block_size, encryptor, original_shape):
    """解密整个图像"""
    h, w = encrypted_image.shape
    original_h, original_w = original_shape
    decrypted_image = np.zeros(original_shape, dtype=np.uint8)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = encrypted_image[i:i+block_size, j:j+block_size]
            decrypted_block = block_decryption(block, encryptor,(block_size, block_size))
            decrypted_image[i:i+block_size, j:j+block_size] = decrypted_block
    return decrypted_image

def confusion_main(image, mode, block_size, seqs):
    """主函数，选择加密或解密"""
    encryptor = A_DNA_AES.DNAImageEncryptor("My_Secret_Key_123")
    if mode == 'encryption':
        return confusion_operator(image, block_size, encryptor)
    elif mode == 'decryption':
        # Need to pass original shape for decryption
        original_shape = (image.shape[0], image.shape[1])
        return de_confusion_operator(image, block_size, encryptor, original_shape)
    else:
        raise ValueError("mode must be 'encryption' or 'decryption'")

# Test usage
image = np.zeros((8,8), dtype=np.uint8)
image[0][0] = 1
seqs = 0
block_size = 4
encryptor = A_DNA_AES.DNAImageEncryptor("My_Secret_Key_123")
# Encryption
encrypted = confusion_operator(image, block_size, encryptor)

# Decryption - need to pass original shape
decrypted = de_confusion_operator(encrypted, block_size, encryptor, image.shape)

print(image)
print(encrypted)
print( decrypted)

'''

import numpy as np
import A_DNA_AES
from typing import Tuple

class BlockCryptoProcessor:
    def __init__(self, key: str = "My_Secret_Key_123"):
        """初始化加密处理器"""
        self.encryptor = A_DNA_AES.DNAImageEncryptor(key)
        # 获取加密器使用的IV，确保加解密一致
        self.iv = self.encryptor.get_iv()

    def _process_block(self, block: np.ndarray, encrypt: bool) -> np.ndarray:
        """处理单个块(加密/解密)"""
        if encrypt:
            return self.encryptor.encrypt_image(block)
        else:
            return self.encryptor.decrypt_image(block)

    def _pad_block(self, block: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """填充块到目标形状"""
        h, w = block.shape
        t_h, t_w = target_shape
        padded = np.zeros(target_shape, dtype=block.dtype)
        padded[:min(h, t_h), :min(w, t_w)] = block[:min(h, t_h), :min(w, t_w)]
        return padded

    def process_image(self, image: np.ndarray, block_size: int, encrypt: bool) -> np.ndarray:
        """处理整个图像"""
        h, w = image.shape
        # 计算填充后的尺寸
        pad_h = ((h + block_size - 1) // block_size) * block_size
        pad_w = ((w + block_size - 1) // block_size) * block_size

        # 创建结果数组
        result = np.zeros((pad_h, pad_w), dtype=image.dtype)

        for i in range(0, pad_h, block_size):
            for j in range(0, pad_w, block_size):
                # 获取当前块(处理边界情况)
                block = image[i:i+block_size, j:j+block_size]
                # 如果块尺寸不足，先填充
                if block.shape != (block_size, block_size):
                    block = self._pad_block(block, (block_size, block_size))

                # 处理块
                processed_block = self._process_block(block, encrypt)

                # 放回结果(保持原始尺寸)
                result[i:i+block_size, j:j+block_size] = processed_block

        # 返回时裁剪到原始尺寸
        return result[:h, :w]

# 测试用例
def confusion_main(image, state, block, processor):
    if state == 'encryption':
        encrypted = processor.process_image(image, block, encrypt=True)
        return encrypted
    if state == 'decryption':
        decrypted = processor.process_image(image, block, encrypt=False)
        return decrypted


'''
def en(test_img,processor):
    encrypted = confusion_main(test_img,'encryption',4,processor)
    print("加密结果:")
    print(encrypted)
    return encrypted
def de(encrypted,processor):
    decrypted = confusion_main(encrypted,'decryption',4,processor)
    print("\n解密结果:")
    print(decrypted)
    return decrypted
# 创建测试图像
processor = BlockCryptoProcessor()
test_img = np.zeros((8, 8), dtype=np.uint8)
test_img[0:4, 0:4] = 1  # 左上角4x4区域设为1
print(type(test_img))
# 初始化处理器
#processor = BlockCryptoProcessor()
print(test_img)
# 加密

#encrypted = processor.process_image(test_img, block_size=4, encrypt=True)
encrypted = en(test_img,processor)
decrypted = de(encrypted,processor)

# 解密
#decrypted = confusion_main(encrypted,'decryption',4,processor)
#decrypted = processor.process_image(encrypted, block_size=4, encrypt=False)


# 验证
print("\n原始与解密结果是否一致:", np.array_equal(test_img, decrypted))

# 创建测试图像
test_img = np.zeros((8, 8), dtype=np.uint8)
test_img[0:4, 0:4] = 1  # 左上角4x4区域设为1
print(type(test_img))
# 初始化处理器
processor = BlockCryptoProcessor()
print(test_img)
# 加密

encrypted = processor.process_image(test_img, block_size=4, encrypt=True)
print("加密结果:")
print(encrypted)

# 解密

decrypted = processor.process_image(encrypted, block_size=4, encrypt=False)
print("\n解密结果:")
print(decrypted)

# 验证
print("\n原始与解密结果是否一致:", np.array_equal(test_img, decrypted))

'''
