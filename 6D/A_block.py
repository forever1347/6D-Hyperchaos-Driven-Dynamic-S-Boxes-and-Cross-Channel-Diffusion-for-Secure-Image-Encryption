import numpy as np
import A_DNA
import A_DNA_AES
from hashlib import sha256

def block_encryption(image_block, seqs):
    """加密 2x2 图像块"""
    #image_block = A_DNA.encrypt_image(image_block, sha256(b"your_password").digest())
    encryptor = A_DNA_AES.DNAImageEncryptor("My_Secret_Key_123")
    encrypted_img = encryptor.encrypt_image(image_block)
    return encrypted_img

def block_decryption(encrypted_block, seqs, original_shape):
    """解密 2x2 图像块"""
    #decrypted_block = A_DNA.decrypt_image(encrypted_block,sha256(b"your_password").digest(), original_shape)
    encryptor = A_DNA_AES.DNAImageEncryptor("My_Secret_Key_123")
    decrypted_img = encryptor.decrypt_image(encrypted_block)
    return decrypted_img

def confusion_operator(image, block_size, seqs):
    """加密整个图像"""
    h, w = image.shape
    encrypted_image = np.zeros((2*h, 2*w), dtype=np.uint8)
    for i in range(0, h, block_size):
        m = 2*i
        for j in range(0, w, block_size):
            n = 2*j
            block = image[i:i+block_size, j:j+block_size]
            encrypted_block = block_encryption(block, seqs)
            encrypted_image[m:m+2*block_size, n:n+2*block_size] = encrypted_block
    return encrypted_image

def de_confusion_operator(encrypted_image, block_size, seqs, original_shape):
    """解密整个图像"""
    h, w = encrypted_image.shape
    original_h, original_w = original_shape
    block_size *= 2  # Since encrypted blocks are twice as large

    decrypted_image = np.zeros(original_shape, dtype=np.uint8)

    for i in range(0, h, block_size):
        m = i//2
        for j in range(0, w, block_size):
            n = j//2
            block = encrypted_image[i:i+block_size, j:j+block_size]
            decrypted_block = block_decryption(block, seqs,
                                             (block_size//2, block_size//2))
            decrypted_image[m:m+block_size//2, n:n+block_size//2] = decrypted_block
    return decrypted_image

def confusion_main(image, mode, block_size, seqs):
    """主函数，选择加密或解密"""
    if mode == 'encryption':
        return confusion_operator(image, block_size, seqs)
    elif mode == 'decryption':
        # Need to pass original shape for decryption
        original_shape = (image.shape[0]//2, image.shape[1]//2)
        return de_confusion_operator(image, block_size, seqs, original_shape)
    else:
        raise ValueError("mode must be 'encryption' or 'decryption'")
'''
# Test usage
image = np.zeros((8,8), dtype=np.uint8)
image[0][0] = 1
seqs = 0
block_size = 4

# Encryption
encrypted = confusion_operator(image, block_size, seqs)

# Decryption - need to pass original shape
decrypted = de_confusion_operator(encrypted, block_size, seqs, image.shape)

print(image)
print(encrypted)
print( decrypted)
'''
