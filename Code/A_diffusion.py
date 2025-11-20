import numpy as np

def block_encryption(image_block, seq1_block, seq2_block, seq3_block):
    """加密 2x2 图像块"""
    w,h = image_block.shape
    seq1_block = seq1_block
    seq2_block = seq2_block
    seq3_block = seq3_block
    # 加密逻辑
    image_block ^= seq1_block
    for _ in range(1):

        for i in range(1,h):
            for j in range(1,w):
                image_block[i][j] ^=(image_block[i-1][j]^image_block[i][j-1])
        for i in range(h-2,-1,-1):
            for j in range(h-2,-1,-1):
                image_block[i][j] ^=(image_block[i+1][j]^image_block[i][j+1])
        return image_block

def block_decryption(image_block, seq1_block, seq2_block, seq3_block):
    """解密 2x2 图像块"""
    w,h = image_block.shape
    seq1_block = seq1_block
    seq2_block = seq2_block
    seq3_block = seq3_block

    # 解密逻辑（逆向操作）
    for _ in range(1):
        for i in range(0,h-1):
            for j in range(0,w-1):
                image_block[i][j] ^=(image_block[i+1][j]^image_block[i][j+1])
        for i in range(h-1, 0, -1):
            for j in range(w-1, 0, -1):
                image_block[i][j] ^=(image_block[i-1][j]^image_block[i][j-1])
    image_block ^= seq1_block
    return image_block

def confusion_operator(image, block_size, seqs):
    """加密整个图像"""
    h, w = image.shape
    # 将混沌序列转换为 uint8 并 reshape
    chaos_seq1 = np.asarray(seqs[0]).reshape(h, w)
    chaos_seq2 = np.asarray(seqs[1]).reshape(h, w)
    chaos_seq3 = np.asarray(seqs[2]).reshape(h, w)
    encrypted_image = np.zeros_like(image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            seq1 = chaos_seq1[i:i+block_size, j:j+block_size]
            seq2 = chaos_seq2[i:i+block_size, j:j+block_size]
            seq3 = chaos_seq3[i:i+block_size, j:j+block_size]

            encrypted_block = block_encryption(block, seq1, seq2, seq3)
            encrypted_image[i:i+block_size, j:j+block_size] = encrypted_block
    return encrypted_image

def de_confusion_operator(image, block_size, seqs):
    """解密整个图像"""
    h, w = image.shape
    # 将混沌序列转换为 uint8 并 reshape
    chaos_seq1 = np.asarray(seqs[0]).reshape(h, w)
    chaos_seq2 = np.asarray(seqs[1]).reshape(h, w)
    chaos_seq3 = np.asarray(seqs[2]).reshape(h, w)

    decrypted_image = np.zeros_like(image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            seq1 = chaos_seq1[i:i+block_size, j:j+block_size]
            seq2 = chaos_seq2[i:i+block_size, j:j+block_size]
            seq3 = chaos_seq3[i:i+block_size, j:j+block_size]

            decrypted_block = block_decryption(block, seq1, seq2, seq3)
            decrypted_image[i:i+block_size, j:j+block_size] = decrypted_block

    return decrypted_image

def confusion_main(image, mode, block_size, seqs):
    """主函数，选择加密或解密"""
    h, w = image.shape
    image = image.astype(np.uint8)  # 确保输入是 uint8

    if mode == 'encryption':
        return confusion_operator(image, block_size, seqs)
    elif mode == 'decryption':
        return de_confusion_operator(image, block_size, seqs)
    else:
        raise ValueError("mode must be 'encryption' or 'decryption'")
