import numpy as np

def encryption(image, seqs,iv):
    w, h = image.shape
    seq_length = len(seqs)
    temp = iv
    m = 0

    indices = []
    # 部分一：对角线
    for i in range(w - 1, 0, -1):
        for j in range(2 * i - 1, -1, -1):
            col = int(j + h // 3 - i)
            if 0 <= col < h:
                indices.append((i, col))
    for i in range(w-1,-1,-1):
        indices.append((i,3*w-1-i))



    # 部分二：反对角线
    for i in range(0, w - 1):
        for j in range(2 * (w - 1 - i)):
            col = int(j + h / 3 + i + 1)
            if 0 <= col < h:
                indices.append((i, col))
    for i in range(w-1,-1,-1):
        indices.append((i,w-1-i))


    flag = 0
    for i in range(0, w):
        for j in range(0, int(h/3) - 1):
            k = j + flag
            col = k if k >= 0 else k + h
            indices.append((i, col))
        flag -= 1
    for i in range(0,w,1):
        indices.append((i,w+i))

    indices = indices[::-1]
    # 类CFB加密逻辑
    for idx, (i, j) in enumerate(indices):
        key = seqs[idx % seq_length]
        plain = image[i, j]
        cipher = (plain ^ key ^ temp) % 256
        image[i, j] = cipher
        temp = cipher

    return image

def decryption(image, seqs,iv):
    w, h = image.shape
    seq_length = len(seqs)
    temp = iv
    m = 0

    indices = []
    # 部分一：对角线
    for i in range(w - 1, 0, -1):
        for j in range(2 * i - 1, -1, -1):
            col = int(j + h // 3 - i)
            if 0 <= col < h:
                indices.append((i, col))
    for i in range(w-1,-1,-1):
        indices.append((i,3*w-1-i))



    # 部分二：反对角线
    for i in range(0, w - 1):
        for j in range(2 * (w - 1 - i)):
            col = int(j + h / 3 + i + 1)
            if 0 <= col < h:
                indices.append((i, col))
    for i in range(w-1,-1,-1):
        indices.append((i,w-1-i))
    flag = 0
    for i in range(0, w):
        for j in range(0, int(h/3) - 1):
            k = j + flag
            col = k if k >= 0 else k + h
            indices.append((i, col))
        flag -= 1
    for i in range(0,w,1):
        indices.append((i,w+i))
    indices = indices[::-1]
    # 类CFB解密逻辑
    for idx, (i, j) in enumerate(indices):
        key = seqs[idx % seq_length]
        cipher = image[i, j]
        plain = (cipher ^ key ^temp ) % 256
        image[i, j] = plain
        temp = cipher

    return image

def main_Cross(g, b, r, mode, seqs,iv):
    w = r.shape[0]
    image = np.concatenate((g, b, r), axis=1)
    seq = []
    seq = [0] * (3 * w * w)  # 预分配空间
    for i in range(w * w):
        for j in range(3):
            seq[i * 3 + j] = seqs[j][i]  # 或其他计算方式
    if mode == 'encryption':
        image = encryption(image, seq,iv)
    elif mode == 'decryption':
        image = decryption(image, seq,iv)

    split_pos = w
    g = image[:, :split_pos]
    b = image[:, split_pos:2 * split_pos]
    r = image[:, 2 * split_pos:]
    return g, b, r
if __name__ == "__main__":
    # ========== 测试验证 ==========

    g = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
    b = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
    r = np.random.randint(0, 256, (4, 4), dtype=np.uint8)

    original_g = g.copy()
    original_b = b.copy()
    original_r = r.copy()

    seqs = np.random.randint(0, 256, (3, 48), dtype=np.uint8)
    print("密钥序列：\n", seqs)

    g_enc, b_enc, r_enc = main_Cross(g.copy(), b.copy(), r.copy(), 'encryption', seqs,94)
    g_dec, b_dec, r_dec = main_Cross(g_enc.copy(), b_enc.copy(), r_enc.copy(), 'decryption', seqs,94)

    is_same = np.array_equal(g_dec, original_g) and np.array_equal(b_dec, original_b) and np.array_equal(r_dec, original_r)

    print("✅ 解密后是否与原图一致？", is_same)
    print("原始 G/B/R 通道：\n", original_g, original_b, original_r)
    print("解密后 G/B/R 通道：\n", g_dec, b_dec, r_dec)

