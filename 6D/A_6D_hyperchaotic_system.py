import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import hashlib
import numpy as np
import cv2

import hashlib

def hash_sha256_text(message):
    sha256 = hashlib.sha256()
    sha256.update(message.encode('utf-8'))
    return sha256.digest()  # 返回 bytes

def hash_sha256_image(image):
    sha256 = hashlib.sha256()
    image_bytes = image.tobytes()
    sha256.update(image_bytes)
    return sha256.digest()  # 返回 bytes

def xor_operater(bytes1, bytes2):
    """对两个 bytes 对象逐字节异或，返回二进制字符串"""
    xor_result = bytearray()
    for b1, b2 in zip(bytes1, bytes2):
        xor_result.append(b1 ^ b2)
    return ''.join(format(byte, '08b') for byte in xor_result)  # 返回二进制字符串

def init_chaotic_system(image, message):
    # 1. 补全消息到 256 位
    strlen = len(message)
    if strlen != 256:
        message += '0' * (256 - strlen)

    # 2. 分割消息并计算哈希
    mid = len(message) // 2
    str1, str2 = message[:mid], message[mid:]
    hash_str1 = hash_sha256_text(str1)
    hash_str2 = hash_sha256_text(str2)

    # 3. 异或两个哈希值（返回二进制字符串）
    text_hash_binary = xor_operater(hash_str1, hash_str2)

    # 4. 计算图像哈希（bytes）
    img_hash = hash_sha256_image(image)

    # 5. 将二进制字符串转为 bytes 再与图像哈希异或
    # 先将二进制字符串转为整数，再转为 bytes
    text_hash_int = int(text_hash_binary, 2)
    text_hash_bytes = text_hash_int.to_bytes((text_hash_int.bit_length() + 7) // 8, 'big')

    # 补齐长度到 32 字节（SHA-256 哈希长度）
    if len(text_hash_bytes) < 32:
        text_hash_bytes = text_hash_bytes.ljust(32, b'\x00')

    # 异或操作（返回二进制字符串）
    final_binary = xor_operater(text_hash_bytes, img_hash)

    # 6. 分割二进制字符串为 6 个 40 位数字
    chunks = [final_binary[i:i+40] for i in range(0, 240, 40)]
    nums = [int(chunk, 2) / (1 << 40) for chunk in chunks]  # 归一化到 [0, 1)

    return nums[:6]  # 确保返回 6 个数

def system(t, state, params):
    a, b, c, d,e,f,h = params
    x, y, z, w ,u,q= state
    dxdt = a * x * y
    dydt = c - a * x**2 + b* z
    dzdt = -b * y +  1.1 *( w**2 - d+ u*z)
    dwdt = -c * z * w + 0.5 *q*y
    dudt =  e*y*z - f*x*u
    dqdt = h * w + u - q
    return [dxdt, dydt, dzdt, dwdt, dudt,dqdt]

def generate_system_value(image,params,len,key):
    t_span = (0, 100)  # 让系统充分演化
    t_eval = np.linspace(50, 100, len)  # 取后期稳定值
    # 求解微分方程
    #-------------------------------------------------------------------------------key-------------------------------------------------------------------

    initial_state = init_chaotic_system(image,key)
    sol = solve_ivp(system, t_span, initial_state, args=(params,), t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

    iv = hashlib.sha512(key.encode()).digest()
    iv1 = iv[0]
    iv2 = iv[32]
    for i in range(1,1,32):
        iv1 ^=iv[i]
    for i in range(33,1,64):
        iv2 ^=iv[i]
    iv = [iv1,iv2]
    return sol,initial_state,iv

#这里的接口需要传输的initial_state=[y1,y2,y3,y4,y5,y6],返回解后的值
def init_system_parms(len, key = 'fa1b3f8c9d2e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1'):
    params = [6.8,8,15,20,1,2,-20]
    image = cv2.imread('Airplane.png', cv2.IMREAD_COLOR)
    sol,initial_state,iv = generate_system_value(image,params,len,key)
    return sol,initial_state,iv
'''
key1 = '4a1b3f8c9d2e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1'
key2 = '9a1b3f8c9d2e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1'
key3 = 'fa1b3f8c9d2e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1'

sol1,initial_state1,iv1 = initial_state = init_system_parms(256,key1)
sol2,initial_state2,iv2 = initial_state = init_system_parms(256,key2)
sol3,initial_state3,iv3 = initial_state = init_system_parms(256,key3)
print(initial_state1,iv1)
print(initial_state2,iv2)
print(initial_state3,iv3)
'''
