import numpy as np
import hashlib
import random
from typing import Dict, Tuple
from pprint import pprint
from scipy.integrate import solve_ivp

class ChaosSBoxGenerator:
    def __init__(self, seed: bytes):
        self.seed = seed

    def _chaotic_system(self, t, state, params):
        a, b, c, d, e, f, h = params
        x, y, z, w, u, q = state
        dxdt = a * x * y
        dydt = c - a * x**2 + b * z
        dzdt = -b * y + 1.1 * (w**2 - d + u * z)
        dwdt = -c * z * w + 0.5 * q * y
        dudt = e * y * z - f * x * u
        dqdt = h * w + u - q
        return [dxdt, dydt, dzdt, dwdt, dudt, dqdt]

    def generate_sbox(self) -> np.ndarray:
        # 将seed转换为初始值和参数
        seed_int = int.from_bytes(self.seed, 'big')
        np.random.seed(seed_int % 2**32)

        init_state = np.random.rand(6)  # 6个状态变量
        params = [6.8,8,15,20,1,2,-20]  # 参数范围 0~10

        # 使用solve_ivp求解，增加采样点数量
        sol = solve_ivp(fun=self._chaotic_system,
                       t_span=(0, 100),
                       y0=init_state,
                       args=(params,),
                       t_eval=np.linspace(0, 100, 2000),  # 总点数增加到6000
                       method='RK45')

        sol.y = [y[1000:] for y in sol.y]  # 对每个变量的时间序列切片
        sol.t = sol.t[1000:]               # 同时截取对应的时间点`

        # 检查求解结果的有效性
        if not sol.success:
            raise RuntimeError("混沌系统求解失败")

        # 确定每个变量的实际可用点数
        min_length = min(len(y) for y in sol.y)

        total_points = min_length * 6
        chaos_values = np.empty(total_points)

        # 交替合并6个变量的值
        for i in range(6):
            chaos_values[i::6] = sol.y[i][:min_length]  # 使用实际可用点数

        # 移除无效值(NaN/inf)
        chaos_values = chaos_values[np.isfinite(chaos_values)]
        if len(chaos_values) == 0:
            raise RuntimeError("混沌系统未生成有效数据")

        # 归一化到0-255范围
        scaled_values = ((chaos_values - np.min(chaos_values)) * 255 /
                       (np.max(chaos_values) - np.min(chaos_values) + 1e-10)).astype(int)

        # 确保唯一性
        unique_values = []
        seen = set()
        for val in scaled_values:
            val = int(val) % 256  # 确保在0-255范围内
            if val not in seen:
                seen.add(val)
                unique_values.append(val)
                if len(unique_values) == 256:
                    break

        # 如果不足256个唯一值，补充剩余值
        if len(unique_values) < 256:
            remaining = list(set(range(256)) - set(unique_values))
            unique_values.extend(remaining[:256-len(unique_values)])

        # 添加索引并排序
        indexed_chaos = list(zip(unique_values, range(256)))
        indexed_chaos.sort()  # 按混沌值排序

        # 生成S盒
        sbox = np.array([idx for val, idx in indexed_chaos], dtype=np.uint8)
        return sbox

    def generate_inverse_sbox(self, sbox: np.ndarray) -> np.ndarray:
        inv_sbox = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            inv_sbox[sbox[i]] = i
        return inv_sbox

class ChaosSBoxEncryptor:
    def __init__(self, key: str, mode: str = 'CFB'):
        """
        初始化加密器
        :param key: 加密密钥
        :param mode: 加密模式(CFB/CBC/ECB/CTR)
        """
        # 密钥派生
        key_hash = hashlib.sha256(key.encode()).digest()
        self.original_iv = int.from_bytes(key_hash[:16], 'big') % 256
        self.counter = self.original_iv
        self.mode = mode.upper()
        random.seed(key_hash[0])

        # 生成4个混沌S盒和逆S盒
        self.sboxes = []
        self.inv_sboxes = []
        for i in range(4):
            seed = key_hash[i*8:(i+1)*8]  # 使用密钥的不同部分生成不同S盒
            generator = ChaosSBoxGenerator(seed)
            sbox = generator.generate_sbox()
            inv_sbox = generator.generate_inverse_sbox(sbox)

            # 转换为字典格式方便使用
            sbox_dict = {i: int(sbox[i]) for i in range(256)}
            inv_sbox_dict = {i: int(inv_sbox[i]) for i in range(256)}

            self.sboxes.append(sbox_dict)
            self.inv_sboxes.append(inv_sbox_dict)

        # 生成置换表
        self.permutation_table = self._generate_permutation_table(key_hash)
        self.inv_permutation_table = {v: k for k, v in self.permutation_table.items()}

    def _generate_inverse_sbox(self, sbox: dict) -> dict:
        return {v: k for k, v in sbox.items()}

    def _generate_permutation_table(self, key_hash: bytes) -> dict:
        perm = list(range(256))
        random.seed(key_hash[0])
        random.shuffle(perm)
        return {i: perm[i] for i in range(256)}

    def _encrypt_block(self, byte: int, index: int) -> int:
        sbox = self.sboxes[index % 4]
        byte = sbox[byte]
        byte = self.permutation_table[byte]
        return byte

    def _decrypt_block(self, byte: int, index: int) -> int:
        inv_sbox = self.inv_sboxes[index % 4]
        byte = self.inv_permutation_table[byte]
        byte = inv_sbox[byte]
        return byte

    def _increment_counter(self) -> int:
        self.counter = (self.counter + 1) % 256
        return self.counter

    def encrypt_image(self, img_array: np.ndarray) -> np.ndarray:
        if not isinstance(img_array, np.ndarray) or img_array.dtype != np.uint8:
            raise ValueError("输入必须是uint8类型的numpy数组")

        flat = img_array.flatten()
        encrypted = []
        iv = self.original_iv

        for i, byte in enumerate(flat):
            if self.mode == 'CFB':
                enc = self._encrypt_block(iv, i)
                cipher = byte ^ enc
                iv = cipher
            elif self.mode == 'CBC':
                xor = byte ^ iv
                cipher = self._encrypt_block(xor, i)
                iv = cipher
            elif self.mode == 'ECB':
                cipher = self._encrypt_block(byte, i)
            elif self.mode == 'CTR':
                ctr_enc = self._encrypt_block(iv, i)
                cipher = byte ^ ctr_enc
                iv = self._increment_counter()
            else:
                raise ValueError("不支持的加密模式")
            encrypted.append(cipher)

        return np.array(encrypted, dtype=np.uint8).reshape(img_array.shape)

    def decrypt_image(self, img_array: np.ndarray) -> np.ndarray:
        if not isinstance(img_array, np.ndarray) or img_array.dtype != np.uint8:
            raise ValueError("输入必须是uint8类型的numpy数组")

        flat = img_array.flatten()
        decrypted = []
        iv = self.original_iv

        for i, byte in enumerate(flat):
            if self.mode == 'CFB':
                enc = self._encrypt_block(iv, i)
                plain = byte ^ enc
                iv = byte
            elif self.mode == 'CBC':
                dec = self._decrypt_block(byte, i)
                plain = dec ^ iv
                iv = byte
            elif self.mode == 'ECB':
                plain = self._decrypt_block(byte, i)
            elif self.mode == 'CTR':
                ctr_enc = self._encrypt_block(iv, i)
                plain = byte ^ ctr_enc
                iv = self._increment_counter()
            else:
                raise ValueError("不支持的加密模式")
            decrypted.append(plain)

        return np.array(decrypted, dtype=np.uint8).reshape(img_array.shape)

    def verify_sbox_properties(self):
        """验证所有S盒的密码学性质"""
        results = {}
        for i, sbox in enumerate(self.sboxes):
            # 验证是否为双射
            is_bijective = len(set(sbox.values())) == 256
            # 验证固定点(不应有太多固定点)
            fixed_points = sum(1 for k, v in sbox.items() if k == v)

            results[f'SBox{i+1}'] = {
                'Bijective': is_bijective,
                'FixedPoints': fixed_points,
                'ExampleMapping': {0: sbox[0], 255: sbox[255]}  # 示例映射
            }
        return results

def visualize_matrix(matrix, size=4):
    """可视化矩阵的左上角部分"""
    for row in matrix[:size]:
        print(" ".join(f"{val:3}" for val in row[:size]))

def enhanced_test_suite():
    """增强的测试套件"""
    print("="*50)
    print("开始加密系统测试")

    # 测试1: 基本功能测试
    print("\n[测试1] 基本加密解密验证")
    test_matrix = np.array([[(i + j) % 256 for i in range(16)] for j in range(16)], dtype=np.uint8)
    key = "test_key_12345"

    for mode in ['CFB', 'CBC', 'ECB', 'CTR']:
        encryptor = ChaosSBoxEncryptor(key, mode=mode)
        encrypted = encryptor.encrypt_image(test_matrix)
        decrypted = encryptor.decrypt_image(encrypted)
        assert np.array_equal(test_matrix, decrypted), f"{mode}模式解密失败"
        print(f"✓ {mode}模式测试通过")

    # 测试2: S盒性质验证
    print("\n[测试2] S盒密码学性质验证")
    encryptor = ChaosSBoxEncryptor(key)
    sbox_properties = encryptor.verify_sbox_properties()
    pprint(sbox_properties)

    # 测试3: 雪崩效应测试
    print("\n[测试3] 雪崩效应测试")
    test_byte = 128
    encrypted1 = encryptor._encrypt_block(test_byte, 0)

    # 改变1位密钥
    encryptor2 = ChaosSBoxEncryptor(key + "x")
    encrypted2 = encryptor2._encrypt_block(test_byte, 0)

    # 计算比特变化率
    xor_result = encrypted1 ^ encrypted2
    changed_bits = bin(xor_result).count('1')
    change_rate = changed_bits / 8 * 100
    print(f"密钥微小变化导致输出字节{change_rate:.2f}%的比特变化")

    # 测试4: 加密效果可视化
    print("\n[测试4] 加密效果可视化")
    print("原始矩阵 (左上角4x4):")
    visualize_matrix(test_matrix)

    encrypted = encryptor.encrypt_image(test_matrix)
    print("\n加密后矩阵 (左上角4x4):")
    visualize_matrix(encrypted)

    # 测试5: 性能测试
    print("\n[测试5] 性能测试")
    import time
    large_matrix = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    start = time.time()
    encrypted = encryptor.encrypt_image(large_matrix)
    encrypt_time = time.time() - start

    start = time.time()
    decrypted = encryptor.decrypt_image(encrypted)
    decrypt_time = time.time() - start

    print(f"加密512x512图像耗时: {encrypt_time:.4f}秒")
    print(f"解密512x512图像耗时: {decrypt_time:.4f}秒")
    print(f"加解密总耗时: {encrypt_time+decrypt_time:.4f}秒")
    assert np.array_equal(large_matrix, decrypted), "大矩阵解密失败"

    print("\n所有测试通过!")
def test_sbox_generation():
    print("测试S盒生成...")
    key = "test_key_12345"
    key_hash = hashlib.sha256(key.encode()).digest()

    for i in range(4):
        seed = key_hash[i*8:(i+1)*8]
        generator = ChaosSBoxGenerator(seed)
        sbox = generator.generate_sbox()

        print(f"SBox{i+1}:")
        print("长度:", len(sbox))
        print("唯一值数量:", len(set(sbox)))
        print("最小值:", min(sbox), "最大值:", max(sbox))

        # 验证是否为0-255的排列
        assert sorted(sbox) == list(range(256)), "S盒不是完整排列"

if __name__ == "__main__":
    test_sbox_generation()  # 先单独测试S盒生成
    enhanced_test_suite()   # 再运行完整测试
