import numpy as np
import hashlib
from scipy.integrate import solve_ivp
from typing import Dict, List
from pprint import pprint
import random
from itertools import product

class ChaosSBoxGenerator:
    def __init__(self, seed: bytes, initial_state: tuple[float, float, float, float, float, float] = None):
        self.seed = seed
        self.initial_state = initial_state

    def _chaotic_system(self, t, state, params):
        """保持原有的6D混沌系统不变"""
        a, b, c, d, e, f, h = params
        x, y, z, w, u, q = state
        dxdt = a * x * y
        dydt = c - a * x**2 + b * z
        dzdt = -b * y + 1.1 * (w**2 - d + u * z)
        dwdt = -c * z * w + 0.5 * q * y
        dudt = e * y * z - f * x * u
        dqdt = h * w + u - q
        return [dxdt, dydt, dzdt, dwdt, dudt, dqdt]

    def _generate_chaos_sequence(self, length: int) -> np.ndarray:
        """生成混沌序列用于排序"""
        seed_int = int.from_bytes(self.seed, 'big')
        np.random.seed(seed_int % (2**32))

        # 初始状态和参数
        if self.initial_state is not None:
            init_state = np.array(self.initial_state, dtype=np.float64)
        else:
            init_state = np.random.rand(6)

        params = [6.8, 8, 15, 20, 1, 2, -20]

        # 求解混沌系统
        sol = solve_ivp(
            fun=self._chaotic_system,
            t_span=(0, 100),
            y0=init_state,
            args=(params,),
            t_eval=np.linspace(0, 100, length * 2),  # 生成足够多的点
            method='RK45'
        )

        # 合并所有状态变量并截取所需长度
        chaos_seq = np.concatenate(sol.y)[:length]
        return chaos_seq

    def nonlinearity(self, sbox):
        """计算S盒的非线性度"""
        n = 8
        size = 1 << n
        nl = np.inf

        for bit in range(8):
            f = [(sbox[x] >> bit) & 1 for x in range(size)]
            walsh = np.zeros(size, dtype=int)

            for a in range(size):
                sum_val = 0
                for x in range(size):
                    a_dot_x = bin(a & x).count('1') % 2
                    fx_xor = f[x] ^ a_dot_x
                    sum_val += (-1) ** fx_xor
                walsh[a] = sum_val

            current_nl = (1 << (n - 1)) - 0.5 * np.max(np.abs(walsh))
            if current_nl < nl:
                nl = current_nl

        return int(nl)

    def differential_uniformity(self, sbox):
        """计算S盒的差分均匀性"""
        n = 8
        size = 1 << n
        ddt = np.zeros((size, size), dtype=int)

        for alpha in range(size):
            for x in range(size):
                beta = sbox[x] ^ sbox[x ^ alpha]
                ddt[alpha][beta] += 1

        max_du = np.max(ddt[1:])
        return max_du

    def algebraic_degree(self, sbox):
        """计算S盒的代数次数"""
        n = 8
        size = 1 << n
        max_degree = 0

        for bit in range(8):
            f = [(sbox[x] >> bit) & 1 for x in range(size)]
            anf = f.copy()

            for i in range(n):
                for x in range(size):
                    if (x >> i) & 1:
                        anf[x] ^= anf[x ^ (1 << i)]

            degree = bin(np.max(np.where(anf)[0])).count('1') if np.any(anf) else 0
            if degree > max_degree:
                max_degree = degree

        return max_degree

    def strict_avalanche_criterion(self, sbox):
        """计算S盒的严格雪崩准则"""
        n = 8
        size = 1 << n
        sac_matrix = np.zeros((n, n))

        for input_bit in range(n):
            mask = 1 << input_bit
            for output_bit in range(n):
                changes = 0
                for x in range(size):
                    x_flipped = x ^ mask
                    output_original = (sbox[x] >> output_bit) & 1
                    output_flipped = (sbox[x_flipped] >> output_bit) & 1
                    if output_original != output_flipped:
                        changes += 1
                sac_matrix[input_bit][output_bit] = changes / size

        sac_score = 1 - np.max(np.abs(sac_matrix - 0.5))
        return sac_score

    def sbox_fitness(self, sbox):
        """计算S盒的综合适应度"""
        nl = self.nonlinearity(sbox)
        du = self.differential_uniformity(sbox)
        ad = self.algebraic_degree(sbox)
        sac = self.strict_avalanche_criterion(sbox)

        # 通用图像加密权重
        fitness = 0.5 * nl + 0.3 * (1.0 / du) + 0.15 * ad + 0.05 * sac
        return fitness

    def generate_sbox(self) -> np.ndarray:
        """改进的S盒生成方法，使用混沌序列排序和适应度优化"""
        best_sbox = None
        best_fitness = -np.inf

        # 生成多个候选S盒，选择适应度最高的
        for _ in range(10):  # 生成10个候选S盒
            chaos_values = self._generate_chaos_sequence(256)
            indexed_pairs = [(i, chaos_values[i]) for i in range(256)]
            indexed_pairs.sort(key=lambda x: x[1])
            sbox = np.array([x[0] for x in indexed_pairs], dtype=np.uint8)

            # 验证S盒是否为双射
            assert len(set(sbox)) == 256 and sorted(sbox) == list(range(256)), "无效S盒"

            # 计算适应度
            fitness = self.sbox_fitness(sbox)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sbox = sbox

        return best_sbox

    def generate_inverse_sbox(self, sbox: np.ndarray) -> np.ndarray:
        """生成逆S盒"""
        inv_sbox = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            inv_sbox[sbox[i]] = i
        return inv_sbox

class ChaosSBoxEncryptor:
    def __init__(self, key: str, mode: str = 'CFB', initial_state: tuple[float, float, float, float, float, float] = None, iv: int = 0):
        """
        初始化加密器
        :param key: 加密密钥
        :param mode: 加密模式(CFB/CBC/ECB/CTR)
        """
        key_hash = hashlib.sha512(key.encode()).digest()
        print("sbox-keyhash:------------",key_hash)
        self.original_iv = iv
        self.counter = self.original_iv
        self.mode = mode.upper()

        # 生成8个混沌S盒和逆S盒
        self.sboxes = []
        self.inv_sboxes = []
        for i in range(8):
            seed = key_hash[i*8:(i+1)*8]
            generator = ChaosSBoxGenerator(seed, initial_state)
            sbox = generator.generate_sbox()
            inv_sbox = generator.generate_inverse_sbox(sbox)

            sbox_dict = {i: int(sbox[i]) for i in range(256)}
            inv_sbox_dict = {i: int(inv_sbox[i]) for i in range(256)}

            self.sboxes.append(sbox_dict)
            self.inv_sboxes.append(inv_sbox_dict)
        print("sbox",sbox_dict)
        # 生成置换表
        self.permutation_table = self._generate_permutation_table(key_hash)
        self.inv_permutation_table = {v: k for k, v in self.permutation_table.items()}

    def _generate_permutation_table(self, key_hash: bytes) -> dict:
        perm = list(range(256))
        random.seed(key_hash[0])
        random.shuffle(perm)
        return {i: perm[i] for i in range(256)}

    def _encrypt_block(self, byte: int, index: int) -> int:
        sbox = self.sboxes[index % 8]
        byte = sbox[byte]
        byte = self.permutation_table[byte]
        return byte

    def _decrypt_block(self, byte: int, index: int) -> int:
        inv_sbox = self.inv_sboxes[index % 8]
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
            sbox_array = np.array([sbox[x] for x in range(256)])
            generator = ChaosSBoxGenerator(b'', None)

            nl = generator.nonlinearity(sbox_array)
            du = generator.differential_uniformity(sbox_array)
            ad = generator.algebraic_degree(sbox_array)
            sac = generator.strict_avalanche_criterion(sbox_array)
            fitness = generator.sbox_fitness(sbox_array)

            results[f'SBox{i+1}'] = {
                'Nonlinearity': nl,
                'DifferentialUniformity': du,
                'AlgebraicDegree': ad,
                'SAC': sac,
                'Fitness': fitness,
                'FixedPoints': sum(1 for k, v in sbox.items() if k == v)
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
    initial_state = [1,1,1,1,1,1]
    for mode in ['CFB', 'CBC', 'ECB', 'CTR']:
        encryptor = ChaosSBoxEncryptor(key, mode, initial_state)
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
    encryptor2 = ChaosSBoxEncryptor(key + "x")
    encrypted2 = encryptor2._encrypt_block(test_byte, 0)
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
    decrypted = encryptor.decrypt_image(encrypted)
    print("\n解密后矩阵 (左上角4x4):")
    visualize_matrix(decrypted)

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

if __name__ == "__main__":
    enhanced_test_suite()
