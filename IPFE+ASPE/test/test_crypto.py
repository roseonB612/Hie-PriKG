import numpy as np
import sys
import os

# 核心修改：只退回到上一级目录（项目根目录），不要加 'test'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crypto_core.ipfe_encryption import IPFECore
from crypto_core.aspe_encryption import generate_invertible_matrix, encrypt_entity_aspe, generate_query_token_aspe

def test_ipfe_homomorphic_addition():
    """测试第一层 IPFE：密态内积是否等于明文内积"""
    dim = 384
    x = np.random.randn(dim)  # 模拟数据
    y = np.random.randn(dim)  # 模拟查询
    
    ipfe = IPFECore(dim, seed=42)
    ct_x = ipfe.encrypt(x)
    sk_y = ipfe.key_gen(y)
    
    decrypted_inner_product = ipfe.decrypt(ct_x, sk_y, y)
    true_inner_product = np.dot(x, y)
    
    # 断言：两者误差必须小于 1e-7
    assert abs(decrypted_inner_product - true_inner_product) < 1e-7, "IPFE 核心数学同态性遭到破坏！"

def test_aspe_inner_product_preservation():
    """测试第二层 ASPE：密文匹配得分是否与明文相关度成比例"""
    dim = 128 # 测试用较小维度即可
    ext_dim = dim + 2
    
    # 1. 模拟环境与密钥
    S = np.random.randint(0, 2, ext_dim)
    M1 = generate_invertible_matrix(ext_dim)
    M2 = generate_invertible_matrix(ext_dim)
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)
    
    # 2. 模拟数据与查询
    x = np.random.randn(dim)
    q = np.random.randn(dim)
    
    # 3. 加密与生成令牌 (固定 r=1.5 方便测试)
    fixed_r = 1.5
    ca, cb = encrypt_entity_aspe(x, S, M1, M2)
    qa, qb, r = generate_query_token_aspe(q, S, M1_inv, M2_inv, fixed_r=fixed_r)
    
    # 4. 计算得分
    secure_score = np.dot(ca, qa) + np.dot(cb, qb)
    plaintext_score = np.dot(x, q)
    
    # 5. 校验 ASPE 公式: Score = r * (x·q)
    expected_score = fixed_r * plaintext_score
    
    # 断言：校验安全标量积计算是否正确
    assert abs(secure_score - expected_score) < 1e-5, "ASPE 加密内积还原失败，请检查矩阵可逆性与乘法顺序！"