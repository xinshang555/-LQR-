import parameter as p
import numpy as np
import math as m
from scipy.linalg import solve_continuous_are
np.set_printoptions(precision=3, suppress=True, linewidth=200)
#方程顺序表明
'''

整个状态空间方程如下
          [x'  ]             [x  ]             
          [x'' ]             [x' ]
          [y'  ]             [y  ]             [Tlw]
          [y'' ]             [y' ]             [Tll]
Martix_1* [tl' ] = Martix_2* [tl ] + Martix_3* [Trw] + Martix_4(常数矩阵,不重要,跟初始条件有关)
          [tl'']             [tl']             [Trl]
          [tr' ]             [tr ]
          [tr'']             [tr']
          [f'  ]             [f  ]
          [f'' ]             [f'']


'''
##计算函数（l_l：左腿虚拟腿长, l_r：右腿虚拟腿长）
def calculate(l_l, l_r, q=None, r=None):
    """解 LQR 反馈增益 K。

    参数
    ----
    l_l, l_r : float
        左/右腿虚拟腿长,m。平衡姿态下取 p.LEG_NOMINAL (0.040)。
    q : 长度10的序列,可选
        状态权重对角元,顺序 [x, x', y, y', tl, tl', tr, tr', f, f']。
        缺省用 parameter.Q_DEFAULT。
    r : 长度4的序列,可选
        输入权重对角元,顺序 [Tlw, Tll, Trw, Trl]。
        缺省用 parameter.R_DEFAULT。

    返回
    ----
    Martix_K : (4,10) ndarray
        控制律 u = -K @ x

    注:q/r 必须给出,否则 R 奇异,solve_continuous_are 必然失败。
    原实现把 q/r 硬编码为全 0,因此**从未能成功运行过**。
    """
#矩阵参数表明
    A = 2*p.m_w + 2*p.m_l + p.m_B + 2*p.J_wz/p.r**2
    B = -p.m_w*l_l - p.m_l*p.l_lc(l_l)*m.sin(p.theta_lc(l_l)) - p.J_wz*l_l/p.r**2
    C = -p.m_w*l_r - p.m_l*p.l_rc(l_r)*m.sin(p.theta_rc(l_r)) - p.J_wz*l_r/p.r**2
    D = -p.m_B*p.l_Bc*m.sin(p.theta_Bc)
    E = 2*p.m_w*p.d_z**2 + 2*p.J_wz*p.d_z**2/p.r**2 + 2*p.J_wy + p.m_l*(2*p.d_z**2 + p.l_rc(l_r)**2*m.cos(p.theta_rc(l_r))**2 + p.l_lc(l_l)**2*m.cos(p.theta_lc(l_l))**2) + p.J_rlcy + p.J_llcy + p.m_B*p.l_Bc**2*m.cos(p.theta_Bc)**2 + p.J_By
    F = -p.m_w*p.d_z*l_l - p.d_z*l_l*p.J_wz/p.r**2 - p.m_l*p.d_z*p.l_lc(l_l)*m.sin(p.theta_lc(l_l))
    G = p.m_w*p.d_z*l_r + p.d_z*l_r*p.J_wz/p.r**2 + p.m_l*p.d_z*p.l_rc(l_r)*m.sin(p.theta_rc(l_r))
    H = -p.m_w*l_l - p.J_wz*l_l/p.r**2 - p.m_l*p.l_lc(l_l)*m.sin(p.theta_lc(l_l))
    I = -p.m_w*p.d_z*l_l - p.J_wz*l_l*p.d_z/p.r**2 - p.m_l*p.l_lc(l_l)*p.d_z*m.sin(p.theta_lc(l_l))
    J = p.m_w*l_l**2 + p.J_wz*l_l**2/p.r**2 + p.m_l*p.l_lc(l_l)**2 + p.J_llcz_of(l_l)
    K = -p.m_w*l_r - p.J_wz*l_r/p.r**2 - p.m_l*p.l_rc(l_r)*m.sin(p.theta_rc(l_r))
    L = p.m_w*p.d_z*l_r + p.J_wz*l_r*p.d_z/p.r**2 + p.m_l*p.l_rc(l_r)*p.d_z*m.sin(p.theta_rc(l_r))
    M = p.m_w*l_r**2 + p.J_wz*l_r**2/p.r**2 + p.m_l*p.l_rc(l_r)**2 + p.J_rlcz_of(l_r)
    P = -p.m_B*p.l_Bc*m.sin(p.theta_Bc)
    Q = p.m_B*p.l_Bc**2 + p.J_Bz
    R = p.m_l*p.g*l_l + 1/2*p.m_B*p.g*l_l - p.m_l*p.g*p.l_lc(l_l)*m.sin(p.theta_lc(l_l))
    S = p.m_l*p.g*l_r + 1/2*p.m_B*p.g*l_r - p.m_l*p.g*p.l_rc(l_r)*m.sin(p.theta_rc(l_r))
    T = -p.m_B*p.g*p.l_Bc*m.sin(p.theta_Bc)
    U = -1/p.r
    V = -1/p.r
    W = p.d_z/p.r
    X = -p.d_z/p.r
    Y = -(1 + l_l/p.r)
    Z = -(1 + l_r/p.r)

    Martix_1 = np.array([
        [1,0,0,0,0,0,0,0,0,0],
        [0,A,0,0,0,B,0,C,0,D],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,E,0,F,0,G,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,H,0,I,0,J,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0],
        [0,K,0,L,0,0,0,M,0,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,P,0,0,0,0,0,0,0,Q]
    ])

    Martix_2 = np.array([
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,R,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,S,0,0,0],
        [0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,T,0]
    ])

    Martix_3 = np.array([
        [0, 0, 0, 0],
        [U, 0, V, 0],
        [0, 0, 0, 0],
        [W, 0, X, 0],
        [0, 0, 0, 0],
        [Y, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, Z, 1],
        [0, 0, 0, 0],
        [0,-1, 0,-1]
    ])
    Martix_1_inv = np.linalg.inv(Martix_1)
    Martix_A = Martix_1_inv@Martix_2
    Martix_B = Martix_1_inv@Martix_3
    # ---- 权重矩阵 ----
    # 原来的 q / r 是全 0,导致 R 奇异(solve_continuous_are 要求 R 正定),
    # ARE 无解。改为可从外部传入,缺省取 parameter 里按毫米尺度归一化的值。
    if q is None:
        q = p.Q_DEFAULT
    if r is None:
        r = p.R_DEFAULT
    q = list(q)
    r = list(r)
    if len(q) != 10:
        raise ValueError(f"q 需要 10 个元素(状态10维),收到 {len(q)} 个")
    if len(r) != 4:
        raise ValueError(f"r 需要 4 个元素(输入4维),收到 {len(r)} 个")
    if any(v < 0 for v in q) or any(v <= 0 for v in r):
        raise ValueError("q 必须非负,r 必须严格为正(R 必须正定)")

    Martix_Q = np.diag(q)
    Martix_R = np.diag(r)
    Martix_P = solve_continuous_are(Martix_A, Martix_B, Martix_Q, Martix_R)
    Martix_K = np.linalg.inv(Martix_R)@Martix_B.T@Martix_P
    return Martix_K
