import numpy as np

def div(u, NX, NY, NZ):
    """计算散度"""
    divsp = np.zeros((NX, NY, NZ))
    # x方向
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if i == 0:
                    divsp[i, j, k] += u[i+1, j, k] - u[i, j, k]
                elif i == NX-1:
                    divsp[i, j, k] += u[i, j, k] - u[i-1, j, k]
                else:
                    divsp[i, j, k] += (u[i+1, j, k] - u[i-1, j, k]) / 2.0
    # y方向
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if j == 0:
                    divsp[i, j, k] += u[i, j+1, k] - u[i, j, k]
                elif j == NY-1:
                    divsp[i, j, k] += u[i, j, k] - u[i, j-1, k]
                else:
                    divsp[i, j, k] += (u[i, j+1, k] - u[i, j-1, k]) / 2.0
    # z方向
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if k == 0:
                    divsp[i, j, k] += u[i, j, k+1] - u[i, j, k]
                elif k == NZ-1:
                    divsp[i, j, k] += u[i, j, k] - u[i, j, k-1]
                else:
                    divsp[i, j, k] += (u[i, j, k+1] - u[i, j, k-1]) / 2.0
    return divsp

def grad(p, NX, NY, NZ):
    """计算梯度"""
    gradp = np.zeros((NX, NY, NZ, 3))
    # x方向
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if i == 0:
                    gradp[i, j, k, 0] = p[i+1, j, k, 0] - p[i, j, k,0]
                elif i == NX-1:
                    gradp[i, j, k, 0] = p[i, j, k, 0] - p[i-1, j, k, 0]
                else:
                    gradp[i, j, k, 0] = (p[i+1, j, k, 0] - p[i-1, j, k, 0]) / 2.0
    # y方向
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if j == 0:
                    gradp[i, j, k, 0] = p[i, j+1, k, 0] - p[i, j, k, 0]
                elif j == NY-1:
                    gradp[i, j, k, 0] = p[i, j, k, 0] - p[i, j-1, k, 0] 
                else:
                    gradp[i, j, k, 0] = (p[i, j+1, k, 0] - p[i, j-1, k, 0]) / 2.0
    # z方向
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if k == 0:
                    gradp[i, j, k, 0] = p[i, j, k+1, 0] - p[i, j, k, 0]
                elif k == NZ-1:
                    gradp[i, j, k, 0] = p[i, j, k, 0] - p[i, j, k-1, 0]
                else:
                    gradp[i, j, k, 0] = (p[i, j, k+1, 0] - p[i, j, k-1, 0]) / 2.0
    return gradp

def pressure_solve(u, p, NX, NY, NZ):
    """简单的压力求解，使用雅可比迭代"""
    tolerance = 1e-6
    p_new = p.copy()
    max_iter = 10000
    for _ in range(max_iter):
        p_old = p_new.copy()
        # 计算散度
        divsp = div(u, NX, NY, NZ)
        # 用雅可比方法求解压力修正
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    sum_p = 0
                    if i > 0:
                        sum_p += p_old[i-1, j, k]
                    if i < NX-1:
                        sum_p += p_old[i+1, j, k]
                    if j > 0:
                        sum_p += p_old[i, j-1, k]
                    if j < NY-1:
                        sum_p += p_old[i, j+1, k]
                    if k > 0:
                        sum_p += p_old[i, j, k-1]
                    if k < NZ-1:
                        sum_p += p_old[i, j, k+1]
                    p_new[i, j, k] = (divsp[i, j, k] + sum_p) / 6
        # 检查收敛性
        if np.max(np.abs(p_new - p_old)) < tolerance:
            break
    return p_new

def skew_matrix(omega):
    """
    Generate a skew-symmetric matrix from a 3D vector.
    
    Parameters:
    omega (np.array): A 3D vector (omega_x, omega_y, omega_z).
    
    Returns:
    np.array: A 3x3 skew-symmetric matrix.
    """
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def rotate(omega, theta):
    """
    Compute the rotation matrix given a rotation axis and an angle.
    
    Parameters:
    omega (np.array): A 3D vector representing the rotation axis (|omega| = 1).
    theta (float): The rotation angle in radians.
    
    Returns:
    np.array: A 3x3 rotation matrix.
    """
    # Ensure omega is a unit vector
    omega = omega / np.linalg.norm(omega)
    
    # Compute the skew-symmetric matrix
    skew_omega = skew_matrix(omega)
    
    # Compute the rotation matrix
    R = np.eye(3) + np.sin(theta) * skew_omega + (1 - np.cos(theta)) * np.dot(skew_omega, skew_omega)
    
    return R
