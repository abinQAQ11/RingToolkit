import numpy as np
from math import sin, cos, tan, sinh, cosh, sqrt
# ----------------------------------------------------------------------------------------------------------------------
'''直线节的传输矩阵'''
def drift(length):
    d_matrix = np.eye(6)
    d_matrix[0, 1] = length
    d_matrix[2, 3] = length
    return d_matrix

'''六极磁铁的传输矩阵 | 暂不考虑磁场强度，视作直线节'''
def sext(length):
    s_matrix = np.eye(6)
    s_matrix[0, 1] = length
    s_matrix[2, 3] = length
    return s_matrix

"""八极磁铁的传输矩阵 | 暂且视作单位矩阵"""
def oct():
    o_matrix = np.eye(6)
    return o_matrix

'''四极磁铁的传输矩阵'''
def quad(length, strength):
    q_matrix = np.eye(6)
    if strength > 0.0:   # 聚焦四极铁
        sqrt_p = sqrt(strength) # positive
        q_matrix[0, 0] = cos(sqrt_p * length)
        q_matrix[0, 1] = sin(sqrt_p * length) / sqrt_p
        q_matrix[1, 0] = sin(sqrt_p * length) * sqrt_p * (-1.0)
        q_matrix[1, 1] = cos(sqrt_p * length)
        q_matrix[2, 2] = cosh(sqrt_p * length)
        q_matrix[2, 3] = sinh(sqrt_p * length) / sqrt_p
        q_matrix[3, 2] = sinh(sqrt_p * length) * sqrt_p
        q_matrix[3, 3] = cosh(sqrt_p * length)
    elif strength < 0.0: # 散焦四极铁
        sqrt_n = sqrt(abs(strength)) # negative
        q_matrix[0, 0] = cosh(sqrt_n * length)
        q_matrix[0, 1] = sinh(sqrt_n * length) / sqrt_n
        q_matrix[1, 0] = sinh(sqrt_n * length) * sqrt_n
        q_matrix[1, 1] = cosh(sqrt_n * length)
        q_matrix[2, 2] = cos(sqrt_n * length)
        q_matrix[2, 3] = sin(sqrt_n * length) / sqrt_n
        q_matrix[3, 2] = sin(sqrt_n * length) * sqrt_n * (-1.0)
        q_matrix[3, 3] = cos(sqrt_n * length)
    elif strength == 0.0:
        q_matrix[0, 1] = length
        q_matrix[2, 3] = length
    return q_matrix

'''(纯)弯转磁铁的传输矩阵'''
def bend(length, strength, rho):
    k_x = strength + 1 / rho**2
    k_y = strength * (-1.0)
    b_matrix = np.eye(6)
    if k_x > 0.0:
        sqrt_k_x = sqrt(k_x)
        b_matrix[0, 0] = cos(sqrt_k_x * length)
        b_matrix[0, 1] = sin(sqrt_k_x * length) / sqrt_k_x
        b_matrix[0, 4] = (1.0 - cos(sqrt_k_x * length)) / (rho * k_x)
        b_matrix[1, 0] = sin(sqrt_k_x * length) * sqrt_k_x * (-1.0)
        b_matrix[1, 1] = cos(sqrt_k_x * length)
        b_matrix[1, 4] = sin(sqrt_k_x * length) / (rho * sqrt_k_x)
    elif k_x < 0.0:
        k_x = abs(k_x)
        sqrt_k_x = sqrt(k_x)
        b_matrix[0, 0] = cosh(sqrt_k_x * length)
        b_matrix[0, 1] = sinh(sqrt_k_x * length) / sqrt_k_x
        b_matrix[0, 4] = (-1.0 + cosh(sqrt_k_x * length)) / (rho * k_x)
        b_matrix[1, 0] = sinh(sqrt_k_x * length) * sqrt_k_x
        b_matrix[1, 1] = cosh(sqrt_k_x * length)
        b_matrix[1, 4] = sinh(sqrt_k_x * length) / (rho * sqrt_k_x)
    else:
        b_matrix[0, 1] = length

    if k_y > 0.0:
        sqrt_k_y = sqrt(k_y)
        b_matrix[2, 2] = cos(sqrt_k_y * length)
        b_matrix[2, 3] = sin(sqrt_k_y * length) / sqrt_k_y
        b_matrix[3, 2] = sin(sqrt_k_y * length) * sqrt_k_y * (-1.0)
        b_matrix[3, 3] = cos(sqrt_k_y * length)
    elif k_y < 0.0:
        k_y = abs(k_y)
        sqrt_k_y = sqrt(k_y)
        b_matrix[2, 2] = cosh(sqrt_k_y * length)
        b_matrix[2, 3] = sinh(sqrt_k_y * length) / sqrt_k_y
        b_matrix[3, 2] = sinh(sqrt_k_y * length) * sqrt_k_y
        b_matrix[3, 3] = cosh(sqrt_k_y * length)
    else:
        b_matrix[2, 3] = length
    return b_matrix

'''矩形弯铁的边缘矩阵'''
def edge(rho, entrance_angle, exit_angle):
    e_matrix = np.eye(6)
    if exit_angle == 0.0:
        e_matrix[1, 0] = tan(entrance_angle) / rho
        e_matrix[3, 2] = tan(entrance_angle) / rho * (-1.0)
    elif entrance_angle == 0.0:
        e_matrix[1, 0] = tan(exit_angle) / rho
        e_matrix[3, 2] = tan(exit_angle) / rho * (-1.0)
    elif entrance_angle == exit_angle == 0.0:
        e_matrix[1, 0] = tan(entrance_angle) / rho
        e_matrix[3, 2] = tan(exit_angle) / rho * (-1.0)
    return e_matrix