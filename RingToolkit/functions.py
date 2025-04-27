import math
import cmath
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
# ----------------------------------------------------------------------------------------------------------------------
def calculate_twiss(m_s: np.ndarray, plane: str = 'x'):
    """Helper method for Twiss parameter calculation per plane"""
    index = 0 if plane == 'x' else 2
    trace = (m_s[index, index] + m_s[index + 1, index + 1]) * 0.5

    if not (-1.0 < trace < 1.0):
        return 0.0, 0.0, 0.0, 0.0, -100

    phi = math.acos(trace)
    sin_phi = math.sin(phi)

    beta_key = (index, index + 1)
    # Check the sign of sin(phi) to ensure the beta is positive.
    sign_check = (sin_phi < 0.0 and m_s[beta_key] < 0.0) or (sin_phi > 0.0 and m_s[beta_key] > 0.0)
    sin_phi = -sin_phi if not sign_check else sin_phi

    beta = m_s[beta_key] / sin_phi
    alpha = (m_s[index, index] - m_s[index + 1, index + 1]) / (2 * sin_phi)
    gamma = -m_s[index + 1, index] / sin_phi

    return beta, alpha, gamma, sin_phi, phi
# ----------------------------------------------------------------------------------------------------------------------
def get_twiss(cell, cell_m66) -> List:
    """Calculate Twiss parameters through the lattice"""
    s = 0.0
    twiss = []
    m = np.eye(6)
    for elem in cell:
        if elem.type == "octupole":
            continue

        for item in elem.slices:
            l = item['l']
            s += l
            m = np.dot(item['matrix'], m)
            inv_m = np.linalg.inv(m)
            m_s = np.linalg.multi_dot([m, cell_m66, inv_m])

            # Calculate parameters for both planes
            beta_x, alpha_x, gamma_x, sin_phi_x, phi_x = calculate_twiss(m_s, plane='x')
            beta_y, alpha_y, gamma_y, sin_phi_y, phi_y = calculate_twiss(m_s, plane='y')

            if -100 in (phi_x, phi_y):
                return []

            # Calculate dispersion
            disp = (m_s[0][4] * (1.0 - m_s[1][1]) + m_s[0][1] * m_s[1][4]) / (2.0 - m_s[0][0] - m_s[1][1])
            d_disp = (m_s[1][4] * (1.0 - m_s[0][0]) + m_s[1][0] * m_s[0][4]) / (2.0 - m_s[0][0] - m_s[1][1])

            k = item.get('k', 0.0)
            # List of twiss parameters within one period.
            twiss.append((
                s,
                {
                    'Beta_X': beta_x, 'Alpha_X': alpha_x, 'Gamma_X': gamma_x,
                    'Beta_Y': beta_y, 'Alpha_Y': alpha_y, 'Gamma_Y': gamma_y,
                    'Disp_X': disp, 'd_Disp_X': d_disp,
                    'name': item['name'],
                    'type': item['type'],
                    'rho': item.get('rho'),
                    'k2' : item.get('k2', 0.0),
                    't1' : item.get('t1', 0.0),
                    't2' : item.get('t2', 0.0),
                    'l': l,
                    'k': k,
                }
            ))
    return twiss
# ----------------------------------------------------------------------------------------------------------------------
def get_parameters(cell_twiss: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    integrals = np.zeros(5)
    chromat_0 = np.zeros(2)
    chromat_1 = np.zeros(2)
    tunes = np.zeros(2)

    def h_function(alpha: float, beta: float, gamma: float, disp: float, d_disp: float) -> float:
        return gamma * disp ** 2 + 2.0 * alpha * disp * d_disp + beta * d_disp ** 2

    for i, (s, curr) in enumerate(cell_twiss):
        prev = cell_twiss[i - 1][1] if i > 0 else curr

        t1 = curr['t1']
        t2 = curr['t2']
        l = curr['l']
        k = curr['k']
        k2 = curr['k2']
        rho = curr['rho'] or 0.0
        r_inv = 1/rho if rho else 0.0

        dx_prev = prev['Disp_X']
        dx_curr = curr['Disp_X']
        bx, by = curr['Beta_X'], curr['Beta_Y']
        ax, gx = curr['Alpha_X'], curr['Gamma_X']
        hx = h_function(ax, bx, gx, dx_curr, curr['d_Disp_X'])

        # Update phase advances
        tunes += l / np.array([bx, by]) / (2.0 * math.pi)

        # Calculate integrals
        integrals += [
                    l * dx_curr * r_inv,
                    l * r_inv ** 2,
                    l * abs(r_inv) ** 3,
                    l * dx_curr * r_inv * (r_inv ** 2 + 2 * k)
                        - dx_prev * np.tan(t1) * r_inv ** 2 - dx_curr * np.tan(t2) * r_inv ** 2,
                    l * hx * r_inv ** 3
        ]
        chromat_0 += [
                    -l * bx * (k - 2 * r_inv ** 2 + 2 * dx_curr * (r_inv ** 3 - k * r_inv)) / (4 * math.pi),
                     l * by * (k - k * dx_curr * r_inv) / (4 * math.pi)
        ]
        chromat_1 += [
                    -l * bx * (k - 2 * r_inv ** 2 + 2 * dx_curr * (r_inv ** 3 - k * r_inv) - k2 * dx_curr) / (4 * math.pi),
                     l * by * (k - k * dx_curr * r_inv - k2 * dx_curr) / (4 * math.pi)
        ]
    return tunes, integrals, chromat_0, chromat_1
# ----------------------------------------------------------------------------------------------------------------------
def get_sexts(lattice, cx1: float = 1.0, cy1: float = 1.0):
    chromatic_contributions = defaultdict(lambda: np.zeros(2))
    for segment in lattice.twiss:
        elem = segment[1]
        if elem['type'] == 'sextupole':
            k2 = 1.0 if elem['k2'] == 0.0 else elem['k2']
            contribution_x = k2 * elem['Disp_X'] * elem['Beta_X'] * elem['l']
            contribution_y = k2 * elem['Disp_X'] * elem['Beta_Y'] * elem['l']
            chromatic_contributions[elem['name']] += [contribution_x, contribution_y]

    sextupoles = [
        mag.family_name
        for mag in lattice.components
        if mag.type == "sextupole"
    ]
    main_sext = []
    for item in lattice.components:
        if item.family_name in sextupoles and item.k == 0.0:
            main_sext.append(item.family_name)
    others = [s for s in sextupoles if s not in main_sext]

    # 计算目标参数
    sum_main = np.array([chromatic_contributions[s] for s in main_sext])
    sum_other = np.array(sum(chromatic_contributions[s] for s in others))
    target_factor = 4 * math.pi / lattice.periodicity
    target_x = (cx1 - lattice.chromat_x0) * target_factor - sum_other[0]
    target_y = -(cy1 - lattice.chromat_y0) * target_factor - sum_other[1]

    # 解线性方程组
    matrix_a = sum_main[:, [0, 1]].T
    if np.linalg.cond(matrix_a) > 1e12:
        return 0.0, 0.0
    try:
        solution = np.linalg.solve(matrix_a, [target_x, target_y])
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    return solution[0], solution[1]
# ----------------------------------------------------------------------------------------------------------------------
def get_driving_terms(cell_twiss: List, periodicity: int) -> Tuple[List, List, Dict]:
    ring_twiss = cell_twiss * periodicity
    s_total = 0.0
    mu_x = mu_y = 0.0
    # mu_x_ = mu_y_ = 0.0

    d_terms_m = []
    d_terms = []
    h_3_order = {key: 0j for key in ['h_21000', 'h_30000', 'h_10110', 'h_10020', 'h_10200',
                                    'h_20001', 'h_00201', 'h_10002', 'h_11001', 'h_00111']}

    # h_4_order = {key: 0j for key in ['h_22000', 'h_31000', 'h_21010', 'h_21100', 'h_11110',
    #                                 'h_11200', 'h_40000', 'h_30010', 'h_30100', 'h_20020',
    #                                 'h_20110', 'h_20200', 'h_10030', 'h_10120', 'h_10210',
    #                                 'h_10300', 'h_00220', 'h_00310', 'h_00400', 'h_21001',
    #                                 'h_11101', 'h_30001', 'h_20011', 'h_20101', 'h_10021',
    #                                 'h_10111', 'h_10201', 'h_00211', 'h_00301', 'h_11002',
    #                                 'h_20002', 'h_10012', 'h_10102', 'h_00112', 'h_00202',
    #                                 'h_10003', 'h_00103', 'h_00004']}
    for i, (s, curr) in enumerate(ring_twiss):
        # prev = ring_twiss[i - 1][1] if i > 0 else curr

        l = curr['l']
        k = curr['k']
        k2 = curr['k2'] / 2
        bx = curr['Beta_X']
        by = curr['Beta_Y']
        dx = curr['Disp_X']
        mu_x += l / bx
        mu_y += l / by

        # l_ = prev['l']
        # k2_ = prev['k2'] / 2
        # bx_ = prev['Beta_X']
        # by_ = prev['Beta_Y']
        # dx_ = prev['Disp_X']
        # mu_x_ += l_ / bx_
        # mu_y_ += l_ / by_

        # Calculate coefficients
        sqrt_bx = math.sqrt(bx)
        # sqrt_bx_ = math.sqrt(bx_)
        third_ord = {
            'h_11001': (k - 2 * k2 * dx) * l * bx / 4,
            'h_00111': -(k - 2 * k2 * dx) * l * by / 4,
            'h_20001': (k - 2 * k2 * dx) * l * bx * cmath.exp(2j * mu_x) / 8,
            'h_00201': -(k - 2 * k2 * dx) * l * by * cmath.exp(2j * mu_y) / 8,
            'h_10002': (k - k2 * dx) * l * dx * sqrt_bx * cmath.exp(1j * mu_x) / 2,

            'h_21000': -k2 * l * sqrt_bx * bx * cmath.exp(1j * mu_x) / 8,
            'h_30000': -k2 * l * sqrt_bx * bx * cmath.exp(3j * mu_x) / 24,
            'h_10110': k2 * l * sqrt_bx * by * cmath.exp(1j * mu_x) / 4,
            'h_10020': k2 * l * sqrt_bx * by * cmath.exp(1j * (mu_x - 2 * mu_y)) / 8,
            'h_10200': k2 * l * sqrt_bx * by * cmath.exp(1j * (mu_x + 2 * mu_y)) / 8,
                }
        # fourth_ord = {
        #     'h_22000': 1j * k2_ * l_ * k2 * l * sqrt_bx_ ** 3 * sqrt_bx ** 3 * (
        #                 cmath.exp(3j * (mu_x_ - mu_x)) + 3 * cmath.exp(1j * (mu_x_ - mu_x))) / 64,
        #     'h_31000': 1j * k2_ * l_ * k2 * l * sqrt_bx_ ** 3 * sqrt_bx ** 3 * cmath.exp(1j * (3 * mu_x_ - mu_x)) / 32,
        #     'h_11110': 1j * k2_ * l_ * k2 * l * sqrt_bx_ * sqrt_bx * by * (
        #                 bx * (cmath.exp(-1j * (mu_x_ - mu_x)) - cmath.exp(1j * (mu_x_ - mu_x)))
        #                 + by * (cmath.exp(1j * (mu_x_ - mu_x + 2 * mu_y_ - 2 * mu_y)) + cmath.exp(1j * (mu_x_ - mu_x))))
        # }
        # Accumulate coefficients
        for key in h_3_order:
            h_3_order[key] += third_ord.get(key, 0j)

        s_total += l
        if i < len(cell_twiss):
            d_terms.append((
                s_total,
                {
                'h_21000': (h_3_order['h_21000'], np.array([1, 0])),
                'h_30000': (h_3_order['h_30000'], np.array([3, 0])),
                'h_10110': (h_3_order['h_10110'], np.array([1, 0])),
                'h_10020': (h_3_order['h_10020'], np.array([1, -2])),
                'h_10200': (h_3_order['h_10200'], np.array([1, 2])),
                'mu': np.array([[mu_x],[mu_y]]), 'l': l})
                        )
        d_terms_m.append((
            s_total,
            {**{k: np.abs(v) for k, v in h_3_order.items()},
             'Beta_X': bx, 'Beta_Y': by, 'Disp_X': dx,
             'mu_x': mu_x, 'mu_y': mu_y}
        ))
    h = {k: np.abs(v) for k, v in h_3_order.items()}
    return d_terms_m, d_terms, h
# ----------------------------------------------------------------------------------------------------------------------
def get_rdts(d_term: List) -> Tuple[Dict, Dict, float]:
    keys = ['h_21000', 'h_30000', 'h_10110', 'h_10020', 'h_10200']
    new_keys = ['f_21000_rms', 'f_30000_rms', 'f_10110_rms', 'f_10020_rms', 'f_10200_rms']
    rdts = {key: [] for key in keys}
    f_rms = {key: 0.0 for key in keys}

    total_terms = len(d_term)
    last_term = d_term[-1]
    mu_vector = last_term[1]['mu']
    s = 0.0
    for i, term in enumerate(d_term):
        s += term[1]['l']
        for key in keys:
            coeff, m = term[1][key]
            fac_0 = np.dot(m, mu_vector).item()
            fac_1 = 1 - cmath.exp(1j * fac_0)
            c_0 = last_term[1][key][0] / fac_1
            c_t = coeff - c_0

            f_rms[key] += (np.abs(c_t) ** 2) / total_terms
            rdts[key].append((
                s,
                {'C_0': np.abs(c_0), 'C_t': np.abs(c_t)}
            ))
    f3_rms = math.sqrt(sum(f_rms.values()))
    f_rms = {new_key: math.sqrt(value) for new_key, value in zip(new_keys, f_rms.values())}
    return rdts, f_rms, f3_rms
# ----------------------------------------------------------------------------------------------------------------------
