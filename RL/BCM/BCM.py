# RL/BCM/BCM.py
import numpy as np

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class BCM:
    def __init__(self, num_assets, transaction_cost=0.001):
        self.num_assets = num_assets          # sin cash
        self.tc = transaction_cost

    def greedy_action(self, price_rel_full, w_prev):
        """
        Solve:
        max_w  u^T w - c * sum |w - w_prev|
        s.t.   sum w = 1, 0<=w<=1
        price_rel_full: (m+1,) incluye cash = 1.0
        w_prev: (m+1,) pesos previos
        """
        if not _HAS_SCIPY:
            # fallback simple: todo al mejor activo neto (heurística)
            best = int(np.argmax(price_rel_full))
            w = np.zeros_like(w_prev)
            w[best] = 1.0
            return w

        m = len(price_rel_full)
        c = self.tc

        # Variables: w (m) y z (m) con z >= |w - w_prev|
        # Minimize: -u^T w + c * sum z
        c_obj = np.concatenate([-price_rel_full, c * np.ones(m)])

        # Constraints:
        # 1) sum w = 1
        A_eq = np.concatenate([np.ones(m), np.zeros(m)])[None, :]
        b_eq = np.array([1.0])

        # 2) w - w_prev <= z
        # 3) -(w - w_prev) <= z
        I = np.eye(m)
        A_ub = np.vstack([
            np.hstack([ I, -I ]),
            np.hstack([-I, -I ])
        ])
        b_ub = np.concatenate([w_prev, -w_prev])

        # Bounds: 0<=w<=1, z>=0
        bounds = [(0.0, 1.0)] * m + [(0.0, None)] * m

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            # fallback: conserva pesos previos
            return w_prev.copy()

        w = res.x[:m]
        return w

    @staticmethod
    def bcm_loss(mu, a_greedy, eps=1e-8):
        """
        Binary cross-entropy tipo Eq(2) sobre cada componente.
        mu y a_greedy en [0,1], shape (batch, m+1)
        """
        mu = np.clip(mu, eps, 1 - eps)
        a_greedy = np.clip(a_greedy, eps, 1 - eps)
        loss = -(a_greedy * np.log(mu) + (1 - a_greedy) * np.log(1 - mu))
        return loss.mean()