"""
Medical Center Server — AHP-VIKOR Hospital Selection Module

Implements Phases 1 & 2 of the paper's methodology:

  Phase 1  —  Decision Matrix (DM) for healthcare service identification
               P = wF·F + wR·R + wI·I   (feasibility, relevance, impact)

  Phase 2  —  AHP to derive criterion weights
               VIKOR to rank hospitals on those weighted criteria

Reference: Section III.2.1 of the paper.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Saaty's random consistency index (RI) for n=1..10 ───────────────────────
_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
       6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ── Phase 1: Decision Matrix ─────────────────────────────────────────────────

class DecisionMatrix:
    """
    Evaluates healthcare services using three criteria:
      F = Feasibility  |  R = Relevance  |  I = Impact
    """

    def __init__(self,
                 w_feasibility: float = 0.3,
                 w_relevance: float   = 0.4,
                 w_impact: float      = 0.3):
        assert abs(w_feasibility + w_relevance + w_impact - 1.0) < 1e-6, \
            "Weights must sum to 1"
        self.wF = w_feasibility
        self.wR = w_relevance
        self.wI = w_impact

    def evaluate(self, services: list[dict]) -> list[dict]:
        """
        services: list of dicts with keys 'name', 'feasibility', 'relevance', 'impact'
                  (all scored 1-10).
        Returns sorted list with priority scores.
        """
        results = []
        for svc in services:
            P = (self.wF * svc["feasibility"] +
                 self.wR * svc["relevance"] +
                 self.wI * svc["impact"])
            results.append({**svc, "priority_score": round(P, 4)})
        results.sort(key=lambda x: x["priority_score"], reverse=True)
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
        return results


# ── Phase 2: AHP ─────────────────────────────────────────────────────────────

class AHP:
    """
    Analytic Hierarchy Process — derives criterion weights from a pairwise
    comparison matrix and checks consistency.

    Steps (per the paper):
      1. Build pairwise matrix
      2. Normalise columns
      3. Compute priority vector (row means of normalised matrix)
      4. Check Consistency Ratio (CR < 0.10 is acceptable)
    """

    def __init__(self, pairwise_matrix: np.ndarray, criteria: list[str]):
        n = pairwise_matrix.shape[0]
        assert pairwise_matrix.shape == (n, n), "Matrix must be square"
        assert len(criteria) == n, "Criteria list must match matrix size"
        self.A = pairwise_matrix.astype(float)
        self.criteria = criteria
        self.n = n
        self._weights: Optional[np.ndarray] = None
        self._cr: Optional[float] = None

    def compute(self) -> dict:
        # Step 1 — column sums
        col_sums = self.A.sum(axis=0)

        # Step 2 — normalise
        A_norm = self.A / col_sums

        # Step 3 — priority vector
        weights = A_norm.mean(axis=1)
        self._weights = weights

        # Step 4 — consistency check
        Aw = self.A @ weights
        lambda_max = float(np.mean(Aw / weights))
        CI = (lambda_max - self.n) / (self.n - 1) if self.n > 1 else 0.0
        RI = _RI.get(self.n, 1.49)
        CR = CI / RI if RI > 0 else 0.0
        self._cr = CR

        if CR > 0.10:
            logger.warning("AHP CR=%.3f > 0.10 — pairwise judgements may be inconsistent", CR)
        else:
            logger.info("AHP CR=%.3f — consistent", CR)

        return {
            "criteria": self.criteria,
            "weights": {c: round(float(w), 6) for c, w in zip(self.criteria, weights)},
            "lambda_max": round(lambda_max, 6),
            "consistency_index": round(CI, 6),
            "consistency_ratio": round(CR, 6),
            "consistent": CR <= 0.10,
        }

    @property
    def weights(self) -> np.ndarray:
        if self._weights is None:
            self.compute()
        return self._weights


# ── Phase 2: VIKOR ───────────────────────────────────────────────────────────

class VIKOR:
    """
    VIKOR multi-criteria ranking of hospitals.

    Parameters
    ----------
    decision_matrix : np.ndarray  shape (n_hospitals, n_criteria)
    weights         : np.ndarray  shape (n_criteria,)  — from AHP
    benefit_criteria: list[bool]  True = higher is better
    v               : float       weight of group utility (default 0.5)
    """

    def __init__(self,
                 decision_matrix: np.ndarray,
                 weights: np.ndarray,
                 benefit_criteria: list[bool],
                 hospitals: list[str],
                 criteria: list[str],
                 v: float = 0.5):
        self.X = decision_matrix.astype(float)
        self.w = weights.astype(float)
        self.benefit = benefit_criteria
        self.hospitals = hospitals
        self.criteria = criteria
        self.v = v

    def rank(self) -> dict:
        n_h, n_c = self.X.shape

        # Best and worst values for each criterion
        f_best = np.array([
            self.X[:, j].max() if self.benefit[j] else self.X[:, j].min()
            for j in range(n_c)
        ])
        f_worst = np.array([
            self.X[:, j].min() if self.benefit[j] else self.X[:, j].max()
            for j in range(n_c)
        ])

        # Normalised weighted distances
        denom = f_best - f_worst
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        norm_dist = self.w * (f_best - self.X) / denom

        # S = utility measure, R = regret measure
        S = norm_dist.sum(axis=1)
        R = norm_dist.max(axis=1)

        S_min, S_max = S.min(), S.max()
        R_min, R_max = R.min(), R.max()

        s_denom = S_max - S_min if (S_max - S_min) > 1e-12 else 1e-12
        r_denom = R_max - R_min if (R_max - R_min) > 1e-12 else 1e-12

        Q = self.v * (S - S_min) / s_denom + (1 - self.v) * (R - R_min) / r_denom

        ranked_idx = np.argsort(Q)
        results = []
        for rank, idx in enumerate(ranked_idx, 1):
            results.append({
                "rank":     rank,
                "hospital": self.hospitals[idx],
                "Q":  round(float(Q[idx]),  6),
                "S":  round(float(S[idx]),  6),
                "R":  round(float(R[idx]),  6),
            })

        return {
            "rankings": results,
            "hospitals": self.hospitals,
            "criteria":  self.criteria,
        }


# ── High-level convenience function ─────────────────────────────────────────

def run_hospital_selection(hospitals: list[dict], criteria: list[str],
                           pairwise_matrix: np.ndarray,
                           benefit_criteria: list[bool]) -> dict:
    """
    Full AHP-VIKOR pipeline.

    hospitals: list of dicts  {'name': str, 'scores': [float, …]}
    criteria : list of criterion names (len == pairwise_matrix.shape[0])
    """
    # AHP weights
    ahp = AHP(pairwise_matrix, criteria)
    ahp_result = ahp.compute()

    # Build decision matrix
    dm = np.array([h["scores"] for h in hospitals])
    hospital_names = [h["name"] for h in hospitals]

    # VIKOR ranking
    vikor = VIKOR(dm, ahp.weights, benefit_criteria, hospital_names, criteria)
    vikor_result = vikor.rank()

    return {
        "ahp":   ahp_result,
        "vikor": vikor_result,
    }


# ── Demo / default scenario ──────────────────────────────────────────────────

def default_scenario() -> dict:
    """
    Example scenario used to demonstrate the hospital selection framework.
    5 hospitals evaluated on 5 criteria derived from the paper.
    """
    hospitals = [
        {"name": "City General Hospital",      "scores": [8.5, 7.0, 9.0, 6.5, 8.0]},
        {"name": "Metro Medical Center",       "scores": [7.0, 9.0, 7.5, 8.0, 7.5]},
        {"name": "Regional Health Institute",  "scores": [6.5, 7.5, 8.0, 9.0, 6.0]},
        {"name": "St. Luke's Hospital",        "scores": [9.0, 6.5, 7.0, 7.5, 9.0]},
        {"name": "Community Care Hospital",    "scores": [7.5, 8.0, 6.5, 8.5, 7.0]},
    ]

    criteria = [
        "ICU Capacity",
        "Telemedicine Capability",
        "Response Time (min)",   # lower is better
        "EHR Integration",
        "Specialist Availability",
    ]

    # Saaty pairwise matrix — criteria importance
    pairwise = np.array([
        [1,    2,    3,    2,    2  ],
        [1/2,  1,    2,    1,    1  ],
        [1/3,  1/2,  1,    1/2,  1/2],
        [1/2,  1,    2,    1,    1  ],
        [1/2,  1,    2,    1,    1  ],
    ])

    benefit_criteria = [True, True, False, True, True]

    return run_hospital_selection(hospitals, criteria, pairwise, benefit_criteria)
