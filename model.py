from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import PoissonRegressor, TweedieRegressor
except ImportError:
    PoissonRegressor = None
    TweedieRegressor = None
def _rankdata_numpy(a: np.ndarray) -> np.ndarray:
    """Computes fractional (average) ranks in pure NumPy, identical to scipy.stats.rankdata."""
    a_arr = np.asarray(a)
    n = len(a_arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    sorter = np.argsort(a_arr)
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)
    a_sorted = a_arr[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], n]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _numpy_spearmanr(a: Any, b: Any) -> Any:
    """Pure NumPy Spearman rank correlation fallback when scipy is not installed."""
    class _Result:
        def __init__(self, stat: float):
            self.statistic = float(stat)
            self.correlation = float(stat)

    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 2 or len(b_arr) < 2 or len(a_arr) != len(b_arr):
        return _Result(0.0)
    ra = _rankdata_numpy(a_arr)
    rb = _rankdata_numpy(b_arr)
    std_a = np.std(ra)
    std_b = np.std(rb)
    if std_a == 0.0 or std_b == 0.0:
        return _Result(0.0)
    c = np.corrcoef(ra, rb)[0, 1]
    if np.isnan(c):
        return _Result(0.0)
    return _Result(float(c))


try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = _numpy_spearmanr


def parse_toi(toi_str: str) -> float:
    if not toi_str or not isinstance(toi_str, str):
        return 0.0
    parts = toi_str.split(":")
    if len(parts) == 2:
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except ValueError:
            return 0.0
    return 0.0


class RidgeRegression:
    def __init__(self, alpha: float = 12.0):
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Any = 0.0
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0

        Z = (X - self.mean_) / self.scale_
        y_mean = np.mean(y, axis=0)
        y_centered = y - y_mean

        A = Z.T @ Z + self.alpha * np.eye(p)
        b = Z.T @ y_centered
        try:
            self.coef_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(A) @ b

        self.intercept_ = y_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.scale_
        return Z @ self.coef_ + self.intercept_


POS_AVERAGES: Dict[str, Dict[str, float]] = {
    "D": {"goals": 0.09, "assists": 0.28, "shots": 1.5, "toi": 19.5},
    "F": {"goals": 0.22, "assists": 0.32, "shots": 1.9, "toi": 15.5},
}

MAX_RRS = 16.0


def compute_line_factor(
    ev_line: Optional[str] = None,
    pp_unit: Optional[str] = None,
    position: str = "F",
    team_has_pp: bool = True,
    opp_pk_pct: Optional[float] = None,
    opp_times_shorthanded: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Computes (goal_mult, assist_mult, toi_adj) based on EV line deployment, PP unit,
    and opponent penalty kill & discipline metrics.
    - PP1: +12% goals, +18% assists, +1.5 min TOI (modulated by opponent PK vulnerability and penalty frequency)
    - PP2: +4% goals, +5% assists, +0.5 min TOI (modulated by opponent PK vulnerability and penalty frequency)
    - No PP (when EV line is known and team has PP data): -8% goals, -12% assists, -0.5 min TOI
    - EV deployment:
      - L1: +5% goals, +5% assists, +1.0 min TOI
      - L2: +2% goals, +2% assists, 0.0 min TOI
      - L3: -5% goals, -5% assists, -1.0 min TOI
      - L4: -12% goals, -15% assists, -2.5 min TOI
      - D1: +4% assists, +1.5 min TOI
      - D2: 0.0
      - D3: -8% assists, -2.5 min TOI
      - D4: -15% assists, -4.0 min TOI
    """
    if not ev_line and not pp_unit:
        return 1.0, 1.0, 0.0

    g_mult = 1.0
    a_mult = 1.0
    toi_adj = 0.0

    # Opponent Special Teams Matchup Modulation (PK quality and penalty frequency)
    if opp_pk_pct is not None or opp_times_shorthanded is not None:
        pk = float(opp_pk_pct if opp_pk_pct is not None else 0.80)
        ts = float(opp_times_shorthanded if opp_times_shorthanded is not None else 3.0)
        pk_vuln = (1.0 - pk) / 0.20
        pen_freq = ts / 3.0
        pp_matchup = float(np.clip(0.6 * pk_vuln + 0.4 * pen_freq, 0.70, 1.40))
        toi_scale = float(np.clip(pen_freq, 0.75, 1.35))
    else:
        pp_matchup = 1.0
        toi_scale = 1.0

    # PP Adjustment
    if pp_unit == "PP1":
        g_mult += 0.12 * pp_matchup
        a_mult += 0.18 * pp_matchup
        toi_adj += 1.5 * toi_scale
    elif pp_unit == "PP2":
        g_mult += 0.04 * pp_matchup
        a_mult += 0.05 * pp_matchup
        toi_adj += 0.5 * toi_scale
    elif pp_unit is None and ev_line and team_has_pp:
        g_mult -= 0.08
        a_mult -= 0.12
        toi_adj -= 0.5

    # EV Line Adjustment
    if ev_line:
        ev = ev_line.upper()
        if ev == "L1":
            g_mult += 0.05
            a_mult += 0.05
            toi_adj += 1.0
        elif ev == "L2":
            g_mult += 0.02
            a_mult += 0.02
        elif ev == "L3":
            g_mult -= 0.05
            a_mult -= 0.05
            toi_adj -= 1.0
        elif ev == "L4":
            g_mult -= 0.12
            a_mult -= 0.15
            toi_adj -= 2.5
        elif ev == "D1":
            a_mult += 0.04
            toi_adj += 1.5
        elif ev == "D2":
            pass
        elif ev == "D3":
            a_mult -= 0.08
            toi_adj -= 2.5
        elif ev == "D4":
            a_mult -= 0.15
            toi_adj -= 4.0

    return max(0.6, round(g_mult, 3)), max(0.5, round(a_mult, 3)), round(toi_adj, 2)


class PerformancePredictor:
    def __init__(self, model_type: str = "poisson", scoring_weights: Optional[Dict[str, float]] = None):
        self.model_type = str(model_type or "poisson").lower()
        self.scoring_weights = scoring_weights
        self.multi_model: Optional[RidgeRegression] = None
        self.glm_models: Dict[str, Any] = {}
        self.trained = False
        self.baseline: Dict[str, float] = {}

    @property
    def is_trained(self) -> bool:
        return self.trained

    def _extract_dataset(
        self,
        game_logs: List[Dict[str, Any]],
        standings: Dict[str, Dict[str, float]]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        if len(game_logs) < 6:
            return (
                np.empty((0, 14)),
                {k: np.empty(0) for k in ("goals", "assists", "shots", "toi")},
                np.empty(0),
            )

        parsed = []
        for g in game_logs:
            parsed.append({
                "date": g.get("gameDate", "2020-01-01"),
                "goals": float(g.get("goals", 0)),
                "assists": float(g.get("assists", 0)),
                "shots": float(g.get("shots", 0)),
                "toi": parse_toi(g.get("toi", "00:00")),
                "is_home": 1.0 if (g.get("homeRoadFlag") == "H" or g.get("is_home") in (1, True)) else 0.0,
                "opp": g.get("opponentAbbrev", "")
            })

        X_rows = []
        y_dict = {"goals": [], "assists": [], "shots": [], "toi": []}
        exposure_weights = []

        for i in range(5, len(parsed)):
            curr = parsed[i]
            exposure_hours = curr["toi"] / 60.0
            if exposure_hours <= 0.0:
                continue

            opp_info = standings.get(curr["opp"], {})
            opp_def = float(opp_info.get("defense_factor", 1.0))

            last5 = parsed[max(0, i - 5):i]
            last20 = parsed[max(0, i - 20):i]
            career = parsed[:i]

            feat = [
                np.mean([x["goals"] for x in last5]),
                np.mean([x["assists"] for x in last5]),
                np.mean([x["shots"] for x in last5]),
                np.mean([x["toi"] for x in last5]),
                np.mean([x["goals"] for x in last20]),
                np.mean([x["assists"] for x in last20]),
                np.mean([x["shots"] for x in last20]),
                np.mean([x["toi"] for x in last20]),
                np.mean([x["goals"] for x in career]),
                np.mean([x["assists"] for x in career]),
                np.mean([x["shots"] for x in career]),
                np.mean([x["toi"] for x in career]),
                curr["is_home"],
                opp_def
            ]

            X_rows.append(feat)
            y_dict["goals"].append(curr["goals"])
            y_dict["assists"].append(curr["assists"])
            y_dict["shots"].append(curr["shots"])
            y_dict["toi"].append(curr["toi"])
            exposure_weights.append(exposure_hours)

        X = np.array(X_rows, dtype=np.float64)
        targets = {k: np.array(v, dtype=np.float64) for k, v in y_dict.items()}
        exposure = np.array(exposure_weights, dtype=np.float64)
        return X, targets, exposure

    def train(self, game_logs: List[Dict[str, Any]], standings: Optional[Dict[str, Dict[str, float]]] = None):
        st = standings or {}
        X, y_dict, exposure = self._extract_dataset(game_logs, st)

        if game_logs:
            parsed_tois = [parse_toi(g.get("toi", "00:00")) for g in game_logs]
            self.baseline = {
                "goals": float(np.mean([float(g.get("goals", 0)) for g in game_logs])),
                "assists": float(np.mean([float(g.get("assists", 0)) for g in game_logs])),
                "shots": float(np.mean([float(g.get("shots", 0)) for g in game_logs])),
                "toi": float(np.mean(parsed_tois)) if parsed_tois else 15.0
            }
        else:
            self.baseline = {"goals": 0.2, "assists": 0.3, "shots": 2.0, "toi": 15.0}

        if len(X) < 10:
            self.trained = False
            return

        # Train GLM models (Poisson / Tweedie) for right-skewed count outcomes
        if self.model_type in ("poisson", "tweedie") and PoissonRegressor is not None:
            self.glm_models = {}
            mean_X = np.mean(X, axis=0)
            std_X = np.std(X, axis=0)
            std_X[std_X == 0.0] = 1.0
            Z = (X - mean_X) / std_X
            rate_targets = {
                name: y_dict[name] / exposure
                for name in ("goals", "assists", "shots")
            }

            # If sample size N < 25, use stronger regularization alpha=3.0, else 1.5
            glm_alpha = 3.0 if len(X) < 25 else 1.5
            ridge_toi_alpha = 25.0 if len(X) < 25 else 15.0

            try:
                # Rate models with log-link and time-on-ice exposure
                g_model = PoissonRegressor(alpha=glm_alpha, max_iter=300)
                g_model.fit(Z, rate_targets["goals"], sample_weight=exposure)
                self.glm_models["goals"] = (g_model, mean_X, std_X)

                a_model = PoissonRegressor(alpha=glm_alpha, max_iter=300)
                a_model.fit(Z, rate_targets["assists"], sample_weight=exposure)
                self.glm_models["assists"] = (a_model, mean_X, std_X)

                # Shots: Poisson or Tweedie (compound Poisson)
                if self.model_type == "tweedie" and TweedieRegressor is not None:
                    s_model = TweedieRegressor(power=1.5, alpha=glm_alpha, max_iter=300)
                else:
                    s_model = PoissonRegressor(alpha=glm_alpha, max_iter=300)
                s_model.fit(Z, rate_targets["shots"], sample_weight=exposure)
                self.glm_models["shots"] = (s_model, mean_X, std_X)

                # TOI: Gaussian linear / Ridge regression
                toi_model = RidgeRegression(alpha=ridge_toi_alpha)
                toi_model.fit(X, y_dict["toi"])
                self.glm_models["toi"] = toi_model

                self.trained = True
                return
            except Exception:
                self.glm_models = {}

        # Fallback / Classic: Multi-output Ridge regression
        y_mat = np.column_stack([y_dict[target] for target in ("goals", "assists", "shots", "toi")])
        ridge_alpha = 25.0 if len(X) < 25 else 15.0
        self.multi_model = RidgeRegression(alpha=ridge_alpha)
        self.multi_model.fit(X, y_mat)
        self.trained = True

    def predict(
        self,
        game_logs: List[Dict[str, Any]],
        is_home: bool,
        opponent_abbrev: str,
        game_date: str,
        standings: Optional[Dict[str, Dict[str, float]]] = None,
        position: str = "F",
        goalie_factor: float = 1.0,
        ev_line: Optional[str] = None,
        pp_unit: Optional[str] = None,
        team_has_pp: bool = True,
        is_b2b: bool = False,
        rest_days: Optional[float] = None,
        opp_is_b2b: bool = False,
        opp_pk_pct: Optional[float] = None,
        opp_times_shorthanded: Optional[float] = None
    ) -> Dict[str, Any]:
        pos_key = "D" if str(position).upper() in ("D", "DEF") else "F"
        pos_avg = POS_AVERAGES[pos_key]
        opp_info = (standings or {}).get(opponent_abbrev, {})
        opp_def = float(opp_info.get("defense_factor", 1.0))
        pk_val = float(opp_pk_pct if opp_pk_pct is not None else opp_info.get("pk_pct", 0.80))
        ts_val = float(opp_times_shorthanded if opp_times_shorthanded is not None else opp_info.get("times_shorthanded_per_game", 3.0))

        home_boost = 1.05 if is_home else 0.95
        effective_b2b = bool(is_b2b) or (rest_days is not None and float(rest_days) <= 1.0)
        fatigue_mult = 0.96 if effective_b2b else 1.0
        if opp_is_b2b:
            fatigue_mult *= 1.03

        if not game_logs:
            preds = {
                "goals": max(0.0, pos_avg["goals"] * opp_def * home_boost * fatigue_mult),
                "assists": max(0.0, pos_avg["assists"] * opp_def * home_boost * fatigue_mult),
                "shots": max(0.0, pos_avg["shots"] * opp_def * home_boost * fatigue_mult),
                "toi": max(5.0, pos_avg["toi"])
            }
            if goalie_factor != 1.0:
                preds["goals"] = max(0.0, preds["goals"] * goalie_factor)
                preds["assists"] = max(0.0, preds["assists"] * (1.0 + 0.5 * (goalie_factor - 1.0)))
        else:
            parsed = [{
                "date": g.get("gameDate", "2020-01-01"),
                "goals": float(g.get("goals", 0)),
                "assists": float(g.get("assists", 0)),
                "shots": float(g.get("shots", 0)),
                "toi": parse_toi(g.get("toi", "00:00"))
            } for g in game_logs]

            last5 = parsed[-5:] if len(parsed) >= 5 else parsed
            last20 = parsed[-20:] if len(parsed) >= 20 else parsed

            x_vec = np.array([[
                np.mean([x["goals"] for x in last5]),
                np.mean([x["assists"] for x in last5]),
                np.mean([x["shots"] for x in last5]),
                np.mean([x["toi"] for x in last5]),
                np.mean([x["goals"] for x in last20]),
                np.mean([x["assists"] for x in last20]),
                np.mean([x["shots"] for x in last20]),
                np.mean([x["toi"] for x in last20]),
                np.mean([x["goals"] for x in parsed]),
                np.mean([x["assists"] for x in parsed]),
                np.mean([x["shots"] for x in parsed]),
                np.mean([x["toi"] for x in parsed]),
                1.0 if is_home else 0.0,
                opp_def
            ]], dtype=np.float64)

            preds = {}
            if self.trained and self.glm_models:
                try:
                    g_mod, mean_X, std_X = self.glm_models["goals"]
                    z_vec = (x_vec - mean_X) / std_X
                    raw_g_rate = float(g_mod.predict(z_vec)[0])

                    a_mod, _, _ = self.glm_models["assists"]
                    raw_a_rate = float(a_mod.predict(z_vec)[0])

                    s_mod, _, _ = self.glm_models["shots"]
                    raw_s_rate = float(s_mod.predict(z_vec)[0])

                    toi_mod = self.glm_models["toi"]
                    # toi_mod is Ridge, expecting original X
                    raw_toi = float(toi_mod.predict(x_vec)[0])

                    exposure = max(raw_toi, 5.0) / 60.0
                    raw_g = raw_g_rate * exposure
                    raw_a = raw_a_rate * exposure
                    raw_s = raw_s_rate * exposure

                    preds["goals"] = max(0.0, raw_g * fatigue_mult)
                    preds["assists"] = max(0.0, raw_a * fatigue_mult)
                    preds["shots"] = max(0.0, raw_s * fatigue_mult)
                    preds["toi"] = max(0.0, raw_toi)
                except Exception:
                    preds = {}

            if not preds and self.trained and self.multi_model is not None:
                # Multi-output Ridge prediction
                coef = getattr(self.multi_model, "coef_", None)
                p_dim = coef.shape[0] if coef is not None else x_vec.shape[1]
                in_vec = x_vec[:, :p_dim] if isinstance(p_dim, int) and p_dim < x_vec.shape[1] else x_vec
                raw_preds = self.multi_model.predict(in_vec)[0]
                preds["goals"] = max(0.0, float(raw_preds[0]) * fatigue_mult)
                preds["assists"] = max(0.0, float(raw_preds[1]) * fatigue_mult)
                preds["shots"] = max(0.0, float(raw_preds[2]) * fatigue_mult)
                preds["toi"] = max(0.0, float(raw_preds[3]))

            # Baseline calculation for fallback or shrinkage blending
            gp = len(parsed)
            k = 10.0
            r5_g = np.mean([x["goals"] for x in last5])
            r5_a = np.mean([x["assists"] for x in last5])
            r5_s = np.mean([x["shots"] for x in last5])
            r5_toi = np.mean([x["toi"] for x in last5])

            if gp < 10:
                adj_g = (r5_g * gp + pos_avg["goals"] * k) / (gp + k)
                adj_a = (r5_a * gp + pos_avg["assists"] * k) / (gp + k)
                adj_s = (r5_s * gp + pos_avg["shots"] * k) / (gp + k)
                adj_toi = (r5_toi * gp + pos_avg["toi"] * k) / (gp + k)
            else:
                adj_g, adj_a, adj_s, adj_toi = r5_g, r5_a, r5_s, r5_toi

            base_g = max(0.0, adj_g * fatigue_mult)
            base_a = max(0.0, adj_a * fatigue_mult)
            base_s = max(0.0, adj_s * fatigue_mult)
            base_toi = max(5.0, adj_toi)

            if preds:
                # Small-sample Bayesian shrinkage towards baseline (N < 35 games)
                if gp < 35:
                    w = gp / (gp + 15.0)
                    preds["goals"] = w * preds["goals"] + (1.0 - w) * base_g
                    preds["assists"] = w * preds["assists"] + (1.0 - w) * base_a
                    preds["shots"] = w * preds["shots"] + (1.0 - w) * base_s
                    preds["toi"] = w * preds["toi"] + (1.0 - w) * base_toi
            else:
                preds = {
                    "goals": base_g,
                    "assists": base_a,
                    "shots": base_s,
                    "toi": base_toi
                }

            if goalie_factor != 1.0:
                preds["goals"] = max(0.0, preds["goals"] * goalie_factor)
                preds["assists"] = max(0.0, preds["assists"] * (1.0 + 0.5 * (goalie_factor - 1.0)))

        # Line combination & power play deployment adjustment (including opponent PK/discipline)
        g_line_mult, a_line_mult, toi_line_adj = compute_line_factor(
            ev_line, pp_unit, position=position, team_has_pp=team_has_pp,
            opp_pk_pct=pk_val, opp_times_shorthanded=ts_val
        )
        if (g_line_mult, a_line_mult, toi_line_adj) != (1.0, 1.0, 0.0):
            preds["goals"] = max(0.0, preds["goals"] * g_line_mult)
            preds["assists"] = max(0.0, preds["assists"] * a_line_mult)
            preds["toi"] = max(5.0, preds["toi"] + toi_line_adj)

        # Enforce physiological caps on single-game expectations
        preds["goals"] = float(np.clip(preds["goals"], 0.0, 1.10))
        preds["assists"] = float(np.clip(preds["assists"], 0.0, 1.70))
        preds["shots"] = float(np.clip(preds["shots"], 0.0, 6.50))
        preds["toi"] = float(np.clip(preds["toi"], 5.0, 28.0))

        is_def = position in ("D", "DEF")
        blk_rate = 0.09 if is_def else 0.04
        blocked = round(preds["toi"] * blk_rate, 2)
        points = round(preds["goals"] + preds["assists"], 2)

        toi_min, toi_sec = divmod(round(preds["toi"] * 60), 60)
        formatted_toi = f"{toi_min:02d}:{toi_sec:02d}"

        # RRS (Real Rating System) calibrated projection:
        rrs_base = calculate_xrrs_projection(
            xg=preds["goals"],
            xa=preds["assists"],
            xsog=preds["shots"],
            xblk=blocked,
            is_def=is_def,
            weights=self.scoring_weights
        )
        rrs_base = min(MAX_RRS, rrs_base)

        return {
            "xG": round(preds["goals"], 2),
            "xA": round(preds["assists"], 2),
            "xPTS": points,
            "xSOG": round(preds["shots"], 2),
            "xBLK": blocked,
            "xTOI": formatted_toi,
            "xRRS": rrs_base,
            "base_pts": rrs_base,
            "ev_line": ev_line,
            "pp_unit": pp_unit
        }


def calculate_xrrs_projection(
    xg: float,
    xa: float,
    xsog: float,
    xblk: float,
    is_def: bool = False,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Computes expected Real Rating (xRRS) calibrated to the Real Sports mobile app.
    When calibrated weights are supplied, the score is a linear combination of
    goals, assists, shots, and blocks with the fitted intercept.
    """
    if weights:
        xrrs = (
            float(weights.get("intercept", 0.0))
            + xg * float(weights.get("goals", 2.50))
            + xa * float(weights.get("assists", 1.05))
            + xsog * float(weights.get("shots", 0.25))
            + xblk * float(weights.get("blocks", 0.50))
        )
        return round(float(np.clip(xrrs, 0.0, MAX_RRS)), 2)

    xpts = xg + xa
    periph = xsog * 0.25 + xblk * 0.50
    scoring_ev = (xg * 2.50) + (xa * 1.05)

    # Multi-point upside probability (Poisson model)
    p_multi = max(0.0, 1.0 - np.exp(-xpts) - xpts * np.exp(-xpts))
    multi_bonus = p_multi * 1.4

    xrrs = scoring_ev + multi_bonus + periph

    # Cap players with negligible scoring upside to prevent artificial booster inflation
    if is_def and xg < 0.08 and xa < 0.20:
        xrrs = min(0.85, xrrs)
    elif not is_def and xg < 0.10 and xa < 0.15:
        xrrs = min(0.95, xrrs)

    return round(float(np.clip(xrrs, 0.0, MAX_RRS)), 2)


def calculate_rrs_from_stats(
    goals: float,
    assists: float,
    shots: float,
    blocks: float
) -> float:
    """
    Computes actual Real Rating based on completed match boxscore events:
    - 0 points (only shots/blocks): 0.0 - 1.0 (shots * 0.25 + blocks * 0.50).
    - 1 Assist: ~1.0 - 1.3.
    - 1 Goal: ~2.5 - 3.5.
    - Multi-point games (2+ points): ~3.7 - 5.2.
    - Star multi-point performances: 8.0+.
    """
    pts = goals + assists
    if pts == 0:
        raw = shots * 0.25 + blocks * 0.50
        return round(min(1.0, max(0.0, raw)), 1)

    base = 0.0
    if goals >= 1:
        base += 2.5
    if goals >= 2:
        base += 1.8 * (goals - 1)

    if assists >= 1:
        base += 1.0
    if assists >= 2:
        base += 0.9 * (assists - 1)

    if pts >= 2:
        base += 0.4 * (pts - 1)

    base += shots * 0.25 + blocks * 0.50
    return round(min(MAX_RRS, max(0.0, base)), 1)


def calibrate_rrs_formula(db: Any, allow_synthetic_fallback: bool = True) -> Optional[Dict[str, Any]]:
    records = db.get_settled_stats_for_calibration() if hasattr(db, "get_settled_stats_for_calibration") else []
    if not records:
        print("\nNo settled player records found in lineup_history. Accumulate match outcomes to run empirical calibration.\n")
        return None

    # Check for Ground Truth (app_rrs from Real Sports app)
    app_records = [r for r in records if r.get("app_rrs") is not None]
    is_ground_truth = len(app_records) >= 5

    if is_ground_truth:
        target_records = app_records
        target_label = "Real Sports Mobile App Ground Truth (app_rrs)"
    elif allow_synthetic_fallback:
        target_records = records
        target_label = "Internal Heuristic Estimates (Synthetic actual_xrrs Fallback)"
    else:
        print("\nNo actual Real Sports mobile app ratings (app_rrs) found in lineup_history.")
        print("To avoid self-referential / circular validation of internal heuristics,")
        print("record actual app scores with: python main.py --record [YYYY-MM-DD]\n")
        return None

    valid = []
    for r in target_records:
        g = r.get("actual_goals")
        a = r.get("actual_assists")
        s = r.get("actual_shots")
        blk = r.get("actual_blocks", 0.0)
        target_val = r.get("app_rrs") if is_ground_truth else (r.get("actual_xrrs") if r.get("actual_xrrs") is not None else r.get("actual_rrs"))
        if g is not None and a is not None and s is not None and target_val is not None:
            valid.append({
                "goals": float(g),
                "assists": float(a),
                "shots": float(s),
                "blocks": float(blk or 0.0),
                "target": float(target_val),
                "slot_boost": float(r.get("slot_boost", 1.0)),
                "booster": float(r.get("booster", 0.0)),
                "app_payout": float(r["app_payout"]) if r.get("app_payout") is not None else None
            })

    if len(valid) < 5:
        print(f"\nNeed at least 5 settled records for empirical regression calibration (found {len(valid)}).")
        print("Run with --history, allow slate settlement, or record app scores with --record.\n")
        return None

    G = np.array([v["goals"] for v in valid], dtype=np.float64)
    A = np.array([v["assists"] for v in valid], dtype=np.float64)
    S = np.array([v["shots"] for v in valid], dtype=np.float64)
    BLK = np.array([v["blocks"] for v in valid], dtype=np.float64)
    ONES = np.ones(len(valid), dtype=np.float64)
    Y = np.array([v["target"] for v in valid], dtype=np.float64)

    has_blk = np.any(BLK > 0.0)
    has_intercept = is_ground_truth or np.any(BLK > 0.0)
    if has_intercept and has_blk:
        X = np.column_stack([ONES, G, A, S, BLK])
        feature_names = ["intercept", "goals", "assists", "shots", "blocks"]
    elif has_intercept:
        X = np.column_stack([ONES, G, A, S])
        feature_names = ["intercept", "goals", "assists", "shots"]
    else:
        X = np.column_stack([G, A, S])
        feature_names = ["goals", "assists", "shots"]

    weights, residuals, rank, s_vals = np.linalg.lstsq(X, Y, rcond=None)
    y_pred = X @ weights

    weights_dict = {}
    for name, w in zip(feature_names, weights):
        weights_dict[name] = round(float(w), 3)

    p = X.shape[1]
    n = len(valid)
    df = max(1, n - p)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    ss_res = np.sum((Y - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    mae = float(np.mean(np.abs(Y - y_pred)))
    rmse = float(np.sqrt(np.mean((Y - y_pred) ** 2)))

    try:
        sigma2 = ss_res / df
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(0.0, np.diag(cov)))
        t_stats = {name: round(float(w / (s + 1e-9)), 2) for name, w, s in zip(feature_names, weights, se)}
    except Exception:
        t_stats = {}

    holdout = None
    if len(valid) >= 20:
        order = sorted(range(len(valid)), key=lambda idx: valid[idx].get("game_date", ""))
        split = max(1, int(len(order) * 0.7))
        train_idx = order[:split]
        test_idx = order[split:]
        if test_idx:
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]
            holdout_weights, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)
            holdout_pred = X_test @ holdout_weights
            holdout_ss_res = np.sum((Y_test - holdout_pred) ** 2)
            holdout_ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
            holdout = {
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "r2": round(float(1.0 - holdout_ss_res / holdout_ss_tot), 4) if holdout_ss_tot > 0 else 0.0,
                "mae": round(float(np.mean(np.abs(Y_test - holdout_pred))), 3),
                "rmse": round(float(np.sqrt(np.mean((Y_test - holdout_pred) ** 2))), 3),
            }

    # Multiplier Stacking Test (Additive vs Multiplicative)
    stacking_result = None
    payout_records = [v for v in valid if v["app_payout"] is not None and v["target"] > 0]
    if len(payout_records) >= 3:
        act_p = np.array([v["app_payout"] for v in payout_records])
        rrs_vals = np.array([v["target"] for v in payout_records])
        s_b = np.array([v["slot_boost"] for v in payout_records])
        b_b = np.array([v["booster"] for v in payout_records])

        pred_add = rrs_vals * (s_b + b_b)
        pred_mult = rrs_vals * s_b * (1.0 + b_b)

        mae_add = float(np.mean(np.abs(act_p - pred_add)))
        mae_mult = float(np.mean(np.abs(act_p - pred_mult)))
        winner = "Additive [slot + booster]" if mae_add <= mae_mult else "Multiplicative [slot * (1 + booster)]"
        stacking_result = {
            "n_payout_records": len(payout_records),
            "mae_additive": round(mae_add, 3),
            "mae_multiplicative": round(mae_mult, 3),
            "verified_structure": winner
        }

    sample_warning = None
    if len(valid) < 15:
        sample_warning = f"Notice: Preliminary sample size (n={len(valid)} < 15) for estimating {len(feature_names)} parameters. Recommend N ≥ 15 for robust t-statistics."

    result = {
        "n_samples": len(valid),
        "is_ground_truth": is_ground_truth,
        "sample_warning": sample_warning,
        "weights": weights_dict,
        "t_stats": t_stats,
        "r2": round(r2, 4),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "holdout": holdout,
        "stacking_test": stacking_result
    }

    persisted = False
    min_calibration_samples = 15
    if is_ground_truth and len(valid) >= min_calibration_samples and hasattr(db, "save_model_params"):
        db.save_model_params(weights_dict)
        persisted = True
    result["persisted"] = persisted

    print("\n" + "=" * 80)
    print("           REAL SPORTS RRS EMPIRICAL FORMULA CALIBRATION (OLS)")
    print("=" * 80)
    print(f"Target:      {target_label}")
    print(f"Sample Size: {len(valid)} settled match boxscores from lineup_history")
    if sample_warning:
        print(f"             {sample_warning}")
    if not is_ground_truth:
        print("Warning: Ground-truth app_rrs not found (synthetic fallback used).")
        print("   Record actual Real Sports mobile app ratings via: python main.py --record [date]")
    print("\nEstimated Event Weights (OLS):")
    for name, w in weights_dict.items():
        t_val = f"(t = {t_stats.get(name, 0.0):+.2f})" if name in t_stats else ""
        print(f"  • {name.capitalize():<12}: {w:+.3f} {t_val}")

    print("\nGoodness of Fit:")
    print(f"  • R² (Variance Explained): {r2:.4f}")
    print(f"  • Mean Absolute Error:     {mae:.3f} RRS")
    print(f"  • Root Mean Squared Error: {rmse:.3f} RRS")

    if holdout:
        print("\nOut-of-Time Holdout:")
        print(f"  • Train/Test Split:        {holdout['n_train']}/{holdout['n_test']}")
        print(f"  • R² (Variance Explained): {holdout['r2']:.4f}")
        print(f"  • Mean Absolute Error:     {holdout['mae']:.3f} RRS")
        print(f"  • Root Mean Squared Error: {holdout['rmse']:.3f} RRS")

    if stacking_result:
        print("\nMultiplier Stacking Structure Verification:")
        print(f"  • Additive Model MAE:       {stacking_result['mae_additive']:.3f} RRS")
        print(f"  • Multiplicative Model MAE: {stacking_result['mae_multiplicative']:.3f} RRS")
        print(f"  • Confirmed Mechanic:       {stacking_result['verified_structure']}")

    if persisted:
        print("\nCalibrated weights saved; future projections will use them.")
    elif is_ground_truth:
        print(
            f"\nCalibrated weights were not saved: at least {min_calibration_samples} "
            "ground-truth samples are required."
        )

    print("=" * 80 + "\n")
    return result


def evaluate_model(
    db: Any,
    standings: Optional[Dict[str, Dict[str, float]]] = None,
    min_games: int = 30,
    point_in_time: bool = True
) -> Optional[Dict[str, Any]]:
    print("Evaluating model accuracy on out-of-sample test splits...")
    players = db.get_all_players() if hasattr(db, "get_all_players") else []
    if not players:
        print("No players found in database.")
        return

    pit_standings = {} if point_in_time else (standings or {})

    actuals = {"goals": [], "assists": [], "shots": [], "toi": []}
    preds_base = {"goals": [], "assists": [], "shots": [], "toi": []}
    preds_ridge = {"goals": [], "assists": [], "shots": [], "toi": []}
    preds_glm = {"goals": [], "assists": [], "shots": [], "toi": []}

    tested_players = 0

    for p in players:
        p_id = p["player_id"]
        logs = db.get_game_logs(p_id)
        if len(logs) < min_games:
            continue

        split = int(len(logs) * 0.8)
        train_logs = logs[:split]
        test_logs = logs[split:]

        # Train both Classic (Ridge) and New (Poisson/GLM) pipelines on strict historical train logs
        model_ridge = PerformancePredictor(model_type="ridge")
        model_ridge.train(train_logs, pit_standings)

        model_glm = PerformancePredictor(model_type="poisson")
        model_glm.train(train_logs, pit_standings)

        for i, test_game in enumerate(test_logs):
            history = logs[:split + i]
            is_h = (test_game.get("homeRoadFlag") == "H" or test_game.get("is_home") in (1, True))
            opp = test_game.get("opponentAbbrev", "")
            g_date = test_game.get("gameDate", "2026-01-01")
            pos = p.get("position", "F")
            is_b2b_val = bool(test_game.get("is_b2b", 0))
            rest_val = float(test_game.get("rest_days", 2.0))
            opp_b2b_val = bool(test_game.get("opp_is_b2b", 0))

            pred_r = model_ridge.predict(
                game_logs=history,
                is_home=is_h,
                opponent_abbrev=opp,
                game_date=g_date,
                standings=pit_standings,
                position=pos,
                is_b2b=is_b2b_val,
                rest_days=rest_val,
                opp_is_b2b=opp_b2b_val
            )

            pred_g = model_glm.predict(
                game_logs=history,
                is_home=is_h,
                opponent_abbrev=opp,
                game_date=g_date,
                standings=pit_standings,
                position=pos,
                is_b2b=is_b2b_val,
                rest_days=rest_val,
                opp_is_b2b=opp_b2b_val
            )

            act_g = float(test_game.get("goals", 0))
            act_a = float(test_game.get("assists", 0))
            act_s = float(test_game.get("shots", 0))
            act_toi = parse_toi(test_game.get("toi", "00:00"))

            # 20-game rolling baseline
            last20 = history[-20:] if len(history) >= 20 else history
            b_g = float(np.mean([float(x.get("goals", 0)) for x in last20]))
            b_a = float(np.mean([float(x.get("assists", 0)) for x in last20]))
            b_s = float(np.mean([float(x.get("shots", 0)) for x in last20]))
            b_toi = float(np.mean([parse_toi(x.get("toi", "00:00")) for x in last20]))

            actuals["goals"].append(act_g)
            actuals["assists"].append(act_a)
            actuals["shots"].append(act_s)
            actuals["toi"].append(act_toi)

            preds_base["goals"].append(b_g)
            preds_base["assists"].append(b_a)
            preds_base["shots"].append(b_s)
            preds_base["toi"].append(b_toi)

            preds_ridge["goals"].append(pred_r["xG"])
            preds_ridge["assists"].append(pred_r["xA"])
            preds_ridge["shots"].append(pred_r["xSOG"])
            preds_ridge["toi"].append(parse_toi(pred_r["xTOI"]))

            preds_glm["goals"].append(pred_g["xG"])
            preds_glm["assists"].append(pred_g["xA"])
            preds_glm["shots"].append(pred_g["xSOG"])
            preds_glm["toi"].append(parse_toi(pred_g["xTOI"]))

        tested_players += 1

    if not actuals["goals"]:
        print("Not enough game logs found to evaluate model.")
        return

    def calc_spearman(y_true, y_pred):
        if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        try:
            res = spearmanr(y_true, y_pred)
            val = float(getattr(res, "statistic", getattr(res, "correlation", 0.0)))
            return 0.0 if np.isnan(val) else val
        except Exception:
            return 0.0

    targets = ("goals", "assists", "shots", "toi")
    mae_base = {}
    rmse_base = {}
    sp_base = {}
    mae_ridge = {}
    rmse_ridge = {}
    sp_ridge = {}
    mae_glm = {}
    rmse_glm = {}
    sp_glm = {}

    for t in targets:
        y_act = np.array(actuals[t], dtype=np.float64)
        y_b = np.array(preds_base[t], dtype=np.float64)
        y_r = np.array(preds_ridge[t], dtype=np.float64)
        y_g = np.array(preds_glm[t], dtype=np.float64)

        mae_base[t] = float(np.mean(np.abs(y_b - y_act)))
        rmse_base[t] = float(np.sqrt(np.mean((y_b - y_act) ** 2)))
        sp_base[t] = calc_spearman(y_act, y_b)

        mae_ridge[t] = float(np.mean(np.abs(y_r - y_act)))
        rmse_ridge[t] = float(np.sqrt(np.mean((y_r - y_act) ** 2)))
        sp_ridge[t] = calc_spearman(y_act, y_r)

        mae_glm[t] = float(np.mean(np.abs(y_g - y_act)))
        rmse_glm[t] = float(np.sqrt(np.mean((y_g - y_act) ** 2)))
        sp_glm[t] = calc_spearman(y_act, y_g)

    print("-" * 75)
    leak_note = (
        "Opponent standings neutralized (no as-of standings)"
        if point_in_time
        else "End-of-season standings (with look-ahead)"
    )
    print(f"Validation Results: {tested_players} players, {len(actuals['goals'])} test games [{leak_note}]:")
    print(f"  Goals (MAE):    {mae_glm['goals']:.3f} per game  (Baseline 20-game avg: {mae_base['goals']:.3f})")
    print(f"  Assists (MAE):  {mae_glm['assists']:.3f} per game  (Baseline 20-game avg: {mae_base['assists']:.3f})")
    print(f"  Shots (MAE):    {mae_glm['shots']:.3f} per game  (Baseline 20-game avg: {mae_base['shots']:.3f})")
    print(f"  TOI (MAE):      {mae_glm['toi']:.2f} min       (Baseline 20-game avg: {mae_base['toi']:.2f} min)")
    print("-" * 75)

    print("\n" + "=" * 94)
    print("             MODEL VALIDATION & BENCHMARK COMPARISON (OUT-OF-SAMPLE)")
    print("=" * 94)
    print(f"{'Target':<10} {'Metric':<16} {'20-Game Baseline':<18} {'Classic (Ridge)':<18} {'New (Poisson GLM)':<18} {'Delta vs Classic'}")
    print("-" * 94)
    for tgt in targets:
        # MAE
        d_mae = ((mae_glm[tgt] - mae_ridge[tgt]) / mae_ridge[tgt] * 100) if mae_ridge[tgt] > 0 else 0.0
        d_mae_str = f"{d_mae:+.1f}% {'(improved)' if d_mae < 0 else ''}"
        print(f"{tgt.capitalize():<10} {'MAE':<16} {mae_base[tgt]:<18.3f} {mae_ridge[tgt]:<18.3f} {mae_glm[tgt]:<18.3f} {d_mae_str}")

        # RMSE
        d_rmse = ((rmse_glm[tgt] - rmse_ridge[tgt]) / rmse_ridge[tgt] * 100) if rmse_ridge[tgt] > 0 else 0.0
        d_rmse_str = f"{d_rmse:+.1f}% {'(improved)' if d_rmse < 0 else ''}"
        print(f"{'':<10} {'RMSE':<16} {rmse_base[tgt]:<18.3f} {rmse_ridge[tgt]:<18.3f} {rmse_glm[tgt]:<18.3f} {d_rmse_str}")

        # Spearman
        d_sp = sp_glm[tgt] - sp_ridge[tgt]
        d_sp_str = f"{d_sp:+.3f} {'(higher rank)' if d_sp > 0 else ''}"
        print(f"{'':<10} {'Spearman (r_s)':<16} {sp_base[tgt]:<+18.3f} {sp_ridge[tgt]:<+18.3f} {sp_glm[tgt]:<+18.3f} {d_sp_str}")
        print("-" * 94)
    print("=" * 94 + "\n")

    # Lineup-level validation report
    lineup_validation = None
    if hasattr(db, "get_lineup_level_validation_summary"):
        lineup_sum = db.get_lineup_level_validation_summary()
        if lineup_sum.get("settled_count", 0) > 0:
            lineup_validation = lineup_sum
            print(f"\nLineup-Level Validation ({lineup_sum['settled_count']} settled slates, {lineup_sum['total_players']} players):")
            print(f"  Lineup Total MAE:       {lineup_sum['lineup_mae']:.2f} RRS")
            print(f"  Lineup Total RMSE:      {lineup_sum['lineup_rmse']:.2f} RRS")
            print(f"  Overall Projection Bias:{lineup_sum['lineup_bias']:+.2f} RRS")
            print(f"  Beat-Projection Rate:   {lineup_sum['hit_rate']*100:.1f}%")
            if lineup_sum.get("slot_avg_errors"):
                slot_strs = [f"#{s}: {err:.1f}" for s, err in sorted(lineup_sum['slot_avg_errors'].items())]
                print(f"  Compounded Error by Slot: {', '.join(slot_strs)}")
            if lineup_sum.get("booster_avg_error") is not None and lineup_sum.get("no_booster_avg_error") is not None:
                print(f"  Booster Compounding: Boosted Avg Error {lineup_sum['booster_avg_error']:.2f} RRS vs Unboosted {lineup_sum['no_booster_avg_error']:.2f} RRS")
            print("-" * 68)

    metrics = {
        "baseline": {
            target: {"mae": mae_base[target], "rmse": rmse_base[target], "spearman": sp_base[target]}
            for target in targets
        },
        "ridge": {
            target: {"mae": mae_ridge[target], "rmse": rmse_ridge[target], "spearman": sp_ridge[target]}
            for target in targets
        },
        "glm": {
            target: {"mae": mae_glm[target], "rmse": rmse_glm[target], "spearman": sp_glm[target]}
            for target in targets
        },
    }

    return {
        "players": tested_players,
        "test_games": len(actuals["goals"]),
        "point_in_time": point_in_time,
        "metrics": metrics,
        "lineup_validation": lineup_validation,
    }
