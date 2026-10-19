import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional


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


def compute_line_factor(
    ev_line: Optional[str] = None,
    pp_unit: Optional[str] = None,
    position: str = "F",
    team_has_pp: bool = True
) -> Tuple[float, float, float]:
    """
    Computes (goal_mult, assist_mult, toi_adj) based on EV line deployment and PP unit.
    - PP1: +12% goals, +18% assists, +1.5 min TOI
    - PP2: +4% goals, +5% assists, +0.5 min TOI
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
    If ev_line and pp_unit are None, returns (1.0, 1.0, 0.0).
    """
    if not ev_line and not pp_unit:
        return 1.0, 1.0, 0.0

    g_mult = 1.0
    a_mult = 1.0
    toi_adj = 0.0

    # PP Adjustment
    if pp_unit == "PP1":
        g_mult += 0.12
        a_mult += 0.18
        toi_adj += 1.5
    elif pp_unit == "PP2":
        g_mult += 0.04
        a_mult += 0.05
        toi_adj += 0.5
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
    def __init__(self):
        self.multi_model: Optional[RidgeRegression] = None
        self.models: Dict[str, RidgeRegression] = {}
        self.trained = False
        self.baseline: Dict[str, float] = {}

    @property
    def is_trained(self) -> bool:
        return self.trained

    def _extract_dataset(
        self,
        game_logs: List[Dict[str, Any]],
        standings: Dict[str, Dict[str, float]]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if len(game_logs) < 6:
            return np.empty((0, 15)), {k: np.empty(0) for k in ("goals", "assists", "shots", "toi")}

        parsed = []
        for g in game_logs:
            parsed.append({
                "date": g.get("gameDate", "2020-01-01"),
                "goals": float(g.get("goals", 0)),
                "assists": float(g.get("assists", 0)),
                "shots": float(g.get("shots", 0)),
                "toi": parse_toi(g.get("toi", "00:00")),
                "is_home": 1.0 if g.get("homeRoadFlag") == "H" else 0.0,
                "opp": g.get("opponentAbbrev", "")
            })

        X_rows = []
        y_dict = {"goals": [], "assists": [], "shots": [], "toi": []}

        for i in range(5, len(parsed)):
            curr = parsed[i]
            prev = parsed[i - 1]

            try:
                d_curr = datetime.strptime(curr["date"], "%Y-%m-%d")
                d_prev = datetime.strptime(prev["date"], "%Y-%m-%d")
                rest = min(5.0, max(0.0, float((d_curr - d_prev).days)))
            except Exception:
                rest = 2.0

            opp_info = standings.get(curr["opp"], {})
            opp_def = opp_info.get("defense_factor", 1.0)

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
                rest,
                opp_def
            ]

            X_rows.append(feat)
            y_dict["goals"].append(curr["goals"])
            y_dict["assists"].append(curr["assists"])
            y_dict["shots"].append(curr["shots"])
            y_dict["toi"].append(curr["toi"])

        X = np.array(X_rows, dtype=np.float64)
        targets = {k: np.array(v, dtype=np.float64) for k, v in y_dict.items()}
        return X, targets

    def train(self, game_logs: List[Dict[str, Any]], standings: Optional[Dict[str, Dict[str, float]]] = None):
        st = standings or {}
        X, y_dict = self._extract_dataset(game_logs, st)

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

        y_mat = np.column_stack([y_dict[target] for target in ("goals", "assists", "shots", "toi")])
        self.multi_model = RidgeRegression(alpha=15.0)
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
        team_has_pp: bool = True
    ) -> Dict[str, Any]:
        pos_key = "D" if str(position).upper() in ("D", "DEF") else "F"
        pos_avg = POS_AVERAGES[pos_key]
        opp_info = (standings or {}).get(opponent_abbrev, {})
        opp_def = float(opp_info.get("defense_factor", 1.0))
        home_boost = 1.05 if is_home else 0.95

        if not game_logs:
            preds = {
                "goals": max(0.0, pos_avg["goals"] * opp_def * home_boost),
                "assists": max(0.0, pos_avg["assists"] * opp_def * home_boost),
                "shots": max(0.0, pos_avg["shots"] * opp_def * home_boost),
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

            last = parsed[-1]
            try:
                d_curr = datetime.strptime(game_date, "%Y-%m-%d")
                d_prev = datetime.strptime(last["date"], "%Y-%m-%d")
                rest = min(5.0, max(0.0, float((d_curr - d_prev).days)))
            except Exception:
                rest = 2.0

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
                rest,
                opp_def
            ]])

            preds = {}
            if self.trained and self.multi_model is not None:
                raw_preds = self.multi_model.predict(x_vec)[0]
                preds["goals"] = max(0.0, float(raw_preds[0]))
                preds["assists"] = max(0.0, float(raw_preds[1]))
                preds["shots"] = max(0.0, float(raw_preds[2]))
                preds["toi"] = max(0.0, float(raw_preds[3]))
            else:
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

                preds["goals"] = max(0.0, adj_g * opp_def * home_boost)
                preds["assists"] = max(0.0, adj_a * opp_def * home_boost)
                preds["shots"] = max(0.0, adj_s * opp_def * home_boost)
                preds["toi"] = max(5.0, adj_toi)

            if goalie_factor != 1.0:
                preds["goals"] = max(0.0, preds["goals"] * goalie_factor)
                preds["assists"] = max(0.0, preds["assists"] * (1.0 + 0.5 * (goalie_factor - 1.0)))

        # Line combination & power play deployment adjustment
        g_line_mult, a_line_mult, toi_line_adj = compute_line_factor(
            ev_line, pp_unit, position=position, team_has_pp=team_has_pp
        )
        if (g_line_mult, a_line_mult, toi_line_adj) != (1.0, 1.0, 0.0):
            preds["goals"] = max(0.0, preds["goals"] * g_line_mult)
            preds["assists"] = max(0.0, preds["assists"] * a_line_mult)
            preds["toi"] = max(5.0, preds["toi"] + toi_line_adj)

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
            is_def=is_def
        )

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
    is_def: bool = False
) -> float:
    """
    Computes expected Real Rating (xRRS) calibrated to the Real Sports mobile app:
    - Zero-point baseline: 0.0 - 0.9 (peripherals: SOG ~0.16, BLK ~0.35).
    - Goal expectancy: ~2.5 - 2.8 rating.
    - Assist expectancy: ~1.0 - 1.2 rating.
    - Multi-point exponential upside.
    - Low-event / non-scoring skaters are capped to prevent the 'high booster trap'.
    """
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

    return round(float(max(0.0, xrrs)), 2)


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
    return round(min(10.0, max(0.0, base)), 1)


def calibrate_rrs_formula(db: Any, allow_synthetic_fallback: bool = True) -> Optional[Dict[str, Any]]:
    records = db.get_settled_stats_for_calibration() if hasattr(db, "get_settled_stats_for_calibration") else []
    if not records:
        print("\n⚠️ No settled player records found in lineup_history. Accumulate match outcomes to run empirical calibration.\n")
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
        print("\n⚠️ No actual Real Sports mobile app ratings (app_rrs) found in lineup_history.")
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
        print(f"\n⚠️ Need at least 5 settled records for empirical regression calibration (found {len(valid)}).")
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
        "stacking_test": stacking_result
    }

    print("\n" + "=" * 80)
    print("           REAL SPORTS RRS EMPIRICAL FORMULA CALIBRATION (OLS)")
    print("=" * 80)
    print(f"Target:      {target_label}")
    print(f"Sample Size: {len(valid)} settled match boxscores from lineup_history")
    if sample_warning:
        print(f"             ⚠️ {sample_warning}")
    if not is_ground_truth:
        print("⚠️ Warning: Ground-truth app_rrs not found (synthetic fallback used).")
        print("   Record actual Real Sports mobile app ratings via: python main.py --record [date]")
    print("\nEstimated Event Weights (OLS):")
    for name, w in weights_dict.items():
        t_val = f"(t = {t_stats.get(name, 0.0):+.2f})" if name in t_stats else ""
        print(f"  • {name.capitalize():<12}: {w:+.3f} {t_val}")

    print("\nGoodness of Fit:")
    print(f"  • R² (Variance Explained): {r2:.4f}")
    print(f"  • Mean Absolute Error:     {mae:.3f} RRS")
    print(f"  • Root Mean Squared Error: {rmse:.3f} RRS")

    if stacking_result:
        print("\nMultiplier Stacking Structure Verification:")
        print(f"  • Additive Model MAE:       {stacking_result['mae_additive']:.3f} RRS")
        print(f"  • Multiplicative Model MAE: {stacking_result['mae_multiplicative']:.3f} RRS")
        print(f"  • Confirmed Mechanic:       {stacking_result['verified_structure']}")

    print("=" * 80 + "\n")
    return result


def evaluate_model(
    db: Any,
    standings: Optional[Dict[str, Dict[str, float]]] = None,
    min_games: int = 30,
    point_in_time: bool = True
):
    print("Evaluating model accuracy on out-of-sample test splits...")
    players = db.get_all_players() if hasattr(db, "get_all_players") else []
    if not players:
        print("No players found in database.")
        return

    pit_standings = {} if point_in_time else (standings or {})

    errors = {"goals": [], "assists": [], "shots": [], "toi": []}
    base_errors = {"goals": [], "assists": [], "shots": [], "toi": []}
    tested_players = 0

    for p in players:
        p_id = p["player_id"]
        logs = db.get_game_logs(p_id)
        if len(logs) < min_games:
            continue

        split = int(len(logs) * 0.8)
        train_logs = logs[:split]
        test_logs = logs[split:]

        model = PerformancePredictor()
        model.train(train_logs, pit_standings)

        for i, test_game in enumerate(test_logs):
            history = logs[:split + i]
            pred = model.predict(
                game_logs=history,
                is_home=(test_game.get("homeRoadFlag") == "H"),
                opponent_abbrev=test_game.get("opponentAbbrev", ""),
                game_date=test_game.get("gameDate", "2026-01-01"),
                standings=pit_standings,
                position=p.get("position", "F")
            )

            actual_g = float(test_game.get("goals", 0))
            actual_a = float(test_game.get("assists", 0))
            actual_s = float(test_game.get("shots", 0))
            actual_toi = parse_toi(test_game.get("toi", "00:00"))
            pred_toi = parse_toi(pred["xTOI"])

            errors["goals"].append(abs(pred["xG"] - actual_g))
            errors["assists"].append(abs(pred["xA"] - actual_a))
            errors["shots"].append(abs(pred["xSOG"] - actual_s))
            errors["toi"].append(abs(pred_toi - actual_toi))

            # 20-game rolling average baseline
            last20 = history[-20:] if len(history) >= 20 else history
            base_g = np.mean([float(x.get("goals", 0)) for x in last20])
            base_a = np.mean([float(x.get("assists", 0)) for x in last20])
            base_s = np.mean([float(x.get("shots", 0)) for x in last20])
            base_toi = np.mean([parse_toi(x.get("toi", "00:00")) for x in last20])

            base_errors["goals"].append(abs(base_g - actual_g))
            base_errors["assists"].append(abs(base_a - actual_a))
            base_errors["shots"].append(abs(base_s - actual_s))
            base_errors["toi"].append(abs(base_toi - actual_toi))

        tested_players += 1

    if not errors["goals"]:
        print("Not enough game logs found to evaluate model.")
        return

    print("-" * 68)
    leak_note = "Strict point-in-time (no look-ahead leakage)" if point_in_time else "End-of-season standings (with look-ahead)"
    print(f"Validation Results: {tested_players} players, {len(errors['goals'])} test games [{leak_note}]:")
    print(f"  Goals (MAE):    {np.mean(errors['goals']):.3f} per game  (Baseline 20-game avg: {np.mean(base_errors['goals']):.3f})")
    print(f"  Assists (MAE):  {np.mean(errors['assists']):.3f} per game  (Baseline 20-game avg: {np.mean(base_errors['assists']):.3f})")
    print(f"  Shots (MAE):    {np.mean(errors['shots']):.3f} per game  (Baseline 20-game avg: {np.mean(base_errors['shots']):.3f})")
    print(f"  TOI (MAE):      {np.mean(errors['toi']):.2f} min       (Baseline 20-game avg: {np.mean(base_errors['toi']):.2f} min)")
    print("-" * 68)

    # Lineup-level validation report
    if hasattr(db, "get_lineup_level_validation_summary"):
        lineup_sum = db.get_lineup_level_validation_summary()
        if lineup_sum.get("settled_count", 0) > 0:
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
