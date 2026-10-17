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


class PerformancePredictor:
    def __init__(self):
        self.multi_model: Optional[RidgeRegression] = None
        self.models: Dict[str, RidgeRegression] = {}
        self.trained = False
        self.baseline: Dict[str, float] = {}

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

    def train(self, game_logs: List[Dict[str, Any]], standings: Dict[str, Dict[str, float]]):
        X, y_dict = self._extract_dataset(game_logs, standings)

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
        standings: Dict[str, Dict[str, float]],
        position: str = "F",
        goalie_factor: float = 1.0
    ) -> Dict[str, Any]:
        if not game_logs:
            preds = dict(self.baseline)
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

            opp_info = standings.get(opponent_abbrev, {})
            opp_def = opp_info.get("defense_factor", 1.0)
            if goalie_factor != 1.0:
                opp_def = round(opp_def * goalie_factor, 3)

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
                if goalie_factor != 1.0:
                    preds["goals"] = max(0.0, preds["goals"] * goalie_factor)
                    preds["assists"] = max(0.0, preds["assists"] * (1.0 + 0.5 * (goalie_factor - 1.0)))
            else:
                home_boost = 1.05 if is_home else 0.95
                r5_g = np.mean([x["goals"] for x in last5])
                r5_a = np.mean([x["assists"] for x in last5])
                r5_s = np.mean([x["shots"] for x in last5])
                r5_toi = np.mean([x["toi"] for x in last5])
                preds["goals"] = max(0.0, r5_g * opp_def * home_boost)
                preds["assists"] = max(0.0, r5_a * opp_def * home_boost)
                preds["shots"] = max(0.0, r5_s * opp_def * home_boost)
                preds["toi"] = max(5.0, r5_toi)

        is_def = position in ("D", "DEF")
        blk_rate = 0.09 if is_def else 0.04
        blocked = round(preds["toi"] * blk_rate, 2)
        points = round(preds["goals"] + preds["assists"], 2)

        toi_min = int(preds["toi"])
        toi_sec = int((preds["toi"] - toi_min) * 60)
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
            "base_pts": rrs_base
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
    periph = xsog * 0.16 + xblk * 0.35
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
    - 0 points (only shots/blocks): 0.0 - 0.9.
    - 1 Assist: ~1.0 - 1.3.
    - 1 Goal: ~2.5 - 3.5.
    - Multi-point games (2+ points): ~3.7 - 5.2.
    - Star multi-point performances: 8.0+.
    """
    pts = goals + assists
    if pts == 0:
        raw = shots * 0.16 + blocks * 0.45
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

    base += shots * 0.16 + blocks * 0.30
    return round(min(10.0, max(0.0, base)), 1)


def evaluate_model(db: Any, standings: Dict[str, Dict[str, float]], min_games: int = 30):
    print("Evaluating model accuracy on out-of-sample test splits...")
    players = db.get_all_players()
    if not players:
        print("No players found in database.")
        return

    errors = {"goals": [], "assists": [], "shots": [], "toi": []}
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
        model.train(train_logs, standings)

        for i, test_game in enumerate(test_logs):
            history = logs[:split + i]
            pred = model.predict(
                game_logs=history,
                is_home=(test_game.get("homeRoadFlag") == "H"),
                opponent_abbrev=test_game.get("opponentAbbrev", ""),
                game_date=test_game.get("gameDate", "2026-01-01"),
                standings=standings,
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

        tested_players += 1

    if not errors["goals"]:
        print("Not enough game logs found to evaluate model.")
        return

    print("-" * 48)
    print(f"Validation Results ({tested_players} players, {len(errors['goals'])} test games):")
    print(f"  Goals (MAE):    {np.mean(errors['goals']):.3f} per game")
    print(f"  Assists (MAE):  {np.mean(errors['assists']):.3f} per game")
    print(f"  Shots (MAE):    {np.mean(errors['shots']):.3f} per game")
    print(f"  TOI (MAE):      {np.mean(errors['toi']):.2f} minutes")
    print("-" * 48)
