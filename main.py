import argparse
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from boosters import (
    BOOSTERS_FILE,
    _warned_ambiguous_names,
    get_player_booster,
    load_boosters_file,
    name_key,
    normalize_booster_name,
    parse_booster_val,
    resolve_booster,
    save_boosters_file,
    sync_boosters_to_db,
)
from cli_utils import parse_history_arg, parse_iso_date_arg
from db import Database
from model import (
    POS_AVERAGES,
    PerformancePredictor,
    calculate_xrrs_projection,
    calibrate_rrs_formula,
    compute_line_factor,
    evaluate_model,
)
from nhl_api import NHLClient, nhl_today
from paths import migrate_legacy_files
from reporting import handle_record_app_scores, show_history_report
from settlement import settle_past_lineups
from slate import build_team_matchups, select_game_day

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from optimizer import (
    SLOT_BOOSTS,
    assign_slots,
    build_lineup_correlation_matrix,
    compute_pair_correlation,
    compute_pair_synergy,
    print_lineup_table,
    simulate_lineup_distribution,
    solve_optimal_lineup,
    solve_stacked_lineup,
)

__all__ = [
    "SLOT_BOOSTS",
    "assign_slots",
    "build_lineup_correlation_matrix",
    "compute_pair_correlation",
    "compute_pair_synergy",
    "print_lineup_table",
    "simulate_lineup_distribution",
    "solve_optimal_lineup",
    "solve_stacked_lineup",
    "BOOSTERS_FILE",
    "_warned_ambiguous_names",
    "get_player_booster",
    "load_boosters_file",
    "name_key",
    "normalize_booster_name",
    "parse_booster_val",
    "resolve_booster",
    "save_boosters_file",
    "sync_boosters_to_db",
    "handle_record_app_scores",
    "parse_history_arg",
    "parse_iso_date_arg",
    "settle_past_lineups",
    "show_history_report",
]

VERSION = "2.6.3"

ESPN_TEAM_MAP = {
    "NJ": "NJD",
    "TB": "TBL",
    "SJ": "SJS",
    "LA": "LAK",
    "UTAH": "UTA"
}

NHL_TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens", "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils", "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks", "SEA": "Seattle Kraken",
    "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs", "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals", "WPG": "Winnipeg Jets"
}


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logger = logging.getLogger("nhl_api")
    logger.setLevel(level)
    if not any(getattr(handler, "_rrs_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler._rrs_handler = True
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


def print_quant_banner(mode: str = "Optimal Draft & Stacking", target_date: str = ""):
    sub = f"Mode: {mode}"
    if target_date:
        sub += f" | Date: {target_date}"
    banner = (
        rf"  ____   ____   ____     RRS ENGINE v{VERSION}" "\n"
        r" |  _ \ |  _ \ / ___|    NHL Quant Draft Optimizer | Real Rating System" "\n"
        rf" | |_) || |_) |\___ \    {sub}" "\n"
        r" |  _ < |  _ <  ___) |" "\n"
        r" |_| \_\|_| \_\|____/ " "\n"
    )
    print(banner)


def is_out_status(status: Optional[str]) -> bool:
    if not status:
        return False
    st = status.strip().lower()
    if st in ("out", "injured reserve", "ir", "suspension", "suspended"):
        return True
    if any(k in st for k in ("injured reserve", "suspended", "suspension")):
        return True
    return False


def is_dtd_status(status: Optional[str]) -> bool:
    if not status:
        return False
    st = status.strip().lower()
    if st in ("day-to-day", "dtd", "questionable", "doubtful", "gtd", "game-time decision", "game time decision"):
        return True
    if any(k in st for k in ("day-to-day", "questionable", "doubtful", "game-time", "gtd")):
        return True
    return False


def match_injury(
    injuries: Dict[Any, Dict[str, str]],
    player_name: str,
    player_team: str,
    is_duplicate: bool
) -> Optional[Dict[str, str]]:
    p_nk = name_key(player_name)
    p_team = player_team.upper()
    for key, inj_data in injuries.items():
        if isinstance(key, tuple):
            inj_name, raw_team = key
        else:
            inj_name = inj_data.get("name") or key
            raw_team = inj_data.get("team", "")

        if name_key(inj_name) == p_nk:
            mapped_team = ESPN_TEAM_MAP.get(raw_team.upper(), raw_team.upper())
            if is_duplicate:
                if mapped_team == p_team:
                    return inj_data
            else:
                if not mapped_team or mapped_team == p_team:
                    return inj_data
    return None


def collect_slate_skaters(
    client: NHLClient,
    db: Database,
    team_matchups: Dict[str, Tuple[str, bool]],
    goalies: Dict[str, Dict[str, Any]],
    injuries: Dict[Any, Dict[str, str]],
    boosters: Dict[str, float],
    team_lines: Optional[Dict[str, Dict[Any, Dict[str, Any]]]] = None,
    game_date: str = "",
    today_str: Optional[str] = None,
    scoring_weights: Optional[Dict[str, float]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int, int, Set[str]]:
    all_club_stats = client.get_all_club_stats(list(team_matchups.keys()))
    if team_lines is None:
        team_lines = client.prefetch_teams_lines(list(team_matchups.keys()))

    if today_str is None:
        today_str = nhl_today()
    if scoring_weights is None and db is not None:
        scoring_weights = db.get_model_params()
    is_past_date = bool(game_date and game_date < today_str)

    all_skater_names = []
    all_skater_name_pos = []
    for team in team_matchups.keys():
        for p in all_club_stats.get(team, []):
            fn = f"{p['firstName']['default']} {p['lastName']['default']}"
            all_skater_names.append(fn)
            all_skater_name_pos.append((fn, p.get("positionCode", "F")))
    name_counts = Counter(all_skater_names)
    name_pos_counts = Counter(all_skater_name_pos)

    active_skaters: List[Dict[str, Any]] = []
    dtd_skaters: List[Dict[str, Any]] = []
    depth_skaters: List[Dict[str, Any]] = []
    players_to_save: List[Dict[str, Any]] = []
    matched_booster_keys: Set[str] = set()

    excluded_count = 0
    dtd_skipped_count = 0

    standings = db.get_standings() if db else {}

    for team, (opp, is_home) in team_matchups.items():
        skaters = all_club_stats.get(team, [])
        opp_goalie = goalies.get(opp, {})
        g_factor = opp_goalie.get("goalie_factor", 1.0)
        g_name = opp_goalie.get("goalie_name", "")
        t_lines = (team_lines or {}).get(team, {})
        team_has_pp = t_lines.get("_meta", {}).get("has_pp", any(v.get("pp_unit") for k, v in t_lines.items() if isinstance(v, dict) and k != "_meta"))

        sched_ctx = db.get_schedule_context(team, opp, game_date) if (db and game_date) else {
            "is_home": is_home, "is_b2b": False, "rest_days": 3.0, "opp_is_b2b": False, "opp_rest_days": 3.0
        }
        is_b2b = bool(sched_ctx.get("is_b2b", False))
        rest_days = float(sched_ctx.get("rest_days", 3.0))
        opp_is_b2b = bool(sched_ctx.get("opp_is_b2b", False))

        opp_info = (standings or {}).get(opp, {})
        opp_pk_pct = opp_info.get("pk_pct")
        opp_times_shorthanded = opp_info.get("times_shorthanded_per_game")

        for p in skaters:
            p_id = p["playerId"]
            full_name = f"{p['firstName']['default']} {p['lastName']['default']}"
            pos = p.get("positionCode", "F")
            pos_key = "D" if pos in ("D", "DEF") else "F"
            pos_group = pos_key
            gp = p.get("gamesPlayed", 0)
            avg_toi = p.get("avgTimeOnIcePerGame", 0.0) / 60.0
            is_dup = name_counts[full_name] > 1
            is_pos_dup = name_pos_counts[(full_name, pos)] > 1
            display_name = f"{full_name} ({pos}, {team})" if is_pos_dup else (f"{full_name} ({pos})" if is_dup else full_name)

            p_line = t_lines.get((name_key(full_name), pos_group)) or t_lines.get(name_key(full_name)) or {}
            if is_past_date:
                ev_line = None
                pp_unit = None
                pk_unit = None
                line_tag = "--"
            else:
                ev_line = p_line.get("ev_line")
                pp_unit = p_line.get("pp_unit")
                pk_unit = p_line.get("pk_unit")
                if ev_line and pp_unit:
                    line_tag = f"{ev_line}/{pp_unit}"
                elif ev_line:
                    line_tag = ev_line
                elif pp_unit:
                    line_tag = pp_unit
                else:
                    line_tag = "--"

            players_to_save.append({"id": p_id, "name": full_name, "team": team, "pos": pos})

            b_val, matched_key = resolve_booster(boosters, full_name, pos, team=team, is_duplicate=is_dup)
            if matched_key:
                matched_booster_keys.add(matched_key)

            has_booster = b_val > 0.0

            # Injury check
            inj = match_injury(injuries, full_name, team, is_duplicate=is_dup)
            if inj:
                status = inj.get("status")
                detail = inj.get("detail", "")
                if is_out_status(status):
                    excluded_count += 1
                    continue
                elif is_dtd_status(status) and not has_booster:
                    dtd_skipped_count += 1
                    status_lbl = status or "Day-To-Day"
                    dtd_skaters.append({
                        "name": full_name,
                        "display_name": display_name,
                        "team": team,
                        "pos": pos,
                        "inj_detail": f"{detail} ({status_lbl})" if detail else status_lbl,
                        "is_dup": is_dup,
                        "is_pos_dup": is_pos_dup,
                        "ev_line": ev_line,
                        "pp_unit": pp_unit,
                        "pk_unit": pk_unit,
                        "line_tag": line_tag,
                        "team_has_pp": team_has_pp
                    })
                    continue

            # Fast filter: skip non-scoring depth players if no custom booster
            if not has_booster and (gp < 1 or avg_toi < 14.5):
                depth_skaters.append({
                    "name": full_name,
                    "display_name": display_name,
                    "team": team,
                    "pos": pos,
                    "gp": gp,
                    "avg_toi": avg_toi,
                    "is_dup": is_dup,
                    "is_pos_dup": is_pos_dup,
                    "ev_line": ev_line,
                    "pp_unit": pp_unit,
                    "pk_unit": pk_unit,
                    "line_tag": line_tag,
                    "team_has_pp": team_has_pp
                })
                continue

            # Baseline calculation for game-day previews with Bayesian shrinkage
            pos_avg = POS_AVERAGES[pos_key]
            k = 10.0
            if gp == 0:
                g_pg = pos_avg["goals"] * g_factor
                a_pg = pos_avg["assists"] * (1.0 + 0.5 * (g_factor - 1.0))
                s_pg = pos_avg["shots"]
                avg_toi = pos_avg["toi"]
            elif gp < 10:
                raw_g = p.get("goals", 0) / gp
                raw_a = p.get("assists", 0) / gp
                raw_s = p.get("shots", 0) / gp
                shrunk_g = (raw_g * gp + pos_avg["goals"] * k) / (gp + k)
                shrunk_a = (raw_a * gp + pos_avg["assists"] * k) / (gp + k)
                shrunk_s = (raw_s * gp + pos_avg["shots"] * k) / (gp + k)
                avg_toi = (avg_toi * gp + pos_avg["toi"] * k) / (gp + k) if avg_toi > 0 else pos_avg["toi"]
                g_pg = shrunk_g * g_factor
                a_pg = shrunk_a * (1.0 + 0.5 * (g_factor - 1.0))
                s_pg = shrunk_s
            else:
                g_pg = ((p.get("goals", 0) / gp) * g_factor)
                a_pg = ((p.get("assists", 0) / gp) * (1.0 + 0.5 * (g_factor - 1.0)))
                s_pg = (p.get("shots", 0) / gp)

            g_line_mult, a_line_mult, toi_line_adj = compute_line_factor(
                ev_line, pp_unit, position=pos, team_has_pp=team_has_pp,
                opp_pk_pct=opp_pk_pct, opp_times_shorthanded=opp_times_shorthanded
            )
            g_pg = max(0.0, g_pg * g_line_mult)
            a_pg = max(0.0, a_pg * a_line_mult)
            avg_toi = max(5.0, avg_toi + toi_line_adj)

            blk_pg = avg_toi * (0.09 if pos_key == "D" else 0.04)
            is_d = (pos_key == "D")
            est_xrrs = calculate_xrrs_projection(g_pg, a_pg, s_pg, blk_pg, is_def=is_d, weights=scoring_weights)

            active_skaters.append({
                "id": p_id,
                "name": full_name,
                "display_name": display_name,
                "team": team,
                "opp": opp,
                "is_home": is_home,
                "opp_goalie": g_name,
                "goalie_factor": g_factor,
                "pos": pos,
                "gp": gp,
                "avg_toi": avg_toi,
                "xRRS": est_xrrs,
                "base_pts": est_xrrs,
                "booster": b_val,
                "is_dup": is_dup,
                "is_pos_dup": is_pos_dup,
                "ev_line": ev_line,
                "pp_unit": pp_unit,
                "pk_unit": pk_unit,
                "line_tag": line_tag,
                "team_has_pp": team_has_pp,
                "is_b2b": is_b2b,
                "rest_days": rest_days,
                "opp_is_b2b": opp_is_b2b,
                "opp_pk_pct": opp_pk_pct,
                "opp_times_shorthanded": opp_times_shorthanded
            })

    if players_to_save:
        db.save_players_batch(players_to_save)

    return active_skaters, dtd_skaters, depth_skaters, excluded_count, dtd_skipped_count, matched_booster_keys


def sync_game_day(client: NHLClient, db: Database, target_date: str = "", force: bool = False) -> List[Dict[str, Any]]:
    print("Fetching schedule from NHL API...")
    schedule = client.get_upcoming_schedule(force=force)
    if not schedule:
        print("Schedule is currently unavailable.")
        return []

    today_day = select_game_day(schedule, target_date, nhl_today())

    if not today_day:
        print("No scheduled games found.")
        return []

    game_date = today_day["date"]
    games = today_day.get("games", [])
    print(f"Game Day: {game_date} ({len(games)} matches)")

    team_matchups = build_team_matchups(games)

    injuries = client.get_injuries(force=force)
    goalies = client.get_starting_goalies(games=games, force=force)
    team_lines = client.prefetch_teams_lines(list(team_matchups.keys()), force=force)

    if os.path.exists(BOOSTERS_FILE):
        existing_boosters = load_boosters_file(BOOSTERS_FILE)
    else:
        existing_boosters = db.get_all_boosters()

    print("Fetching club rosters and stats in parallel...")
    active_skaters, dtd_skaters, depth_skaters, excluded_count, dtd_skipped, _ = collect_slate_skaters(
        client, db, team_matchups, goalies, injuries, existing_boosters, team_lines=team_lines,
        game_date=game_date
    )

    candidates = sorted(active_skaters, key=lambda x: x["xRRS"], reverse=True)
    dtd_skaters.sort(key=lambda x: x["name"])

    # Save to boosters.txt
    save_boosters_file(existing_boosters, candidates, dtd_players=dtd_skaters, depth_players=depth_skaters)
    sync_boosters_to_db(db, existing_boosters)

    print(f"Saved {len(candidates)} active game-day players to {BOOSTERS_FILE}")
    if excluded_count > 0 or dtd_skaters:
        print(f"Injuries: {excluded_count} Out/IR players excluded, {len(dtd_skaters)} Day-To-Day players commented out.")
    if goalies:
        print(f"Goalies: {len(goalies)} starting goalies tracked.")
    if team_matchups:
        m_count = len(team_matchups)
        loaded_count = sum(1 for t in team_matchups if t in team_lines and team_lines[t])
        missing_teams = [t for t in team_matchups if t not in team_lines or not team_lines[t]]
        missing_str = f" (missing: {', '.join(missing_teams)})" if missing_teams else ""
        print(f"Lines: loaded {loaded_count} of {m_count} teams{missing_str}.")
    return candidates


# Note: assign_slots, compute_pair_synergy, solve_optimal_lineup,
# solve_stacked_lineup, and print_lineup_table are imported from optimizer.py


def is_slate_started(games: List[Dict[str, Any]], game_date: str, today_str: Optional[str] = None) -> bool:
    if today_str is None:
        today_str = nhl_today()
    if game_date < today_str:
        return True
    if any(g.get("gameState") in ("LIVE", "CRIT", "OFF", "FINAL") for g in games):
        return True
    now_utc = datetime.now(timezone.utc)
    for g in games:
        st = g.get("startTimeUTC")
        if st:
            try:
                game_time = datetime.fromisoformat(st.replace("Z", "+00:00"))
                if game_time <= now_utc:
                    return True
            except Exception:
                pass
    return False


def build_skater_projection(
    skater: Dict[str, Any],
    logs: List[Dict[str, Any]],
    standings: Dict[str, Dict[str, float]],
    game_date: str,
    today_str: Optional[str] = None,
    db: Optional[Database] = None,
    scoring_weights: Optional[Dict[str, float]] = None
) -> Tuple[float, float]:
    if today_str is None:
        today_str = nhl_today()
    train_logs = [log for log in logs if log.get("gameDate", "") < game_date]
    if train_logs:
        predictor = PerformancePredictor(scoring_weights=scoring_weights)
        predictor.train(train_logs, standings)
        opp_raw = skater.get("opp", "")
        opp_clean = re.sub(r'^[vs@]+', '', opp_raw).strip()
        is_future_or_today = (game_date >= today_str)
        ev_line = skater.get("ev_line") if is_future_or_today else None
        pp_unit = skater.get("pp_unit") if is_future_or_today else None
        team_has_pp = skater.get("team_has_pp", True)

        team = skater.get("team", "")
        is_b2b = bool(skater.get("is_b2b", False))
        rest_days = skater.get("rest_days")
        opp_is_b2b = bool(skater.get("opp_is_b2b", False))
        if db is not None and team and opp_clean and game_date:
            try:
                ctx = db.get_schedule_context(team, opp_clean, game_date)
                is_b2b = bool(ctx.get("is_b2b", is_b2b))
                rest_days = ctx.get("rest_days", rest_days)
                opp_is_b2b = bool(ctx.get("opp_is_b2b", opp_is_b2b))
            except Exception:
                pass

        opp_info = (standings or {}).get(opp_clean, {})
        opp_pk_pct = skater.get("opp_pk_pct", opp_info.get("pk_pct"))
        opp_times_shorthanded = skater.get("opp_times_shorthanded", opp_info.get("times_shorthanded_per_game"))

        pred = predictor.predict(
            game_logs=train_logs,
            is_home=skater.get("is_home", True),
            opponent_abbrev=opp_clean,
            game_date=game_date,
            standings=standings,
            position=skater.get("pos", "F"),
            goalie_factor=skater.get("goalie_factor", 1.0),
            ev_line=ev_line,
            pp_unit=pp_unit,
            team_has_pp=team_has_pp,
            is_b2b=is_b2b,
            rest_days=rest_days,
            opp_is_b2b=opp_is_b2b,
            opp_pk_pct=opp_pk_pct,
            opp_times_shorthanded=opp_times_shorthanded
        )
        return pred["xRRS"], pred["base_pts"]
    else:
        # Past slates must not use end-of-season club stats.
        if game_date < today_str:
            return 0.0, 0.0
        xrrs = skater.get("xRRS", 0.0)
        base = skater.get("base_pts", xrrs)
        return xrrs, base


def main():
    parser = argparse.ArgumentParser(description=f"RRS Engine v{VERSION} - Real Rating System & NHL Quant Draft Optimizer")
    parser.add_argument("--version", action="version", version=f"RRS Engine {VERSION}")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sync", action="store_true", help="Sync game-day roster and refresh boosters.txt")
    group.add_argument("--history", nargs="?", const="all", default=None, type=parse_history_arg, help="Show performance tracking and settled RRS history (optional: YYYY-MM-DD)")
    group.add_argument("--record", nargs="?", const="today", default=None, help="Record actual Real Sports mobile app ratings and payouts (YYYY-MM-DD or YYYY-MM-DD:s1,s2,s3,s4,s5)")
    group.add_argument("--parse-images", type=str, default=None, metavar="FOLDER", help="Parse Real Sports draft screenshots from folder to populate boosters.txt")
    group.add_argument("--train", action="store_true", help="Run ML model validation (MAE metrics)")
    group.add_argument("--calibrate", action="store_true", help="Calibrate RRS scoring formula weights via OLS regression on settled history")
    group.add_argument("--stack", action="store_true", help="Generate only the correlated tournament stack lineup")
    parser.add_argument("--date", type=parse_iso_date_arg, default="", help="Target date (YYYY-MM-DD), default is current day")
    parser.add_argument("--slots", type=str, default="", help="Custom comma-separated slot multipliers (default: 2.0,1.8,1.6,1.4,1.2)")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    log_group.add_argument("--quiet", action="store_true", help="Suppress warning and info logs.")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose, quiet=args.quiet)

    custom_slots = None
    if args.slots:
        try:
            custom_slots = [float(x.strip()) for x in args.slots.split(",") if x.strip()]
            if len(custom_slots) < 5:
                print(f"Warning: Provided {len(custom_slots)} custom slots (expected 5). Falling back to default.")
                custom_slots = None
        except ValueError:
            print("Warning: Invalid slot format. Expected comma-separated floats (e.g. 2.0,1.8,1.6,1.4,1.2).")
            custom_slots = None

    if args.sync:
        mode_label = "Sync Roster & Boosters"
    elif args.parse_images is not None:
        mode_label = "Parse Draft Screenshots"
    elif args.train:
        mode_label = "Model Validation & Backtest"
    elif args.calibrate:
        mode_label = "Empirical Scoring Calibration"
    elif args.record is not None:
        mode_label = "Record Real App Scores"
    elif args.history is not None:
        mode_label = "Performance Tracking & History"
    elif args.stack:
        mode_label = "Tournament Stack Optimizer"
    else:
        mode_label = "Optimal Draft & Stacking"

    print_quant_banner(mode=mode_label, target_date=args.date)

    migrate_legacy_files(Path(__file__).resolve().parent)
    db = Database()
    client = NHLClient(db=db)
    standings = client.get_standings()

    # Settle any past games in history
    settle_past_lineups(client, db)

    if args.parse_images is not None:
        from image_parser import handle_parse_images
        result = handle_parse_images(args.parse_images, client, db, target_date=args.date)
        if not result:
            raise SystemExit(1)
        return

    if args.record is not None:
        handle_record_app_scores(db, args.record)
        return

    if args.history is not None:
        target_d = args.history if args.history != "all" else None
        show_history_report(db, target_date=target_d)
        return

    if args.calibrate:
        calibrate_rrs_formula(db)
        return

    if args.train:
        evaluate_model(db, standings)
        return

    # If --sync is passed or boosters.txt does not exist, run sync first
    if args.sync or not os.path.exists(BOOSTERS_FILE):
        if not args.sync:
            print("No boosters file found; syncing the current slate first.")
        if not sync_game_day(client, db, target_date=args.date, force=True):
            raise SystemExit(1)
        if args.sync:
            return

    # 1. Load boosters from boosters.txt into DB if file exists
    if os.path.exists(BOOSTERS_FILE):
        boosters = load_boosters_file(BOOSTERS_FILE)
        sync_boosters_to_db(db, boosters)
    else:
        boosters = db.get_all_boosters()

    model_weights = db.get_model_params()

    # 2. Get schedule and target date
    schedule = client.get_upcoming_schedule()
    if not schedule:
        print("Schedule is unavailable.")
        raise SystemExit(1)

    target_day = select_game_day(schedule, args.date, nhl_today())

    if not target_day:
        print(f"No games found for date: {args.date or 'today'}")
        raise SystemExit(1)

    game_date = target_day["date"]
    games = target_day.get("games", [])

    team_matchups = build_team_matchups(games)

    # 3. Collect and predict for all active players in today's games
    injuries = client.get_injuries()
    goalies = client.get_starting_goalies(games=games)
    team_lines = client.prefetch_teams_lines(list(team_matchups.keys()))

    print("Fetching club rosters and stats in parallel...")
    active_skaters, _, _, excluded_injuries, dtd_skipped, matched_keys = collect_slate_skaters(
        client, db, team_matchups, goalies, injuries, boosters, team_lines=team_lines,
        game_date=game_date, scoring_weights=model_weights
    )

    if excluded_injuries > 0 or dtd_skipped > 0:
        print(f"Injuries: excluded {excluded_injuries} Out/IR players, skipped {dtd_skipped} Day-To-Day players without booster.")
    if goalies:
        print(f"Goalies: matched {len(goalies)} starting goalies for today's games.")
    if team_matchups:
        m_count = len(team_matchups)
        loaded_count = sum(1 for t in team_matchups if t in team_lines and team_lines[t])
        missing_teams = [t for t in team_matchups if t not in team_lines or not team_lines[t]]
        missing_str = f" (missing: {', '.join(missing_teams)})" if missing_teams else ""
        print(f"Lines: loaded {loaded_count} of {m_count} teams{missing_str}.")

    # Check for unmatched boosters (> 0) to notify user of potential typos
    unmatched_boosters = [k for k, v in boosters.items() if v > 0.0 and k not in matched_keys]
    if unmatched_boosters:
        print(f"Warning: Unused boosters (>0) not matched to any slate skater: {', '.join(sorted(unmatched_boosters))}")

    # Prefetch game logs in parallel for all active skaters
    print(f"Synchronizing game logs for {len(active_skaters)} qualifying skaters (in parallel)...")
    qualifying_ids = [s["id"] for s in active_skaters]
    logs_map = client.prefetch_players_game_logs(qualifying_ids, max_workers=8)

    candidates: List[Dict[str, Any]] = []
    total_qual = len(active_skaters)

    for idx, s in enumerate(active_skaters, 1):
        p_id = s["id"]
        logs = logs_map.get(p_id) or []

        if idx % 5 == 0 or idx == total_qual:
            sys.stdout.write(f"\rTraining ML models & projecting xRRS: [{idx}/{total_qual}]...")
            sys.stdout.flush()

        today_str = nhl_today()
        pred_xrrs, base_pts = build_skater_projection(
            s, logs, standings, game_date, today_str=today_str, db=db,
            scoring_weights=model_weights
        )

        candidates.append({
            "id": p_id,
            "name": s["name"],
            "display_name": s.get("display_name", s["name"]),
            "team": s["team"],
            "opp": f"{'vs' if s['is_home'] else '@'}{s['opp']}",
            "opp_goalie": s["opp_goalie"],
            "goalie_factor": s["goalie_factor"],
            "pos": s["pos"],
            "ev_line": s.get("ev_line"),
            "pp_unit": s.get("pp_unit"),
            "line_tag": s.get("line_tag", "--"),
            "xRRS": pred_xrrs,
            "base_pts": base_pts,
            "booster": s["booster"],
            "is_b2b": s.get("is_b2b", False),
            "opp_is_b2b": s.get("opp_is_b2b", False)
        })

    if total_qual > 0:
        sys.stdout.write(f"\rTraining ML models & projecting xRRS: completed for {len(candidates)} active skaters.      \n")
        sys.stdout.flush()

    today_str = nhl_today()
    can_save_history = not is_slate_started(games, game_date, today_str)

    if candidates and all(c.get("xRRS", 0.0) == 0.0 for c in candidates):
        print(f"\nWarning: No valid historical projections available for slate {game_date}. All skater projections evaluated to 0.0 (historical game logs prior to {game_date} are missing). Run with a current date or populate historical logs.")
        return

    # 4. Solve optimal 5-slot lineup(s)
    if args.stack:
        stacked_lineup, base_score, syn_score = solve_stacked_lineup(candidates, slot_boosts=custom_slots)
        if can_save_history:
            db.save_lineup_history(game_date, "stack", stacked_lineup)
        print_lineup_table(game_date, stacked_lineup, base_score, title="Tournament Stack (Correlated Upside)", synergy_score=syn_score)
    else:
        std_lineup, std_score = solve_optimal_lineup(candidates, slot_boosts=custom_slots)
        if can_save_history:
            db.save_lineup_history(game_date, "standard", std_lineup)
        print_lineup_table(game_date, std_lineup, std_score, title="Lineup #1: Standard Optimal (Max Base EV)")

        stacked_lineup, base_score, syn_score = solve_stacked_lineup(candidates, slot_boosts=custom_slots)
        if can_save_history:
            db.save_lineup_history(game_date, "stack", stacked_lineup)
        std_names = [p["name"] for p in std_lineup]
        stk_names = [p["name"] for p in stacked_lineup]
        if std_names != stk_names or any(p.get("stack") for p in stacked_lineup):
            print_lineup_table(game_date, stacked_lineup, base_score, title="Lineup #2: Tournament Stack (Correlated Synergy)", synergy_score=syn_score)


if __name__ == "__main__":
    main()
