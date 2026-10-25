import argparse
import logging
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from db import Database
from model import (
    POS_AVERAGES,
    PerformancePredictor,
    calculate_rrs_from_stats,
    calculate_xrrs_projection,
    calibrate_rrs_formula,
    compute_line_factor,
    evaluate_model,
)
from nhl_api import NHLClient, nhl_today
from paths import boosters_path, migrate_legacy_files
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
]

BOOSTERS_FILE = str(boosters_path())

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
        r"  ____   ____   ____     RRS ENGINE v2.6.0" "\n"
        r" |  _ \ |  _ \ / ___|    NHL Quant Draft Optimizer | Real Rating System" "\n"
        rf" | |_) || |_) |\___ \    {sub}" "\n"
        r" |  _ < |  _ <  ___) |" "\n"
        r" |_| \_\|_| \_\|____/ " "\n"
    )
    print(banner)


def parse_iso_date_arg(val: str) -> str:
    if not val:
        return ""
    try:
        dt = datetime.strptime(val, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format '{val}'. Expected YYYY-MM-DD.")


def name_key(name: str) -> str:
    if not name:
        return ""
    nk = unicodedata.normalize("NFKD", str(name))
    ascii_str = "".join(c for c in nk if not unicodedata.combining(c))
    return " ".join(ascii_str.lower().split())


def parse_booster_val(val_str: str, line_num: Optional[int] = None) -> float:
    if not val_str:
        return 0.0
    val_clean = str(val_str).split("#", 1)[0].strip()
    s = val_clean.replace(',', '.')
    m = re.match(r'^\+?(\d+\.?\d*|\.\d+)\s*([x%])?$', s)
    if not m or not m.group(1):
        loc = f"Line {line_num}: " if line_num is not None else ""
        print(f"Warning: {loc}Could not parse booster value '{val_str}', defaulting to 0.0")
        return 0.0

    num = float(m.group(1))
    unit = (m.group(2) or "").lower()

    if unit == "%":
        return round(num / 100.0, 4)
    elif unit == "x":
        if num > 5.0:
            loc = f"Line {line_num}: " if line_num is not None else ""
            print(f"Warning: {loc}large booster multiplier {num:g}x")
        return num
    else:
        if num > 5.0:
            val_pct = round(num / 100.0, 4)
            loc = f"Line {line_num}: " if line_num is not None else ""
            print(f"Warning: {loc}value {num:g} > 5.0 without unit treated as percentage ({val_pct:g})")
            return val_pct
        return num


def normalize_booster_name(name: str) -> str:
    raw = name.strip()
    if raw.upper().startswith("[DTD]"):
        raw = raw[5:].strip()
    elif raw.upper().startswith("[DEPTH]"):
        raw = raw[7:].strip()

    # Match trailing position and optional team tag in parentheses:
    # e.g. (D), (C, VAN), (DEF, NYI), (LW, EDM), (F, TOR)
    m = re.search(r'\s*\((def|defense|d|c|l|r|f|rw|lw)(?:\s*,\s*([a-zA-Z]{2,4}))?\)\s*$', raw, re.IGNORECASE)
    pos_tag = None
    team_tag = None
    if m:
        p = m.group(1).upper()
        if p in ("DEF", "DEFENSE", "D"):
            pos_tag = "D"
        elif p in ("L", "LW"):
            pos_tag = "L"
        elif p in ("R", "RW"):
            pos_tag = "R"
        elif p == "C":
            pos_tag = "C"
        elif p == "F":
            pos_tag = "F"
        if m.group(2):
            team_tag = m.group(2).upper()
        base_raw = raw[:m.start()].strip()
    else:
        base_raw = raw

    nk = name_key(base_raw)
    # Special rule for Elias Pettersson
    if "elias pettersson" in nk:
        if pos_tag:
            pos_part = pos_tag
        else:
            lower_raw = raw.lower()
            if any(x in lower_raw for x in ["(d)", "(def)", "(defense)", "#25"]):
                pos_part = "D"
            else:
                pos_part = "C"
        if team_tag:
            return f"{name_key('Elias Pettersson')} ({pos_part}, {team_tag})"
        return f"{name_key('Elias Pettersson')} ({pos_part})"

    if pos_tag and team_tag:
        return f"{nk} ({pos_tag}, {team_tag})"
    if pos_tag:
        return f"{nk} ({pos_tag})"
    return nk


_warned_ambiguous_names: Set[str] = set()


def resolve_booster(
    boosters: Dict[str, float],
    name: str,
    pos: str = "",
    team: str = "",
    is_duplicate: bool = False
) -> Tuple[float, Optional[str]]:
    base = name_key(name)
    pos_clean = pos.strip().upper() if pos else ""
    if pos_clean in ("D", "DEF"):
        pos_canon = "D"
        group_canon = "D"
    elif pos_clean in ("L", "LW"):
        pos_canon = "L"
        group_canon = "F"
    elif pos_clean in ("R", "RW"):
        pos_canon = "R"
        group_canon = "F"
    elif pos_clean == "C":
        pos_canon = "C"
        group_canon = "F"
    elif pos_clean == "F":
        pos_canon = "F"
        group_canon = "F"
    else:
        pos_canon = pos_clean
        group_canon = "F"

    is_forward = (group_canon == "F")
    team_clean = team.strip().upper() if team else ""

    if is_duplicate:
        keys_to_try = []
        if pos_canon and team_clean:
            keys_to_try.append(f"{base} ({pos_canon}, {team_clean})")
        if pos_canon:
            keys_to_try.append(f"{base} ({pos_canon})")
        if group_canon and group_canon != pos_canon:
            if team_clean:
                keys_to_try.append(f"{base} ({group_canon}, {team_clean})")
            keys_to_try.append(f"{base} ({group_canon})")
        if is_forward and "C" not in (pos_canon, group_canon):
            if team_clean:
                keys_to_try.append(f"{base} (C, {team_clean})")
            keys_to_try.append(f"{base} (C)")

        for k in keys_to_try:
            if k in boosters:
                return boosters[k], k
            if k.lower() in boosters:
                return boosters[k.lower()], k

        if base in boosters or base.lower() in boosters:
            if base not in _warned_ambiguous_names:
                _warned_ambiguous_names.add(base)
                print(f"Warning: Ambiguous booster key '{name}' matched multiple active players. Please specify position in boosters.txt.")
        return 0.0, None
    else:
        keys_to_try = [base]
        if pos_canon and team_clean:
            keys_to_try.append(f"{base} ({pos_canon}, {team_clean})")
        if pos_canon:
            keys_to_try.append(f"{base} ({pos_canon})")
        if group_canon and group_canon != pos_canon:
            if team_clean:
                keys_to_try.append(f"{base} ({group_canon}, {team_clean})")
            keys_to_try.append(f"{base} ({group_canon})")
        if is_forward and "C" not in (pos_canon, group_canon):
            if team_clean:
                keys_to_try.append(f"{base} (C, {team_clean})")
            keys_to_try.append(f"{base} (C)")

        for k in keys_to_try:
            if k in boosters:
                return boosters[k], k
            if k.lower() in boosters:
                return boosters[k.lower()], k

        return 0.0, None


def get_player_booster(
    boosters: Dict[str, float],
    name: str,
    pos: str = "",
    team: str = "",
    is_duplicate: bool = False
) -> float:
    val, _ = resolve_booster(boosters, name, pos, team=team, is_duplicate=is_duplicate)
    return val


def load_boosters_file(path: str = BOOSTERS_FILE) -> Dict[str, float]:
    boosters = {}
    if not os.path.exists(path):
        return boosters

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                clean_comment = line.lstrip("#").strip()
                if clean_comment.upper().startswith(("[DTD]", "[DEPTH]")):
                    if "=" in clean_comment:
                        p = clean_comment.split("=", 1)
                    elif ":" in clean_comment:
                        p = clean_comment.split(":", 1)
                    else:
                        p = clean_comment.rsplit(None, 1)
                    if len(p) == 2:
                        chk_name = p[0].strip()
                        chk_val = parse_booster_val(p[1], line_num=line_num)
                        if chk_val > 0.0:
                            print(f"Hint (Line {line_num}): '{chk_name}' has booster {chk_val:g} but line is commented out with '#'. Remove '#' to activate.")
                continue

            # Support '=', ':', or whitespace separation
            if "=" in line:
                parts = line.split("=", 1)
            elif ":" in line:
                parts = line.split(":", 1)
            else:
                parts = line.rsplit(None, 1)

            if len(parts) == 2:
                raw_name = parts[0].strip()
                val_raw = parts[1].split("#", 1)[0].strip()
                val = parse_booster_val(val_raw, line_num=line_num)
                canon_name = normalize_booster_name(raw_name)
                if canon_name:
                    boosters[canon_name] = val

    return boosters


def save_boosters_file(
    boosters: Dict[str, float],
    active_players: List[Dict[str, Any]],
    dtd_players: Optional[List[Dict[str, Any]]] = None,
    depth_players: Optional[List[Dict[str, Any]]] = None,
    path: Optional[str] = None
):
    target_path = path or BOOSTERS_FILE
    all_skaters = list(active_players) + list(dtd_players or []) + list(depth_players or [])
    if not all_skaters:
        print("Warning: No player statistics loaded from NHL API; boosters.txt was not modified.")
        return

    # Extract existing team sections to preserve any team that failed to load
    old_sections: Dict[str, List[str]] = {}
    if os.path.exists(target_path):
        try:
            curr_team = None
            curr_team_lines: List[str] = []
            with open(target_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    stripped = line.rstrip("\r\n")
                    m = re.match(r'^#\s+([A-Z]{2,4})\s+[—-]', stripped)
                    if m:
                        if curr_team and curr_team_lines:
                            old_sections[curr_team] = curr_team_lines
                        curr_team = m.group(1)
                        curr_team_lines = [f"# {'=' * 78}", stripped, f"# {'=' * 78}"]
                        continue
                    if curr_team:
                        if stripped.startswith(f"# {'=' * 78}"):
                            continue
                        curr_team_lines.append(stripped)
            if curr_team and curr_team_lines:
                old_sections[curr_team] = curr_team_lines
        except Exception:
            pass

    old_names = set()
    for section in old_sections.values():
        for line in section:
            if "=" in line and not line.lstrip().startswith("#"):
                old_names.add(name_key(line.split("=", 1)[0]))

    lines = [
        "# Real Sports NHL - Player Boosters",
        "# Format: Player Name = Booster (e.g. Matt Boldy = 0.8)",
        "# Note: Partitioned by team for fast post-game lookup.",
        "# Note: Values are populated from previous runs / defaults; verify and match with the app before game lock.",
        "# Note: For duplicate names, specify position: Elias Pettersson (C) = 1.4 vs Elias Pettersson (D) = 0.0",
        "# Default booster is 0.0x if player is not listed.",
        ""
    ]

    name_counts = Counter(p["name"] for p in all_skaters)
    name_pos_counts = Counter((p["name"], p.get("pos", "")) for p in all_skaters)

    all_teams = sorted(list(set(p.get("team", "") for p in all_skaters if p.get("team"))))
    written_keys = set()

    def get_player_key(p: Dict[str, Any]) -> Tuple[str, bool]:
        name = p["name"]
        pos = p.get("pos", "")
        team = p.get("team", "")
        is_pos_dup = p.get("is_pos_dup", name_pos_counts[(name, pos)] > 1)
        is_dup = p.get("is_dup", name_counts[name] > 1)
        if is_pos_dup:
            return f"{name} ({pos}, {team})", True
        elif is_dup:
            return f"{name} ({pos})", True
        else:
            return name, False

    for team in all_teams:
        team_name = NHL_TEAM_NAMES.get(team, team)
        lines.append(f"# {'=' * 78}")
        lines.append(f"# {team} — {team_name}")
        lines.append(f"# {'=' * 78}")

        # 1. Active healthy players on this team (sorted by projected xRRS descending)
        team_active = [p for p in active_players if p.get("team") == team]
        team_active.sort(key=lambda x: x.get("xRRS", x.get("base_pts", 0.0)), reverse=True)

        for p in team_active:
            name = p["name"]
            pos = p.get("pos", "")
            key, is_dup = get_player_key(p)
            ltag = p.get("line_tag") or (f"{p.get('ev_line') or '--'}/{p.get('pp_unit')}" if p.get('pp_unit') else p.get('ev_line'))
            line_comment = f" | {ltag}" if ltag and ltag != "--" else ""
            comment = f"  # {team} {pos}{line_comment}" if is_dup else (f"  # {ltag}" if ltag and ltag != "--" else "")

            if key in written_keys:
                continue
            written_keys.add(key)

            b_val = get_player_booster(boosters, name, pos, team=team, is_duplicate=is_dup)
            lines.append(f"{key:<24} = {b_val:g}{comment}")

        # 2. Day-To-Day players on this team (commented out)
        if dtd_players:
            team_dtd = [p for p in dtd_players if p.get("team") == team]
            team_dtd.sort(key=lambda x: x["name"])

            for p in team_dtd:
                name = p["name"]
                pos = p.get("pos", "")
                key, is_dup = get_player_key(p)
                detail = p.get("inj_detail") or "Day-To-Day"
                ltag = p.get("line_tag") or (f"{p.get('ev_line') or '--'}/{p.get('pp_unit')}" if p.get('pp_unit') else p.get('ev_line'))
                line_comment = f" | {ltag}" if ltag and ltag != "--" else ""

                if key in written_keys:
                    continue
                written_keys.add(key)

                b_val = get_player_booster(boosters, name, pos, team=team, is_duplicate=is_dup)
                lines.append(f"# [DTD] {key:<20} = {b_val:g}  # {team} {pos} {detail}{line_comment}")

        # 3. Depth players on this team (commented out)
        if depth_players:
            team_depth = [p for p in depth_players if p.get("team") == team]
            team_depth.sort(key=lambda x: x.get("avg_toi", 0.0), reverse=True)

            for p in team_depth:
                name = p["name"]
                pos = p.get("pos", "")
                avg_toi = p.get("avg_toi", 0.0)
                mins, secs = divmod(round(avg_toi * 60), 60)
                toi_str = f"{mins:02d}:{secs:02d}"
                ltag = p.get("line_tag") or (f"{p.get('ev_line') or '--'}/{p.get('pp_unit')}" if p.get('pp_unit') else p.get('ev_line'))
                line_comment = f" | {ltag}" if ltag and ltag != "--" else ""

                key, is_dup = get_player_key(p)
                if key in written_keys:
                    continue
                written_keys.add(key)

                b_val = get_player_booster(boosters, name, pos, team=team, is_duplicate=is_dup)
                lines.append(f"# [DEPTH] {key:<18} = {b_val:g}  # {team} {pos} {toi_str}{line_comment}")

        lines.append("")

    # Preserve any team from the old file that was not in the new stats fetch
    missing_teams = sorted([t for t in old_sections if t not in all_teams])
    for team in missing_teams:
        lines.extend(old_sections[team])
        lines.append("")

    slate_names = {name_key(p.get("name", "")) for p in all_skaters}
    preserved = {}
    for name, value in boosters.items():
        base = name_key(name.split(" (", 1)[0])
        if value and base not in slate_names and base not in old_names:
            preserved[name] = value
    if preserved:
        lines.extend([
            "",
            f"# {'=' * 78}",
            "# Preserved entries (not on the current slate)",
            f"# {'=' * 78}",
        ])
        for name in sorted(preserved):
            lines.append(f"{name:<24} = {preserved[name]:g}")

    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def sync_boosters_to_db(db: Database, boosters: Dict[str, float]):
    if not boosters:
        return
    canon_keys = set()
    for raw_name, val in boosters.items():
        canon = normalize_booster_name(raw_name)
        db.set_booster(canon, val)
        canon_keys.add(canon)

    db_boosters = db.get_all_boosters()
    keys_to_delete = [k for k in db_boosters.keys() if k not in canon_keys]
    if keys_to_delete:
        db.delete_boosters(keys_to_delete)


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

    print(f"Saved {len(candidates)} active game-day players to {os.path.basename(BOOSTERS_FILE)}")
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


def settle_past_lineups(client: NHLClient, db: Database):
    unsettled = db.get_unsettled_history()
    if not unsettled:
        return

    today_str = nhl_today()
    past_unsettled = [r for r in unsettled if r["game_date"] < today_str]
    if not past_unsettled:
        return

    all_players = db.get_all_players()
    players_map = {p["player_id"]: p for p in all_players}
    players_by_name = {p["name"]: p for p in all_players}

    logs_cache: Dict[int, Optional[List[Dict[str, Any]]]] = {}
    settled_count = 0

    for r in past_unsettled:
        if not r.get("player_id"):
            continue

        p_id = r["player_id"]
        if p_id not in players_map:
            p_obj = players_by_name.get(r["player_name"])
            p_id = p_obj["player_id"] if p_obj else p_id

        if p_id not in logs_cache:
            logs_cache[p_id] = client.get_player_current_game_logs(p_id)

        fresh_logs = logs_cache[p_id]
        if fresh_logs is None:
            # Network issue; skip settling for now
            continue

        # Merge fresh_logs and db_logs
        db_logs = db.get_game_logs(p_id) or []
        combined_logs_map = {}
        for g in db_logs:
            k = g.get("gameId") or g.get("gameDate")
            if k:
                combined_logs_map[k] = g
        for g in fresh_logs:
            k = g.get("gameId") or g.get("gameDate")
            if k:
                combined_logs_map[k] = g
        logs = list(combined_logs_map.values())

        match = next((g for g in logs if g.get("gameDate") == r["game_date"]), None)
        if match:
            game_id = match.get("gameId")
            g = float(match.get("goals", 0))
            a = float(match.get("assists", 0))
            s = float(match.get("shots", 0))
            toi_str = match.get("toi", "00:00")
            blk = float(match.get("blockedShots", 0.0) or 0.0)

            today_dt = datetime.strptime(today_str, "%Y-%m-%d")
            try:
                r_dt = datetime.strptime(r["game_date"], "%Y-%m-%d")
                days_old = (today_dt - r_dt).days
            except Exception:
                days_old = 0

            if game_id:
                box_skaters = client.get_boxscore_skaters(game_id)
                if not box_skaters:
                    if days_old < 3:
                        # Request failed or protocol empty; skip record until next run
                        continue
                    # Match is >= 3 days old and protocol is still empty: settle using game log data
                else:
                    if p_id in box_skaters:
                        p_box = box_skaters[p_id]
                        g = float(p_box.get("goals", 0))
                        a = float(p_box.get("assists", 0))
                        s = float(p_box.get("sog", 0))
                        blk = float(p_box.get("blockedShots", 0))
                        toi_str = p_box.get("toi", toi_str)
                    else:
                        g = 0.0
                        a = 0.0
                        s = 0.0
                        blk = 0.0
                        toi_str = "DNP"

            actual_xrrs = calculate_rrs_from_stats(g, a, s, blk)
            actual_rrs = round(actual_xrrs * r["total_boost"], 2)
            db.settle_history_player(r["id"], g, a, s, toi_str, actual_xrrs, actual_rrs, blocks=blk)
            settled_count += 1
        else:
            # Check for DNP (Did Not Play): slate is >= 3 days old or player has already played later
            has_later_game = any(g.get("gameDate", "") > r["game_date"] for g in logs)
            try:
                slate_dt = datetime.strptime(r["game_date"], "%Y-%m-%d")
                now_dt = datetime.strptime(today_str, "%Y-%m-%d")
                days_old = (now_dt - slate_dt).days
            except Exception:
                days_old = 0

            if days_old >= 3 or has_later_game:
                db.settle_history_player(r["id"], 0.0, 0.0, 0.0, "DNP", 0.0, 0.0, blocks=0.0)
                settled_count += 1

    if settled_count > 0:
        print(f"History: settled actual results for {settled_count} players from past games.")


def show_history_report(db: Database, target_date: Optional[str] = None):
    if target_date:
        details = db.get_history_slate_details(target_date)
        if not details:
            print(f"No history records found for date: {target_date}")
            return

        table_width = 126
        print("\n" + "=" * table_width)
        print(f"RRS Engine - Lineup Performance Detail ({target_date})")
        print("=" * table_width)
        print(f"{'Mode':<10} {'Slot':<5} {'Player':<20} {'Team':<5} {'Line':<7} {'Boost':<6} {'xRRS':<6} {'Act xR':<8} {'App RRS':<8} {'Proj RRS':<9} {'Act RRS':<10} {'App Payout':<11} {'Stats'}")
        print("-" * table_width)

        for d in details:
            mode_str = d["mode"].capitalize()[:10]
            slot_str = f"#{d['slot']}"
            p_name = d["player_name"][:19]
            team_str = d["team"][:4]
            ev_line = d.get("ev_line")
            pp_unit = d.get("pp_unit")
            if ev_line and pp_unit:
                line_str = f"{ev_line}/{pp_unit}"
            elif ev_line:
                line_str = ev_line
            elif pp_unit:
                line_str = pp_unit
            else:
                line_str = "--"
            tot_b = f"{d['total_boost']:.1f}x"
            p_base = d.get('proj_xrrs', d.get('proj_base_pts', 0.0))
            proj_base = f"{p_base:.1f}"
            app_rrs_str = f"{d['app_rrs']:.1f}" if d.get('app_rrs') is not None else "--"
            app_payout_str = f"{d['app_payout']:.1f} RRS" if d.get('app_payout') is not None else "--"

            if d["settled"]:
                a_base = d.get('actual_xrrs', d.get('actual_base_pts', 0.0))
                act_base = f"{a_base:.1f}"
                proj_rrs = f"{d.get('proj_rrs', d.get('proj_score', 0.0)):.1f} RRS"
                act_rrs = f"{d.get('actual_rrs', d.get('actual_score', 0.0)):.1f} RRS"
                stats_str = f"{int(d['actual_goals'])}G, {int(d['actual_assists'])}A, {int(d['actual_shots'])}S ({d.get('actual_toi', '')})"
            else:
                act_base = "Pending"
                proj_rrs = f"{d.get('proj_rrs', d.get('proj_score', 0.0)):.1f} RRS"
                act_rrs = "Pending"
                stats_str = "--"

            print(f"{mode_str:<10} {slot_str:<5} {p_name:<20} {team_str:<5} {line_str:<7} {tot_b:<6} {proj_base:<6} {act_base:<8} {app_rrs_str:<8} {proj_rrs:<9} {act_rrs:<10} {app_payout_str:<11} {stats_str}")

        print("-" * table_width)
        has_app_data = any(d.get("app_rrs") is not None for d in details)
        if has_app_data:
            app_payouts = [d["app_payout"] for d in details if d.get("app_payout") is not None]
            app_rrs_list = [d["app_rrs"] for d in details if d.get("app_rrs") is not None]
            tot_app_payout = sum(app_payouts) if app_payouts else None
            payout_info = f" | Total App Payout: {tot_app_payout:.1f} RRS" if tot_app_payout is not None else ""
            print(f"Ground Truth Tracking: Recorded App RRS for {len(app_rrs_list)}/{len(details)} players{payout_info}")
            if tot_app_payout is not None:
                tot_proj = sum(d.get("proj_rrs", 0.0) for d in details if d.get("app_payout") is not None)
                bias = tot_app_payout - tot_proj
                print(f"Model vs Real App Bias: {'+' if bias >= 0 else ''}{bias:.2f} RRS")
        print("=" * table_width + "\n")
        return

    slates = db.get_history_slates()
    if not slates:
        print("No history records recorded yet. Run without arguments to generate and track today's lineups.")
        return

    table_width = 100
    print("\n" + "=" * table_width)
    print("RRS Engine - Performance & Rating Tracking Summary")
    print("=" * table_width)
    print(f"{'Date':<12} {'Mode':<10} {'Status':<9} {'Proj RRS':<11} {'Act RRS':<11} {'App Payout':<12} {'Act Diff':<11} {'App Bias'}")
    print("-" * table_width)

    total_actual_rrs = 0.0
    settled_slates = 0
    total_app_payout_sum = 0.0
    app_payout_slates = 0

    for s in slates:
        date_str = s["game_date"]
        mode_str = s["mode"].capitalize()
        status_str = "Settled" if s["all_settled"] else "Pending"
        p_rrs = s.get('total_proj_rrs', s.get('total_proj_score', 0.0))
        proj_rrs = f"{p_rrs:.1f} RRS"

        if s["all_settled"]:
            a_rrs = s.get('total_actual_rrs', s.get('total_actual_score', 0.0))
            act_rrs = f"{a_rrs:.1f} RRS"
            diff_rrs = a_rrs - p_rrs
            diff_str = f"{'+' if diff_rrs >= 0 else ''}{diff_rrs:.1f} RRS"
            total_actual_rrs += a_rrs
            settled_slates += 1
        else:
            act_rrs = "Pending"
            diff_str = "--"

        if s.get("app_payout_count", 0) > 0:
            app_p = s.get("total_app_payout", 0.0)
            app_payout_str = f"{app_p:.1f} RRS"
            total_app_payout_sum += app_p
            app_payout_slates += 1
        else:
            app_payout_str = "--"

        if s.get("avg_app_bias") is not None:
            b_val = s["avg_app_bias"]
            app_bias_str = f"{'+' if b_val >= 0 else ''}{b_val:.2f} RRS"
        else:
            app_bias_str = "--"

        print(f"{date_str:<12} {mode_str:<10} {status_str:<9} {proj_rrs:<11} {act_rrs:<11} {app_payout_str:<12} {diff_str:<11} {app_bias_str}")

    print("-" * table_width)
    if settled_slates > 0 or app_payout_slates > 0:
        summary_line = f"Total Settled: {settled_slates} slates | Cumulative Payout (Heuristic): {total_actual_rrs:.1f} RRS"
        if total_app_payout_sum > 0:
            summary_line += f" | App Ground Truth: {total_app_payout_sum:.1f} RRS"
        print(summary_line)
        bias_summary = db.get_role_bias_summary()
        pp1_diffs = bias_summary.get("pp1_diffs", [])
        other_diffs = bias_summary.get("other_diffs", [])
        if pp1_diffs or other_diffs:
            pp1_str = f"{sum(pp1_diffs)/len(pp1_diffs):+.2f} RRS (n={len(pp1_diffs)})" if pp1_diffs else "N/A"
            other_str = f"{sum(other_diffs)/len(other_diffs):+.2f} RRS (n={len(other_diffs)})" if other_diffs else "N/A"
            print(f"Role Bias Tracking: PP1 Avg Bias: {pp1_str} | Non-PP1 Avg Bias: {other_str}")
    else:
        print("Games are currently in progress or upcoming. Check back after matches finish for settled RRS payout.")
    print("=" * table_width + "\n")


def load_or_fetch_standings(client: NHLClient) -> Dict[str, Dict[str, float]]:
    standings = client.get_standings()
    if standings:
        print("  * Standings defense factors loaded.")
    return standings


def parse_history_arg(val: str) -> str:
    if not val or val.lower() == "all":
        return "all"
    try:
        dt = datetime.strptime(val, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid history date '{val}'. Expected 'all' or 'YYYY-MM-DD'.")


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


def handle_record_app_scores(db: Database, record_arg: str):
    """
    Interactively or in batch records actual Real Sports mobile app ratings (app_rrs)
    and payouts (app_payout) into lineup_history for ground-truth model calibration.
    """
    try:
        if ":" in record_arg:
            date_part, scores_part = record_arg.split(":", 1)
            target_date = parse_iso_date_arg(date_part.strip())
            score_tokens = [s.strip() for s in scores_part.split(",") if s.strip()]
        else:
            target_date = parse_iso_date_arg(record_arg.strip()) if record_arg != "today" else nhl_today()
            score_tokens = []
    except (argparse.ArgumentTypeError, ValueError) as e:
        print(f"Error: {e}")
        return

    lineup = db.get_lineup_for_date(target_date, mode="standard")
    if not lineup:
        all_lineups = db.get_lineup_for_date(target_date)
        if all_lineups:
            lineup = [p for p in all_lineups if p.get("mode") == all_lineups[0].get("mode")]

    if not lineup:
        print(f"\nNo lineup history found for date: {target_date}. Generate lineups first before recording results.\n")
        return

    updates = []
    print(f"\nRecording Real Sports app ratings for {target_date}:")
    print("-" * 75)
    for idx, p in enumerate(lineup):
        p_name = p.get("player_name", "Unknown")
        slot = p.get("slot", idx + 1)
        proj_r = p.get("proj_rrs", 0.0)
        tot_b = p.get("total_boost", p.get("slot_boost", 1.0) + p.get("booster", 0.0))

        app_rrs_val = None
        app_payout_val = None

        if idx < len(score_tokens):
            token = score_tokens[idx]
            if "=" in token:
                parts = token.split("=")
                try:
                    app_rrs_val = float(parts[0].strip())
                    app_payout_val = float(parts[1].strip())
                except ValueError:
                    pass
            else:
                try:
                    app_rrs_val = float(token.strip())
                    # Do not auto-calculate payout with additive assumption to prevent circular stacking verification
                    app_payout_val = None
                except ValueError:
                    pass
        else:
            try:
                prompt_str = f"Slot #{slot} | {p_name:<20} (Proj: {proj_r:>4.1f} RRS, Boost: {tot_b:.1f}x) -> App RRS (or rrs=payout): "
                val_in = input(prompt_str).strip()
                if val_in:
                    if "=" in val_in:
                        parts = val_in.split("=")
                        app_rrs_val = float(parts[0].strip())
                        app_payout_val = float(parts[1].strip())
                    else:
                        app_rrs_val = float(val_in)
                        app_payout_val = None
            except (EOFError, KeyboardInterrupt):
                print("\nInput cancelled.")
                break
            except ValueError:
                print(f"Skipping invalid entry for {p_name}.")
                continue

        if app_rrs_val is not None:
            updates.append({
                "id": p["id"],
                "app_rrs": app_rrs_val,
                "app_payout": app_payout_val
            })
            payout_str = f" (Payout = {app_payout_val:.1f} RRS)" if app_payout_val is not None else ""
            print(f"  Slot #{slot} {p_name:<20}: App RRS = {app_rrs_val:.1f}{payout_str}")

    if updates:
        db.record_app_results(updates)
        print("-" * 75)
        print(f"Successfully recorded Real App scores for {len(updates)} players on {target_date}.\n")


def main():
    parser = argparse.ArgumentParser(description="RRS Engine v2.6.0 - Real Rating System & NHL Quant Draft Optimizer")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sync", action="store_true", help="Sync game-day roster and refresh boosters.txt")
    group.add_argument("--history", nargs="?", const="all", default=None, type=parse_history_arg, help="Show performance tracking and settled RRS history (optional: YYYY-MM-DD)")
    group.add_argument("--record", nargs="?", const="today", default=None, help="Record actual Real Sports mobile app ratings and payouts (YYYY-MM-DD or YYYY-MM-DD:s1,s2,s3,s4,s5)")
    group.add_argument("--parse-images", type=str, default=None, metavar="FOLDER", help="Parse Real Sports draft screenshots from folder to populate boosters.txt")
    parser.add_argument("--date", type=parse_iso_date_arg, default="", help="Target date (YYYY-MM-DD), default is current day")
    parser.add_argument("--train", action="store_true", help="Run ML model validation (MAE metrics)")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate RRS scoring formula weights via OLS regression on settled history")
    parser.add_argument("--stack", action="store_true", help="Generate only the correlated tournament stack lineup")
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
        handle_parse_images(args.parse_images, client, db, target_date=args.date)
        return

    if args.record is not None:
        handle_record_app_scores(db, args.record)
        return

    if args.history is not None:
        target_d = args.history if args.history != "all" else None
        show_history_report(db, target_date=target_d)
        load_or_fetch_standings(client)
        return

    if args.calibrate:
        calibrate_rrs_formula(db)
        return

    if args.train:
        evaluate_model(db, standings)
        return

    # If --sync is passed or boosters.txt does not exist, run sync first
    if args.sync or not os.path.exists(BOOSTERS_FILE):
        sync_game_day(client, db, target_date=args.date, force=True)
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
        return

    target_day = select_game_day(schedule, args.date, nhl_today())

    if not target_day:
        print(f"No games found for date: {args.date or 'today'}")
        return

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
