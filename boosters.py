import os
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from db import Database
from paths import boosters_path

BOOSTERS_FILE = str(boosters_path())

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

