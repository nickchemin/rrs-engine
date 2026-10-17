import os
import sys
import argparse
import itertools
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from db import Database
from nhl_api import NHLClient
from model import (
    PerformancePredictor,
    evaluate_model,
    parse_toi,
    calculate_xrrs_projection,
    calculate_rrs_from_stats
)

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

SLOT_BOOSTS = [2.0, 1.8, 1.6, 1.4, 1.2]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOSTERS_FILE = os.path.join(BASE_DIR, "boosters.txt")


def print_quant_banner():
    print(
        "  ██████╗ ██████╗ ███████╗   RRS ENGINE v2.0\n"
        "  ██╔══██╗██╔══██╗██╔════╝   NHL Quant Draft Optimizer • Real Rating System\n"
    )


def parse_booster_val(val_str: str) -> float:
    if not val_str:
        return 0.0
    cleaned = str(val_str).strip().lower().replace("+", "").replace("x", "").replace("%", "")
    try:
        val = float(cleaned)
        return val / 100.0 if val > 10.0 else val
    except ValueError:
        return 0.0


def normalize_booster_name(name: str) -> str:
    """
    Normalizes player names to consistent canonical keys:
    - 'Elias Pettersson (C)', 'Elias Pettersson (F)', 'Elias Pettersson #40', or 'Elias Pettersson' -> 'Elias Pettersson (C)'
    - 'Elias Pettersson (D)', 'Elias Pettersson #25' -> 'Elias Pettersson (D)'
    """
    raw = name.strip()
    if raw.startswith("[DTD]"):
        raw = raw[5:].strip()
    lower = raw.lower()
    if "elias pettersson" in lower:
        if any(x in lower for x in ["(d)", "(def)", "(defense)", "#25"]):
            return "Elias Pettersson (D)"
        return "Elias Pettersson (C)"
    return raw


def load_boosters_file(path: str = BOOSTERS_FILE) -> Dict[str, float]:
    boosters = {}
    if not os.path.exists(path):
        return boosters

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
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
                val_raw = parts[1].strip().split("#")[0].strip()
                val = parse_booster_val(val_raw)
                canon_name = normalize_booster_name(raw_name)
                if canon_name:
                    boosters[canon_name] = val

    return boosters


def get_player_booster(
    boosters: Dict[str, float],
    name: str,
    pos: str = "",
    is_duplicate: bool = False
) -> float:
    """
    Resolves booster value with support for duplicate player names (e.g. Elias Pettersson C vs D).
    """
    if is_duplicate:
        if pos in ("D", "DEF"):
            return boosters.get(f"{name} (D)", 0.0)
        else:
            return boosters.get(f"{name} (C)", boosters.get(name, 0.0))

    return boosters.get(name, 0.0)


def save_boosters_file(
    boosters: Dict[str, float],
    active_players: List[Dict[str, Any]],
    dtd_players: Optional[List[Dict[str, Any]]] = None,
    path: str = BOOSTERS_FILE
):
    lines = [
        "# Real Sports NHL - Player Boosters",
        "# Format: Player Name = Booster (e.g. Matt Boldy = 0.8)",
        "# Note: For duplicate names, specify position: Elias Pettersson (C) = 1.4 vs Elias Pettersson (D) = 0.0",
        "# Default booster is 0.0x if player is not listed.",
        ""
    ]

    all_skaters = list(active_players) + list(dtd_players or [])
    name_counts = Counter(p["name"] for p in all_skaters)

    written_keys = set()

    # 1. Active healthy players playing today
    for p in active_players:
        name = p["name"]
        pos = p.get("pos", "")
        team = p.get("team", "")
        is_dup = name_counts[name] > 1
        key = f"{name} ({pos})" if is_dup else name
        comment = f"  # {team} {pos}" if is_dup else ""

        if key in written_keys:
            continue
        written_keys.add(key)

        b_val = get_player_booster(boosters, name, pos, is_duplicate=is_dup)
        lines.append(f"{key:<24} = {b_val:.1f}{comment}")

    # 2. Day-to-day players (commented out by default)
    if dtd_players:
        lines.append("")
        lines.append("# Day-To-Day players (uncomment if confirmed playing):")
        for p in dtd_players:
            name = p["name"]
            pos = p.get("pos", "")
            team = p.get("team", "")
            is_dup = name_counts[name] > 1
            key = f"{name} ({pos})" if is_dup else name
            detail = p.get("inj_detail") or "Day-To-Day"

            if key in written_keys:
                continue
            written_keys.add(key)

            b_val = get_player_booster(boosters, name, pos, is_duplicate=is_dup)
            lines.append(f"# [DTD] {key:<20} = {b_val:.1f}  # {team} {pos} {detail}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sync_boosters_to_db(db: Database, boosters: Dict[str, float]):
    for name, val in boosters.items():
        canon = normalize_booster_name(name)
        db.set_booster(canon, val)
    # Purge legacy/unqualified duplicate keys from SQLite DB
    try:
        with db._get_connection() as conn:
            conn.execute("DELETE FROM boosters WHERE player_name IN ('Elias Pettersson', 'Elias Pettersson (F)')")
            conn.commit()
    except Exception:
        pass


def sync_game_day(client: NHLClient, db: Database, target_date: str = "", force: bool = False) -> List[Dict[str, Any]]:
    print("Fetching schedule from NHL API...")
    schedule = client.get_upcoming_schedule(force=force)
    if not schedule:
        print("Schedule is currently unavailable.")
        return []

    if not target_date:
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_day = next((d for d in schedule if d["date"] == today_str and d.get("games")), None)
        if not today_day:
            today_day = next((d for d in schedule if d.get("games")), None)
    else:
        today_day = next((d for d in schedule if d["date"] == target_date and d.get("games")), None)

    if not today_day:
        print("No scheduled games found.")
        return []

    game_date = today_day["date"]
    games = today_day.get("games", [])
    print(f"Game Day: {game_date} ({len(games)} matches)")

    injuries = client.get_injuries(force=force)
    goalies = client.get_starting_goalies(games=games, force=force)

    team_matchups = {}
    for g in games:
        home, away = g["homeTeam"], g["awayTeam"]
        team_matchups[home] = (away, True)
        team_matchups[away] = (home, False)

    existing_boosters = db.get_all_boosters()
    file_boosters = load_boosters_file()
    existing_boosters.update(file_boosters)

    candidates = []
    dtd_players = []
    excluded_count = 0

    print("Fetching club rosters and stats in parallel...")
    all_club_stats = client.get_all_club_stats(list(team_matchups.keys()))

    all_skater_names = []
    for team in team_matchups.keys():
        for p in all_club_stats.get(team, []):
            all_skater_names.append(f"{p['firstName']['default']} {p['lastName']['default']}")
    name_counts = Counter(all_skater_names)

    players_to_save = []
    for team, (opp, is_home) in team_matchups.items():
        skaters = all_club_stats.get(team, [])
        opp_goalie = goalies.get(opp, {})
        g_factor = opp_goalie.get("goalie_factor", 1.0)
        g_name = opp_goalie.get("goalie_name", "")

        for p in skaters:
            p_id = p["playerId"]
            full_name = f"{p['firstName']['default']} {p['lastName']['default']}"
            pos = p.get("positionCode", "F")
            gp = p.get("gamesPlayed", 0)
            avg_toi = p.get("avgTimeOnIcePerGame", 0.0) / 60.0
            is_dup = name_counts[full_name] > 1
            display_name = f"{full_name} ({pos})" if is_dup else full_name

            players_to_save.append({"id": p_id, "name": full_name, "team": team, "pos": pos})

            g_pg = ((p.get("goals", 0) / gp) * g_factor) if gp else 0.0
            a_pg = ((p.get("assists", 0) / gp) * (1.0 + 0.5 * (g_factor - 1.0))) if gp else 0.0
            s_pg = (p.get("shots", 0) / gp) if gp else 0.0
            blk_pg = avg_toi * (0.09 if pos in ("D", "DEF") else 0.04)
            is_d = pos in ("D", "DEF")
            est_xrrs = calculate_xrrs_projection(g_pg, a_pg, s_pg, blk_pg, is_def=is_d)

            b_val = get_player_booster(existing_boosters, full_name, pos, is_duplicate=is_dup)
            has_booster = b_val > 0.0

            # Injury check
            inj = injuries.get(full_name)
            if inj:
                status = inj.get("status")
                detail = inj.get("detail", "")
                if status in ("Out", "Injured Reserve", "Suspension"):
                    excluded_count += 1
                    continue
                elif status == "Day-To-Day" and not has_booster:
                    dtd_players.append({
                        "name": full_name,
                        "display_name": display_name,
                        "team": team,
                        "pos": pos,
                        "inj_detail": f"{detail} (Day-To-Day)" if detail else "Day-To-Day"
                    })
                    continue

            # Filter non-scorers and low TOI players (unless user has a booster for them)
            if not has_booster:
                if gp < 1 or avg_toi < 14.5:
                    continue

            candidates.append({
                "id": p_id,
                "name": full_name,
                "display_name": display_name,
                "team": team,
                "opp": f"{'vs' if is_home else '@'}{opp}",
                "opp_goalie": g_name,
                "pos": pos,
                "xRRS": est_xrrs,
                "base_pts": est_xrrs,
                "avg_toi": avg_toi,
                "booster": b_val
            })

    if players_to_save:
        db.save_players_batch(players_to_save)

    candidates.sort(key=lambda x: x["xRRS"], reverse=True)
    dtd_players.sort(key=lambda x: x["name"])

    # Save to boosters.txt
    save_boosters_file(existing_boosters, candidates, dtd_players=dtd_players)
    sync_boosters_to_db(db, existing_boosters)

    print(f"Saved {len(candidates)} active game-day players to {os.path.basename(BOOSTERS_FILE)}")
    if excluded_count > 0 or dtd_players:
        print(f"Injuries: {excluded_count} Out/IR players excluded, {len(dtd_players)} Day-To-Day players commented out.")
    if goalies:
        print(f"Goalies: {len(goalies)} starting goalies tracked.")
    return candidates


def assign_slots(players: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    slots = [2.0, 1.8, 1.6, 1.4, 1.2]
    # Rearrangement inequality: highest base xRRS gets highest slot multiplier
    sorted_players = sorted(players, key=lambda x: x.get("xRRS", x.get("base_pts", 0.0)), reverse=True)
    assigned = []
    total_score = 0.0

    team_counts = {}
    for p in sorted_players:
        team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1

    for i in range(min(5, len(sorted_players))):
        p = sorted_players[i]
        slot_b = slots[i]
        tot_b = round(slot_b + p["booster"], 2)
        p_xrrs = p.get("xRRS", p.get("base_pts", 0.0))
        score = round(p_xrrs * tot_b, 2)
        total_score += score
        stack_label = f"[{p['team']} Pair]" if team_counts[p["team"]] >= 2 else ""

        assigned.append({
            **p,
            "slot": i + 1,
            "slot_boost": slot_b,
            "total_boost": tot_b,
            "xRRS": p_xrrs,
            "score": score,
            "rrs": score,
            "stack": stack_label
        })

    return assigned, round(total_score, 2)


def solve_optimal_lineup(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    if not candidates:
        return [], 0.0
    if len(candidates) <= 5:
        return assign_slots(candidates)

    # Evaluate top 25 candidates by potential upper-bound EV (maximum slot multiplier 2.0x + booster)
    pool = sorted(candidates, key=lambda x: x.get("xRRS", x.get("base_pts", 0.0)) * (2.0 + x.get("booster", 0.0)), reverse=True)[:25]

    best_lineup: Optional[List[Dict[str, Any]]] = None
    best_score = -1.0

    for comb in itertools.combinations(pool, 5):
        assigned, score = assign_slots(list(comb))
        if score > best_score:
            best_score = score
            best_lineup = assigned

    if best_lineup is not None:
        return best_lineup, best_score
    return assign_slots(pool[:5])


def solve_stacked_lineup(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Finds the optimal correlated lineup for tournaments by identifying the best 2-man or 2+2
    teammate stacks that maximize combined upside and synergy.
    Returns: (assigned_lineup, base_score, synergy_score)
    """
    if len(candidates) < 5:
        lineup, score = solve_optimal_lineup(candidates)
        return lineup, score, score

    by_team: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_team.setdefault(c["team"], []).append(c)

    for t in by_team:
        by_team[t].sort(key=lambda x: x.get("xRRS", x.get("base_pts", 0.0)) * (1.6 + x["booster"]), reverse=True)

    viable_stacks = []
    for team, plist in by_team.items():
        if len(plist) >= 2:
            p1, p2 = plist[0], plist[1]
            # Require at least one forward to ensure real offensive goal correlation
            f_count = sum(1 for p in (p1, p2) if p.get("pos") not in ("D", "DEF"))
            if f_count == 0:
                continue

            p1_r = p1.get("xRRS", p1.get("base_pts", 0.0))
            p2_r = p2.get("xRRS", p2.get("base_pts", 0.0))
            base_val = (p1_r * (1.6 + p1["booster"])) + (p2_r * (1.6 + p2["booster"]))
            g_factor = p1.get("goalie_factor", 1.0)
            syn_val = base_val * (1.0 + 0.12 * g_factor)
            viable_stacks.append({
                "team": team,
                "players": [p1, p2],
                "base_val": base_val,
                "syn_val": syn_val
            })

    viable_stacks.sort(key=lambda x: x["syn_val"], reverse=True)

    if not viable_stacks:
        lineup, score = solve_optimal_lineup(candidates)
        return lineup, score, score

    best_lineup = None
    best_syn_score = -1.0
    best_base_score = 0.0

    # Evaluate each top stack paired with either another stack or best individual players
    for stack1 in viable_stacks[:8]:
        s1_names = {p["name"] for p in stack1["players"]}

        # Option A: Stack 1 + Stack 2 + 1 standalone
        for stack2 in viable_stacks[:8]:
            if stack2["team"] == stack1["team"]:
                continue
            s2_names = {p["name"] for p in stack2["players"]}
            combined_names = s1_names | s2_names
            remaining = [c for c in candidates if c["name"] not in combined_names]
            if remaining:
                remaining.sort(key=lambda x: x.get("xRRS", x.get("base_pts", 0.0)) * (1.6 + x["booster"]), reverse=True)
                five_players = stack1["players"] + stack2["players"] + [remaining[0]]
                assigned, base_sc = assign_slots(five_players)
                synergy_sc = round(base_sc + (stack1["syn_val"] - stack1["base_val"]) + (stack2["syn_val"] - stack2["base_val"]), 2)
                if synergy_sc > best_syn_score:
                    best_syn_score = synergy_sc
                    best_base_score = base_sc
                    best_lineup = assigned

        # Option B: Stack 1 + 3 standalone players
        remaining = [c for c in candidates if c["name"] not in s1_names]
        if len(remaining) >= 3:
            remaining.sort(key=lambda x: x.get("xRRS", x.get("base_pts", 0.0)) * (1.6 + x["booster"]), reverse=True)
            five_players = stack1["players"] + remaining[:3]
            assigned, base_sc = assign_slots(five_players)
            synergy_sc = round(base_sc + (stack1["syn_val"] - stack1["base_val"]), 2)
            if synergy_sc > best_syn_score:
                best_syn_score = synergy_sc
                best_base_score = base_sc
                best_lineup = assigned

    if not best_lineup:
        lineup, score = solve_optimal_lineup(candidates)
        return lineup, score, score

    return best_lineup, best_base_score, best_syn_score


def print_lineup_table(
    date_str: str,
    lineup: List[Dict[str, Any]],
    total_score: float,
    title: str = "Standard Optimal (Max Base EV)",
    synergy_score: Optional[float] = None
):
    has_stacks = any(item.get("stack") for item in lineup)
    width = 110 if has_stacks else 99

    print("\n" + "=" * width)
    header = f"Date: {date_str} | Projected: {total_score:.1f} RRS | Mode: {title}"
    if synergy_score and synergy_score > total_score:
        header += f" | Upside: {synergy_score:.1f} RRS"
    print(header)
    print("=" * width)

    if has_stacks:
        print(f"{'Slot':<5} {'Slot Boost':<11} {'Player':<22} {'Team':<5} {'Opp (Goalie)':<22} {'xRRS':<8} {'Booster':<9} {'Total':<7} {'RRS Payout':<13} {'Synergy'}")
    else:
        print(f"{'Slot':<5} {'Slot Boost':<11} {'Player':<22} {'Team':<5} {'Opp (Goalie)':<22} {'xRRS':<8} {'Booster':<9} {'Total':<7} {'RRS Payout'}")
    print("-" * width)

    for item in lineup:
        slot_str = f"#{item['slot']}"
        s_boost = f"{item['slot_boost']:.1f}x"
        p_boost = f"+{item['booster']:.1f}x"
        tot_boost = f"{item['total_boost']:.1f}x"
        p_xrrs = item.get("xRRS", item.get("base_pts", 0.0))
        score_str = f"{item['score']:.1f} RRS"
        p_name = item.get("display_name") or item["name"]
        g_name = item.get("opp_goalie", "")
        if g_name and " " in g_name:
            parts = g_name.split()
            short_g = f"{parts[0][0]}. {parts[-1]}"
        else:
            short_g = g_name
        opp_str = f"{item['opp']} ({short_g})" if short_g else item['opp']
        stack_str = item.get("stack", "")

        if has_stacks:
            print(f"{slot_str:<5} {s_boost:<11} {p_name:<22} {item['team']:<5} {opp_str:<22} {p_xrrs:<8.1f} {p_boost:<9} {tot_boost:<7} {score_str:<13} {stack_str}")
        else:
            print(f"{slot_str:<5} {s_boost:<11} {p_name:<22} {item['team']:<5} {opp_str:<22} {p_xrrs:<8.1f} {p_boost:<9} {tot_boost:<7} {score_str:<13}")

    print("-" * width)
    # Single-line Quant Summary
    top_p = max(lineup, key=lambda x: x["score"]) if lineup else None
    top_name = (top_p.get("display_name") or top_p["name"]) if top_p else ""
    stacks_found = [p['stack'] for p in lineup if p.get('stack')]
    if stacks_found:
        unique_stacks = sorted(list(set(stacks_found)))
        print(f"⚡ Quant Summary: Top EV driver {top_name} ({top_p['total_boost']:.1f}x Boost) | Synergistic {', '.join(unique_stacks)} upside stack")
    elif top_p:
        print(f"⚡ Quant Summary: Anchor play {top_name} ({top_p['total_boost']:.1f}x Boost) | Projected Lineup Total: {total_score:.1f} RRS")
    print("=" * width + "\n")


def settle_past_lineups(client: NHLClient, db: Database):
    unsettled = db.get_unsettled_history()
    if not unsettled:
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    past_unsettled = [r for r in unsettled if r["game_date"] < today_str]
    if not past_unsettled:
        return

    all_players = db.get_all_players()
    players_map = {p["player_id"]: p for p in all_players}
    players_by_name = {p["name"]: p for p in all_players}

    settled_count = 0
    for r in past_unsettled:
        p_id = r.get("player_id")
        if not p_id or p_id not in players_map:
            p_obj = players_by_name.get(r["player_name"])
            p_id = p_obj["player_id"] if p_obj else None

        if not p_id:
            continue

        logs = client.get_player_current_game_logs(p_id)
        match = next((g for g in logs if g.get("gameDate") == r["game_date"]), None)
        if match:
            g = float(match.get("goals", 0))
            a = float(match.get("assists", 0))
            s = float(match.get("shots", 0))
            toi_str = match.get("toi", "00:00")
            parts = toi_str.split(":")
            toi_min = float(parts[0]) + float(parts[1]) / 60.0 if len(parts) == 2 else 0.0
            blk_rate = 0.09 if match.get("positionCode") in ("D", "DEF") else 0.04
            blk = toi_min * blk_rate
            # Real Rating System (RRS) actual rating:
            actual_xrrs = calculate_rrs_from_stats(g, a, s, blk)
            actual_rrs = round(actual_xrrs * r["total_boost"], 2)
            db.settle_history_player(r["id"], g, a, s, toi_str, actual_xrrs, actual_rrs)
            settled_count += 1

    if settled_count > 0:
        print(f"History: settled actual results for {settled_count} players from past games.")


def show_history_report(db: Database, target_date: Optional[str] = None):
    print_quant_banner()
    if target_date:
        details = db.get_history_slate_details(target_date)
        if not details:
            print(f"No history records found for date: {target_date}")
            return

        print("\n" + "=" * 105)
        print(f"RRS Engine - Lineup Performance Detail ({target_date})")
        print("=" * 105)
        print(f"{'Mode':<12} {'Slot':<5} {'Player':<22} {'Team':<5} {'Boost':<7} {'xRRS':<8} {'Act xRRS':<9} {'Proj RRS':<10} {'Actual RRS':<12} {'Stats'}")
        print("-" * 105)

        for d in details:
            mode_str = d["mode"].capitalize()
            slot_str = f"#{d['slot']}"
            p_name = d["player_name"]
            team_str = d["team"]
            tot_b = f"{d['total_boost']:.1f}x"
            p_base = d.get('proj_xrrs', d.get('proj_base_pts', 0.0))
            proj_base = f"{p_base:.1f}"
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

            print(f"{mode_str:<12} {slot_str:<5} {p_name:<22} {team_str:<5} {tot_b:<7} {proj_base:<8} {act_base:<9} {proj_rrs:<10} {act_rrs:<12} {stats_str}")

        print("=" * 105 + "\n")
        return

    slates = db.get_history_slates()
    if not slates:
        print("No history records recorded yet. Run without arguments to generate and track today's lineups.")
        return

    print("\n" + "=" * 88)
    print("RRS Engine - Performance & Rating Tracking Summary")
    print("=" * 88)
    print(f"{'Date':<12} {'Mode':<14} {'Status':<10} {'Proj RRS':<13} {'Actual RRS':<14} {'Diff RRS'}")
    print("-" * 88)

    total_actual_rrs = 0.0
    settled_slates = 0

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

        print(f"{date_str:<12} {mode_str:<14} {status_str:<10} {proj_rrs:<13} {act_rrs:<14} {diff_str}")

    print("-" * 88)
    if settled_slates > 0:
        print(f"Total Settled: {settled_slates} slates | Cumulative Payout: {total_actual_rrs:.1f} RRS")
    else:
        print("Games are currently in progress or upcoming. Check back after matches finish for settled RRS payout.")
    print("=" * 88 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RRS Engine v2.0 - Real Rating System & NHL Quant Draft Optimizer")
    parser.add_argument("--sync", action="store_true", help="Sync game-day roster and refresh boosters.txt")
    parser.add_argument("--date", type=str, default="", help="Target date (YYYY-MM-DD), default is current day")
    parser.add_argument("--train", action="store_true", help="Run ML model validation (MAE metrics)")
    parser.add_argument("--stack", action="store_true", help="Generate only the correlated tournament stack lineup")
    parser.add_argument("--history", nargs="?", const="all", default=None, help="Show performance tracking and settled RRS history (optional: YYYY-MM-DD)")
    args = parser.parse_args()

    if not args.history:
        print_quant_banner()

    db = Database()
    client = NHLClient(db=db)
    standings = client.get_standings()

    # Settle any past games in history
    settle_past_lineups(client, db)

    if args.history is not None:
        target_d = args.history if args.history != "all" else None
        show_history_report(db, target_date=target_d)
        return

    if args.train:
        evaluate_model(db, standings)
        return

    # If --sync is passed or boosters.txt does not exist, run sync first
    if args.sync or not os.path.exists(BOOSTERS_FILE):
        sync_game_day(client, db, target_date=args.date, force=True)
        if args.sync:
            return

    # 1. Load boosters from boosters.txt into DB
    boosters = load_boosters_file(BOOSTERS_FILE)
    sync_boosters_to_db(db, boosters)

    # 2. Get schedule and target date
    schedule = client.get_upcoming_schedule()
    if not schedule:
        print("Schedule is unavailable.")
        return

    if not args.date:
        today_str = datetime.now().strftime("%Y-%m-%d")
        target_day = next((d for d in schedule if d["date"] == today_str and d.get("games")), None)
        if not target_day:
            target_day = next((d for d in schedule if d.get("games")), None)
    else:
        target_day = next((d for d in schedule if d["date"] == args.date and d.get("games")), None)

    if not target_day:
        print(f"No games found for date: {args.date or 'today'}")
        return

    game_date = target_day["date"]
    games = target_day.get("games", [])

    team_matchups = {}
    for g in games:
        home, away = g["homeTeam"], g["awayTeam"]
        team_matchups[home] = (away, True)
        team_matchups[away] = (home, False)

    # 3. Collect and predict for all active players in today's games
    injuries = client.get_injuries()
    goalies = client.get_starting_goalies(games=games)
    excluded_injuries = 0
    dtd_skipped = 0
    candidates = []

    print("Fetching club rosters and stats in parallel...")
    all_club_stats = client.get_all_club_stats(list(team_matchups.keys()))

    all_skater_names = []
    for team in team_matchups.keys():
        for p in all_club_stats.get(team, []):
            all_skater_names.append(f"{p['firstName']['default']} {p['lastName']['default']}")
    name_counts = Counter(all_skater_names)

    qualifying_skaters = []
    players_to_save = []

    for team, (opp, is_home) in team_matchups.items():
        skaters = all_club_stats.get(team, [])
        opp_goalie = goalies.get(opp, {})
        g_factor = opp_goalie.get("goalie_factor", 1.0)
        g_name = opp_goalie.get("goalie_name", "")

        for p in skaters:
            p_id = p["playerId"]
            full_name = f"{p['firstName']['default']} {p['lastName']['default']}"
            pos = p.get("positionCode", "F")
            gp = p.get("gamesPlayed", 0)
            avg_toi = p.get("avgTimeOnIcePerGame", 0.0) / 60.0
            is_dup = name_counts[full_name] > 1
            display_name = f"{full_name} ({pos})" if is_dup else full_name

            players_to_save.append({"id": p_id, "name": full_name, "team": team, "pos": pos})
            b_val = get_player_booster(boosters, full_name, pos, is_duplicate=is_dup)

            # Injury check
            inj = injuries.get(full_name)
            if inj:
                status = inj.get("status")
                if status in ("Out", "Injured Reserve", "Suspension"):
                    excluded_injuries += 1
                    continue
                elif status == "Day-To-Day" and b_val == 0.0:
                    dtd_skipped += 1
                    continue

            # Fast filter: skip non-scoring depth players if no custom booster
            if b_val == 0.0 and (gp < 1 or avg_toi < 14.5):
                continue

            qualifying_skaters.append({
                "id": p_id,
                "name": full_name,
                "display_name": display_name,
                "team": team,
                "opp": opp,
                "is_home": is_home,
                "opp_goalie": g_name,
                "goalie_factor": g_factor,
                "pos": pos,
                "booster": b_val
            })

    if players_to_save:
        db.save_players_batch(players_to_save)

    if excluded_injuries > 0 or dtd_skipped > 0:
        print(f"Injuries: excluded {excluded_injuries} Out/IR players, skipped {dtd_skipped} Day-To-Day players without booster.")
    if goalies:
        print(f"Goalies: matched {len(goalies)} starting goalies for today's games.")

    # Prefetch game logs in parallel for all qualifying skaters
    print(f"Synchronizing game logs for {len(qualifying_skaters)} qualifying skaters (in parallel)...")
    qualifying_ids = [s["id"] for s in qualifying_skaters]
    logs_map = client.prefetch_players_game_logs(qualifying_ids, max_workers=8)

    total_qual = len(qualifying_skaters)
    for idx, s in enumerate(qualifying_skaters, 1):
        p_id = s["id"]
        logs = logs_map.get(p_id) or []
        if not logs:
            continue

        if idx % 5 == 0 or idx == total_qual:
            sys.stdout.write(f"\rTraining ML models & projecting xRRS: [{idx}/{total_qual}]...")
            sys.stdout.flush()

        predictor = PerformancePredictor()
        predictor.train(logs, standings)
        pred = predictor.predict(
            game_logs=logs,
            is_home=s["is_home"],
            opponent_abbrev=s["opp"],
            game_date=game_date,
            standings=standings,
            position=s["pos"],
            goalie_factor=s["goalie_factor"]
        )

        candidates.append({
            "id": p_id,
            "name": s["name"],
            "display_name": s.get("display_name", s["name"]),
            "team": s["team"],
            "opp": f"{'vs' if s['is_home'] else '@'}{s['opp']}",
            "opp_goalie": s["opp_goalie"],
            "pos": s["pos"],
            "xRRS": pred["xRRS"],
            "base_pts": pred["base_pts"],
            "booster": s["booster"]
        })

    if total_qual > 0:
        sys.stdout.write(f"\rTraining ML models & projecting xRRS: completed for {len(candidates)} active skaters.      \n")
        sys.stdout.flush()

    # 4. Solve optimal 5-slot lineup(s)
    if args.stack:
        stacked_lineup, base_score, syn_score = solve_stacked_lineup(candidates)
        db.save_lineup_history(game_date, "stack", stacked_lineup)
        print_lineup_table(game_date, stacked_lineup, base_score, title="Tournament Stack (Correlated Upside)", synergy_score=syn_score)
    else:
        std_lineup, std_score = solve_optimal_lineup(candidates)
        db.save_lineup_history(game_date, "standard", std_lineup)
        print_lineup_table(game_date, std_lineup, std_score, title="Lineup #1: Standard Optimal (Max Base EV)")

        stacked_lineup, base_score, syn_score = solve_stacked_lineup(candidates)
        db.save_lineup_history(game_date, "stack", stacked_lineup)
        std_names = [p["name"] for p in std_lineup]
        stk_names = [p["name"] for p in stacked_lineup]
        if std_names != stk_names or any(p.get("stack") for p in stacked_lineup):
            print_lineup_table(game_date, stacked_lineup, base_score, title="Lineup #2: Tournament Stack (Correlated Synergy)", synergy_score=syn_score)


if __name__ == "__main__":
    main()
