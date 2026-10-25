from datetime import datetime
from typing import Any, Dict, List, Optional

from db import Database
from model import calculate_rrs_from_stats
from nhl_api import NHLClient, nhl_today


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
