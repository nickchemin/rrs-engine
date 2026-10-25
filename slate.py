from typing import Any, Dict, List, Optional, Tuple


def build_team_matchups(games: List[Dict[str, Any]]) -> Dict[str, Tuple[str, bool]]:
    matchups: Dict[str, Tuple[str, bool]] = {}
    for game in games:
        home = game["homeTeam"]
        away = game["awayTeam"]
        matchups[home] = (away, True)
        matchups[away] = (home, False)
    return matchups


def select_game_day(
    schedule: List[Dict[str, Any]],
    target_date: str,
    today: str,
) -> Optional[Dict[str, Any]]:
    if target_date:
        return next(
            (day for day in schedule if day["date"] == target_date and day.get("games")),
            None,
        )
    day = next(
        (day for day in schedule if day["date"] == today and day.get("games")),
        None,
    )
    if day:
        return day
    return next(
        (day for day in schedule if day["date"] >= today and day.get("games")),
        None,
    )


def cached_player_row(player: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": player.get("name", ""),
        "team": player.get("team") or "",
        "pos": player.get("pos") or player.get("position") or "F",
        "id": player.get("player_id"),
        "is_duplicate": False,
    }
