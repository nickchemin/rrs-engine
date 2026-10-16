import requests
import json
import re
from typing import Dict, List, Any, Optional
from db import Database


class NHLClient:
    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def get_standings(self, force: bool = False) -> Dict[str, Dict[str, float]]:
        if not force:
            cached = self.db.get_standings()
            if cached:
                return cached

        url = "https://api-web.nhle.com/v1/standings/now"
        metrics = {}
        try:
            res = self.session.get(url, timeout=10)
            if res.status_code == 200:
                standings = res.json().get("standings", [])
                total_ga = 0
                total_gp = 0
                team_ga_gp = {}

                for s in standings:
                    team = s.get("teamAbbrev", {}).get("default")
                    gp = s.get("gamesPlayed", 1) or 1
                    ga = s.get("goalAgainst", 0)
                    ga_per_game = ga / gp if gp > 0 else 3.0
                    team_ga_gp[team] = ga_per_game
                    total_ga += ga
                    total_gp += gp

                league_avg_ga = (total_ga / total_gp) if total_gp > 0 else 3.0
                for team, ga_pg in team_ga_gp.items():
                    ratio = ga_pg / league_avg_ga if league_avg_ga > 0 else 1.0
                    metrics[team] = {
                        "ga_per_game": round(ga_pg, 2),
                        "defense_factor": round(ratio, 3)
                    }

                self.db.save_standings(metrics)
        except Exception as e:
            print(f"Failed to fetch standings: {e}")

        return metrics

    def get_upcoming_schedule(self, force: bool = False) -> List[Dict[str, Any]]:
        if not force:
            cached = self.db.get_schedule()
            if cached:
                return cached

        url = "https://api-web.nhle.com/v1/schedule/now"
        schedule = []
        try:
            res = self.session.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                for day in data.get("gameWeek", []):
                    date_str = day.get("date")
                    games = []
                    for g in day.get("games", []):
                        games.append({
                            "gameId": g.get("id"),
                            "date": date_str,
                            "startTimeUTC": g.get("startTimeUTC"),
                            "homeTeam": g.get("homeTeam", {}).get("abbrev"),
                            "awayTeam": g.get("awayTeam", {}).get("abbrev"),
                            "gameState": g.get("gameState")
                        })
                    schedule.append({
                        "date": date_str,
                        "dayOfWeek": day.get("dayAbbrev"),
                        "games": games
                    })
                self.db.save_schedule(schedule)
        except Exception as e:
            print(f"Failed to fetch schedule: {e}")

        return schedule

    def get_player_game_logs(
        self,
        player_id: int,
        seasons: Optional[List[str]] = None,
        force: bool = False
    ) -> List[Dict[str, Any]]:
        if not force and self.db.has_game_logs(player_id):
            return self.db.get_game_logs(player_id)

        # Check legacy .cache folder if present
        import os, json
        cache_file = os.path.join(os.path.dirname(__file__), ".cache", f"game_logs_{player_id}.json")
        if not force and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    legacy_logs = json.load(f)
                    if legacy_logs:
                        self.db.save_game_logs(player_id, legacy_logs)
                        return self.db.get_game_logs(player_id)
            except Exception:
                pass

        if seasons is None:
            seasons = ["20232024", "20242025", "20252026", "20262027"]

        all_logs = []
        for season in seasons:
            url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2"
            try:
                res = self.session.get(url, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    for g in data.get("gameLog", []):
                        g["seasonId"] = season
                        all_logs.append(g)
            except Exception:
                pass

        all_logs.sort(key=lambda x: x.get("gameDate", ""))
        if all_logs:
            self.db.save_game_logs(player_id, all_logs)
        return self.db.get_game_logs(player_id)

    def get_player_current_game_logs(self, player_id: int) -> List[Dict[str, Any]]:
        url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/now"
        try:
            res = self.session.get(url, timeout=8)
            if res.status_code == 200:
                data = res.json()
                logs = data.get("gameLog", [])
                if logs:
                    self.db.save_game_logs(player_id, logs)
                return logs
        except Exception:
            pass
        return self.get_player_game_logs(player_id)

    def get_club_stats(self, team_abbrev: str) -> List[Dict[str, Any]]:
        url = f"https://api-web.nhle.com/v1/club-stats/{team_abbrev}/now"
        try:
            res = self.session.get(url, timeout=8)
            if res.status_code == 200:
                data = res.json()
                return data.get("skaters", [])
        except Exception as e:
            print(f"Failed to fetch club stats for {team_abbrev}: {e}")
        return []

    def get_team_roster(self, team_abbrev: str) -> List[Dict[str, Any]]:
        url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
        try:
            res = self.session.get(url, timeout=8)
            if res.status_code == 200:
                data = res.json()
                return data.get("forwards", []) + data.get("defensemen", [])
        except Exception as e:
            print(f"Failed to fetch roster for {team_abbrev}: {e}")
        return []

    def get_injuries(self, force: bool = False) -> Dict[str, Dict[str, str]]:
        if not force:
            cached = self.db.get_injuries()
            if cached:
                return cached

        url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
        injuries_list = []
        injuries_map = {}
        try:
            res = self.session.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                for team_item in data.get("injuries", []):
                    for inj in team_item.get("injuries", []):
                        ath = inj.get("athlete", {})
                        name = ath.get("displayName")
                        if not name:
                            continue
                        team = ath.get("team", {}).get("abbreviation", "")
                        status = inj.get("status", "Out")
                        detail = inj.get("details", {}).get("type", "")
                        injuries_list.append({
                            "name": name,
                            "team": team,
                            "status": status,
                            "detail": detail
                        })
                        injuries_map[name] = {
                            "team": team,
                            "status": status,
                            "detail": detail
                        }
                if injuries_list:
                    self.db.save_injuries(injuries_list)
        except Exception as e:
            print(f"Failed to fetch injury report: {e}")

        return injuries_map

    def get_starting_goalies(
        self,
        games: Optional[List[Dict[str, Any]]] = None,
        force: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        if not force:
            cached = self.db.get_starting_goalies()
            if cached:
                # Augment cached with computed goalie_factor
                for t, d in cached.items():
                    d["goalie_factor"] = compute_goalie_factor(d.get("save_pct", 0.902))
                return cached

        goalies: Dict[str, Dict[str, Any]] = {}

        # 1. Try Daily Faceoff (confirmed live starting goalies)
        try:
            url = "https://www.dailyfaceoff.com/starting-goalies"
            res = self.session.get(url, timeout=8)
            if res.status_code == 200:
                m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', res.text)
                if m:
                    data = json.loads(m.group(1))
                    df_games = data.get("props", {}).get("pageProps", {}).get("data", [])
                    for g in df_games:
                        h_slug = g.get("homeTeamSlug", "")
                        h_team = TEAM_SLUG_MAP.get(h_slug)
                        if h_team and g.get("homeGoalieName"):
                            sv = float(g.get("homeGoalieSavePercentage") or 0.902)
                            gaa = float(g.get("homeGoalieGoalsAgainstAvg") or 3.0)
                            status = g.get("homeNewsStrengthName") or "Projected"
                            goalies[h_team] = {
                                "goalie_name": g["homeGoalieName"],
                                "status": status,
                                "save_pct": sv,
                                "gaa": gaa,
                                "goalie_factor": compute_goalie_factor(sv)
                            }

                        a_slug = g.get("awayTeamSlug", "")
                        a_team = TEAM_SLUG_MAP.get(a_slug)
                        if a_team and g.get("awayGoalieName"):
                            sv = float(g.get("awayGoalieSavePercentage") or 0.902)
                            gaa = float(g.get("awayGoalieGoalsAgainstAvg") or 3.0)
                            status = g.get("awayNewsStrengthName") or "Projected"
                            goalies[a_team] = {
                                "goalie_name": g["awayGoalieName"],
                                "status": status,
                                "save_pct": sv,
                                "gaa": gaa,
                                "goalie_factor": compute_goalie_factor(sv)
                            }
        except Exception:
            pass

        # 2. Fallback to NHL API gamecenter landing for missing teams
        if games:
            for g in games:
                h_team = g.get("homeTeam")
                a_team = g.get("awayTeam")
                g_id = g.get("gameId")
                if (h_team not in goalies or a_team not in goalies) and g_id:
                    try:
                        res = self.session.get(f"https://api-web.nhle.com/v1/gamecenter/{g_id}/landing", timeout=8)
                        if res.status_code == 200:
                            gc = res.json().get("matchup", {}).get("goalieComparison", {})
                            if h_team and h_team not in goalies:
                                h_leaders = gc.get("homeTeam", {}).get("leaders", [])
                                if h_leaders:
                                    lead = h_leaders[0]
                                    sv = float(lead.get("savePctg") or 0.902)
                                    gaa = float(lead.get("gaa") or 3.0)
                                    name = f"{lead.get('firstName', {}).get('default', '')} {lead.get('lastName', {}).get('default', '')}".strip()
                                    goalies[h_team] = {
                                        "goalie_name": name or "Starting Goalie",
                                        "status": "Projected",
                                        "save_pct": sv,
                                        "gaa": gaa,
                                        "goalie_factor": compute_goalie_factor(sv)
                                    }
                            if a_team and a_team not in goalies:
                                a_leaders = gc.get("awayTeam", {}).get("leaders", [])
                                if a_leaders:
                                    lead = a_leaders[0]
                                    sv = float(lead.get("savePctg") or 0.902)
                                    gaa = float(lead.get("gaa") or 3.0)
                                    name = f"{lead.get('firstName', {}).get('default', '')} {lead.get('lastName', {}).get('default', '')}".strip()
                                    goalies[a_team] = {
                                        "goalie_name": name or "Starting Goalie",
                                        "status": "Projected",
                                        "save_pct": sv,
                                        "gaa": gaa,
                                        "goalie_factor": compute_goalie_factor(sv)
                                    }
                    except Exception:
                        pass

        if goalies:
            self.db.save_starting_goalies(goalies)

        return goalies


TEAM_SLUG_MAP = {
    "anaheim-ducks": "ANA", "boston-bruins": "BOS", "buffalo-sabres": "BUF", "calgary-flames": "CGY",
    "carolina-hurricanes": "CAR", "chicago-blackhawks": "CHI", "colorado-avalanche": "COL", "columbus-blue-jackets": "CBJ",
    "dallas-stars": "DAL", "detroit-red-wings": "DET", "edmonton-oilers": "EDM", "florida-panthers": "FLA",
    "los-angeles-kings": "LAK", "minnesota-wild": "MIN", "montreal-canadiens": "MTL", "nashville-predators": "NSH",
    "new-jersey-devils": "NJD", "new-york-islanders": "NYI", "new-york-rangers": "NYR", "ottawa-senators": "OTT",
    "philadelphia-flyers": "PHI", "pittsburgh-penguins": "PIT", "san-jose-sharks": "SJS", "seattle-kraken": "SEA",
    "st-louis-blues": "STL", "tampa-bay-lightning": "TBL", "toronto-maple-leafs": "TOR", "utah-hockey-club": "UTA",
    "utah": "UTA", "vancouver-canucks": "VAN", "vegas-golden-knights": "VGK", "washington-capitals": "WSH", "winnipeg-jets": "WPG"
}


def compute_goalie_factor(save_pct: float) -> float:
    if save_pct <= 0.0:
        return 1.0
    # League average save pct ~ 0.902
    # Raw factor: ratio of goals allowed per shot
    raw = (1.0 - save_pct) / 0.098
    # Dampen by 40% towards 1.0 for sample stability
    factor = 1.0 + 0.40 * (raw - 1.0)
    return round(max(0.85, min(1.25, factor)), 3)

