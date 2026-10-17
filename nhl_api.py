import requests
import json
import re
import os
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from db import Database, season_from_date


def nhl_now() -> datetime:
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        tz = timezone(timedelta(hours=-5))
    return datetime.now(tz)


def nhl_today() -> str:
    return nhl_now().strftime("%Y-%m-%d")


def current_season_start_year() -> int:
    now = nhl_now()
    return now.year if now.month >= 8 else now.year - 1


class NHLClient:
    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        adapter = HTTPAdapter(
            pool_connections=25,
            pool_maxsize=25,
            max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._boxscore_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._boxscore_lock = threading.Lock()

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
                    gp = s.get("gamesPlayed", 0) or 0
                    if gp == 0:
                        metrics[team] = {
                            "ga_per_game": 3.0,
                            "defense_factor": 1.0
                        }
                        continue
                    ga = s.get("goalAgainst", 0)
                    ga_per_game = ga / gp
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

    def _fetch_game_log(self, player_id: int, season: str) -> List[Dict[str, Any]]:
        url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2"
        try:
            res = self.session.get(url, timeout=6)
            if res.status_code == 200:
                items = res.json().get("gameLog", [])
                valid = []
                for g in items:
                    if g.get("gameTypeId") == 1:
                        continue
                    if not g.get("seasonId"):
                        g["seasonId"] = season or season_from_date(g.get("gameDate", ""))
                    valid.append(g)
                return valid
        except Exception:
            pass
        return []

    def get_player_game_logs(
        self,
        player_id: int,
        seasons: Optional[List[str]] = None,
        force: bool = False
    ) -> List[Dict[str, Any]]:
        if not force and self.db.has_game_logs(player_id):
            cached_logs = self.db.get_game_logs(player_id)
            latest_date = cached_logs[-1].get("gameDate", "") if cached_logs else ""
            cutoff_date = f"{current_season_start_year() - 1}-09-01"
            if latest_date >= cutoff_date:
                # Fast check: sync newly completed games from the current season
                try:
                    url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/now"
                    res = self.session.get(url, timeout=6)
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("gameTypeId") != 1:
                            current_logs = data.get("gameLog", [])
                            valid_logs = []
                            for g in current_logs:
                                if g.get("gameTypeId") == 1:
                                    continue
                                if not g.get("seasonId"):
                                    g["seasonId"] = season_from_date(g.get("gameDate", ""))
                                valid_logs.append(g)
                            if valid_logs:
                                self.db.save_game_logs(player_id, valid_logs)
                except Exception:
                    pass
                return self.db.get_game_logs(player_id)

        # Check legacy .cache folder if present - only if DB has no logs
        cache_file = os.path.join(os.path.dirname(__file__), ".cache", f"game_logs_{player_id}.json")
        if not force and not self.db.has_game_logs(player_id) and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    legacy_logs = json.load(f)
                    if legacy_logs:
                        valid_legacy = []
                        for g in legacy_logs:
                            if g.get("gameTypeId") == 1:
                                continue
                            if not g.get("seasonId"):
                                g["seasonId"] = season_from_date(g.get("gameDate", ""))
                            valid_legacy.append(g)
                        if valid_legacy:
                            self.db.save_game_logs(player_id, valid_legacy)
                            return self.db.get_game_logs(player_id)
            except Exception:
                pass

        if seasons is None:
            curr_start = current_season_start_year()
            candidates = [
                f"{curr_start}{curr_start+1}",
                f"{curr_start-1}{curr_start}",
                f"{curr_start-2}{curr_start-1}"
            ]
            now = nhl_now()
            seasons = []
            for s in candidates:
                s_year = int(s[:4])
                if now.year < s_year or (now.year == s_year and now.month < 9):
                    continue
                seasons.append(s)

        all_logs = []
        with ThreadPoolExecutor(max_workers=min(max(len(seasons), 1), 4)) as executor:
            future_to_season = {executor.submit(self._fetch_game_log, player_id, s): s for s in seasons}
            for future in as_completed(future_to_season):
                logs = future.result()
                if logs:
                    all_logs.extend(logs)

        all_logs.sort(key=lambda x: x.get("gameDate", ""))
        if all_logs:
            self.db.save_game_logs(player_id, all_logs)
        return self.db.get_game_logs(player_id)

    def get_player_current_game_logs(self, player_id: int) -> Optional[List[Dict[str, Any]]]:
        url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/now"
        try:
            res = self.session.get(url, timeout=6)
            if res.status_code == 200:
                data = res.json()
                if data.get("gameTypeId") == 1:
                    return []
                logs = data.get("gameLog", [])
                valid_logs = []
                for g in logs:
                    if g.get("gameTypeId") == 1:
                        continue
                    if not g.get("seasonId"):
                        g["seasonId"] = season_from_date(g.get("gameDate", ""))
                    valid_logs.append(g)
                if valid_logs:
                    self.db.save_game_logs(player_id, valid_logs)
                return valid_logs
            return None
        except Exception:
            return None

    def _fetch_team_skaters(self, team_abbrev: str) -> List[Dict[str, Any]]:
        skaters = self.get_club_stats(team_abbrev)
        if not skaters:
            return []
        roster = self.get_team_roster(team_abbrev)
        if roster:
            roster_ids = {p.get("id") for p in roster if p.get("id")}
            if roster_ids:
                skaters = [s for s in skaters if s.get("playerId") in roster_ids]
        return skaters

    def get_all_club_stats(self, team_abbrevs: List[str], max_workers: int = 8) -> Dict[str, List[Dict[str, Any]]]:
        unique_teams = list(dict.fromkeys(team_abbrevs))
        raw_results: Dict[str, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_team = {executor.submit(self._fetch_team_skaters, t): t for t in unique_teams}
            for future in as_completed(future_to_team):
                t = future_to_team[future]
                try:
                    raw_results[t] = future.result()
                except Exception as e:
                    print(f"Failed to fetch team skaters for {t}: {e}")
                    raw_results[t] = []

        results: Dict[str, List[Dict[str, Any]]] = {}
        seen_pids = set()
        for t in unique_teams:
            filtered = []
            for s in raw_results.get(t, []):
                pid = s.get("playerId")
                if pid:
                    if pid in seen_pids:
                        continue
                    seen_pids.add(pid)
                filtered.append(s)
            results[t] = filtered
        return results

    def prefetch_players_game_logs(self, player_ids: List[int], max_workers: int = 8) -> Dict[int, List[Dict[str, Any]]]:
        unique_ids = list(set(player_ids))
        results: Dict[int, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(self.get_player_game_logs, pid): pid for pid in unique_ids}
            for future in as_completed(future_to_id):
                pid = future_to_id[future]
                try:
                    results[pid] = future.result()
                except Exception:
                    results[pid] = []
        return results

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

    def get_injuries(self, force: bool = False) -> Dict[Tuple[str, str], Dict[str, str]]:
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
                        injuries_map[(name, team)] = {
                            "name": name,
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
        force: bool = False,
        game_date: str = ""
    ) -> Dict[str, Dict[str, Any]]:
        slate_date = game_date or (games[0].get("date", "") if games else "")
        slate_teams = set()
        slate_matchups = set()
        if games:
            for g in games:
                h = g.get("homeTeam")
                a = g.get("awayTeam")
                if h:
                    slate_teams.add(h)
                if a:
                    slate_teams.add(a)
                if h and a:
                    slate_matchups.add((h, a))

        if not force:
            cached = self.db.get_starting_goalies(game_date=slate_date)
            if cached:
                if not slate_teams or slate_teams.issubset(cached.keys()):
                    for t, d in cached.items():
                        d["goalie_factor"] = compute_goalie_factor(d.get("save_pct", 0.902))
                    return cached

        goalies: Dict[str, Dict[str, Any]] = {}

        # 1. Try Daily Faceoff (confirmed live starting goalies)
        try:
            url = "https://www.dailyfaceoff.com/starting-goalies"
            res = self.session.get(url, timeout=8)
            if res.status_code == 200:
                m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', res.text, re.S)
                if m:
                    data = json.loads(m.group(1))
                    df_games = data.get("props", {}).get("pageProps", {}).get("data", [])
                    for g in df_games:
                        h_slug = g.get("homeTeamSlug", "")
                        a_slug = g.get("awayTeamSlug", "")
                        h_team = TEAM_SLUG_MAP.get(h_slug)
                        a_team = TEAM_SLUG_MAP.get(a_slug)

                        if slate_matchups and (h_team, a_team) not in slate_matchups:
                            continue

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
            self.db.save_starting_goalies(goalies, game_date=slate_date)

        return goalies

    def get_boxscore_skaters(self, game_id: int) -> Dict[int, Dict[str, Any]]:
        with self._boxscore_lock:
            if game_id in self._boxscore_cache:
                return self._boxscore_cache[game_id]

        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
        skaters: Dict[int, Dict[str, Any]] = {}
        try:
            res = self.session.get(url, timeout=8)
            if res.status_code == 200:
                data = res.json()
                game_state = data.get("gameState")
                if game_state in ("OFF", "FINAL"):
                    pstats = data.get("playerByGameStats", {})
                    for team_key in ("homeTeam", "awayTeam"):
                        team_data = pstats.get(team_key, {})
                        for p in team_data.get("forwards", []) + team_data.get("defense", []):
                            pid = p.get("playerId")
                            if pid:
                                skaters[pid] = {
                                    "playerId": pid,
                                    "goals": p.get("goals", 0) or 0,
                                    "assists": p.get("assists", 0) or 0,
                                    "sog": p.get("sog", 0) or 0,
                                    "blockedShots": p.get("blockedShots", 0) or 0,
                                    "toi": p.get("toi", "00:00") or "00:00"
                                }
                    with self._boxscore_lock:
                        self._boxscore_cache[game_id] = skaters
                    return skaters
        except Exception as e:
            print(f"Failed to fetch boxscore for game {game_id}: {e}")
        return {}


TEAM_SLUG_MAP = {
    "anaheim-ducks": "ANA", "boston-bruins": "BOS", "buffalo-sabres": "BUF", "calgary-flames": "CGY",
    "carolina-hurricanes": "CAR", "chicago-blackhawks": "CHI", "colorado-avalanche": "COL", "columbus-blue-jackets": "CBJ",
    "dallas-stars": "DAL", "detroit-red-wings": "DET", "edmonton-oilers": "EDM", "florida-panthers": "FLA",
    "los-angeles-kings": "LAK", "minnesota-wild": "MIN", "montreal-canadiens": "MTL", "nashville-predators": "NSH",
    "new-jersey-devils": "NJD", "new-york-islanders": "NYI", "new-york-rangers": "NYR", "ottawa-senators": "OTT",
    "philadelphia-flyers": "PHI", "pittsburgh-penguins": "PIT", "san-jose-sharks": "SJS", "seattle-kraken": "SEA",
    "st-louis-blues": "STL", "tampa-bay-lightning": "TBL", "toronto-maple-leafs": "TOR", "utah-hockey-club": "UTA",
    "utah-mammoth": "UTA", "utah": "UTA", "vancouver-canucks": "VAN", "vegas-golden-knights": "VGK",
    "washington-capitals": "WSH", "winnipeg-jets": "WPG"
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
