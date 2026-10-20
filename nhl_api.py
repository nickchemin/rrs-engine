import requests
import json
import re
import os
import threading
import unicodedata
import time
from datetime import datetime, timezone, timedelta, tzinfo
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from db import Database, season_from_date, TeamLines


class USEasternTz(tzinfo):
    """Accurate US Eastern Time fallback with Daylight Saving Time (EDT/EST)."""
    def utcoffset(self, dt):
        return timedelta(hours=-5) + self.dst(dt)

    def dst(self, dt):
        if dt is None:
            return timedelta(0)
        year = dt.year
        # 2nd Sunday in March (starts 2:00 AM)
        first_mar = datetime(year, 3, 1)
        first_sun_mar = 1 + (6 - first_mar.weekday()) % 7
        second_sun_mar = first_sun_mar + 7
        dst_start = datetime(year, 3, second_sun_mar, 2, 0)
        # 1st Sunday in November (ends 2:00 AM)
        first_nov = datetime(year, 11, 1)
        first_sun_nov = 1 + (6 - first_nov.weekday()) % 7
        dst_end = datetime(year, 11, first_sun_nov, 2, 0)

        dt_naive = dt.replace(tzinfo=None)
        if dst_start <= dt_naive < dst_end:
            return timedelta(hours=1)
        return timedelta(0)

    def tzname(self, dt):
        return "EDT" if self.dst(dt) else "EST"


def nhl_now() -> datetime:
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        tz = USEasternTz()
    return datetime.now(tz)


def nhl_today() -> str:
    return nhl_now().strftime("%Y-%m-%d")


def name_key(name: str) -> str:
    if not name:
        return ""
    nk = unicodedata.normalize("NFKD", str(name))
    ascii_str = "".join(c for c in nk if not unicodedata.combining(c))
    return " ".join(ascii_str.lower().split())


def parse_toi(val: Any) -> float:
    if not val:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip()
    if ":" in val:
        parts = val.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def current_season_start_year() -> int:
    now = nhl_now()
    return now.year if now.month >= 8 else now.year - 1


class NHLClient:
    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._local = threading.local()
        self._custom_session = None
        self._boxscore_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._boxscore_lock = threading.Lock()

    def _create_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        adapter = HTTPAdapter(
            pool_connections=25,
            pool_maxsize=25,
            max_retries=Retry(
                total=0,
                connect=1,
                read=1,
                redirect=3,
                raise_on_status=False
            )
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def request_with_retry(
        self,
        url: str,
        method: str = "GET",
        max_retries: int = 3,
        backoff_base: float = 0.5,
        timeout: float = 10.0,
        **kwargs
    ) -> requests.Response:
        """
        Executes HTTP request with exponential backoff on transient network failures
        and HTTP 429/5xx status codes: delay = backoff_base * (2 ** attempt).
        """
        last_exc = None
        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    res = self.session.get(url, timeout=timeout, **kwargs)
                else:
                    res = self.session.request(method=method, url=url, timeout=timeout, **kwargs)

                if res.status_code in (429, 500, 502, 503, 504):
                    if attempt < max_retries - 1:
                        sleep_s = backoff_base * (2 ** attempt)
                        time.sleep(sleep_s)
                        continue
                return res
            except (requests.exceptions.RequestException, TimeoutError) as e:
                last_exc = e
                if attempt < max_retries - 1:
                    sleep_s = backoff_base * (2 ** attempt)
                    time.sleep(sleep_s)
                else:
                    raise last_exc
        if last_exc:
            raise last_exc
        return res

    @property
    def session(self) -> requests.Session:
        if self._custom_session is not None:
            return self._custom_session
        if not hasattr(self._local, "session"):
            self._local.session = self._create_session()
        return self._local.session

    @session.setter
    def session(self, val: Any):
        self._custom_session = val

    def get_standings(self, force: bool = False) -> Dict[str, Dict[str, float]]:
        if not force:
            cached = self.db.get_standings()
            if cached:
                return cached

        url = "https://api-web.nhle.com/v1/standings/now"
        metrics: Dict[str, Dict[str, float]] = {}
        try:
            res = self.request_with_retry(url, timeout=10)
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
                            "defense_factor": 1.0,
                            "pk_pct": 0.80,
                            "times_shorthanded_per_game": 3.0
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
                        "defense_factor": round(ratio, 3),
                        "pk_pct": 0.80,
                        "times_shorthanded_per_game": 3.0
                    }

                # Enrich with official special teams metrics (PK% and times shorthanded)
                try:
                    curr_season = f"{current_season_start_year()}{current_season_start_year()+1}"
                    pk_url = f"https://api.nhle.com/stats/rest/en/team/penaltykill?cayenneExp=seasonId={curr_season}%20and%20gameTypeId=2"
                    pk_res = self.request_with_retry(pk_url, timeout=6)
                    if pk_res.status_code == 200:
                        pk_rows = pk_res.json().get("data", [])
                        name_to_abbrev = {name_key(slug.replace("-", " ")): t for t, slug in TEAM_TO_SLUG.items()}
                        for row in pk_rows:
                            full_name = row.get("teamFullName", "")
                            t_abbr = name_to_abbrev.get(name_key(full_name))
                            if t_abbr and t_abbr in metrics:
                                pk_pct = float(row.get("penaltyKillPct", 0.80) or 0.80)
                                ts_pg = float(row.get("timesShorthandedPerGame", 3.0) or 3.0)
                                metrics[t_abbr]["pk_pct"] = round(pk_pct, 4)
                                metrics[t_abbr]["times_shorthanded_per_game"] = round(ts_pg, 2)
                except Exception:
                    pass

                self.db.save_standings(metrics)
        except Exception as e:
            print(f"Failed to fetch standings: {e}")

        return metrics

    def get_upcoming_schedule(self, force: bool = False, max_age_hours: float = 3.0) -> List[Dict[str, Any]]:
        if not force:
            cached = self.db.get_schedule(max_age_hours=max_age_hours)
            if cached:
                return cached

        url = "https://api-web.nhle.com/v1/schedule/now"
        schedule = []
        try:
            res = self.request_with_retry(url, timeout=10)
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
            res = self.request_with_retry(url, timeout=6)
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
            curr_start = current_season_start_year()
            curr_season_start = f"{curr_start}-09-01"
            if latest_date >= curr_season_start:
                # Fast check: sync newly completed games from the current season
                try:
                    url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/now"
                    res = self.request_with_retry(url, timeout=6)
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
            elif latest_date:
                # Latest log is before start of current season: check if already checked within 24h
                p_rec = self.db.get_player(player_id)
                now_ts = time.time()
                if p_rec and p_rec.get("updated_at") and (now_ts - float(p_rec["updated_at"])) < 86400:
                    return cached_logs

                try:
                    last_season_start = int(season_from_date(latest_date)[:4])
                except Exception:
                    last_season_start = curr_start - 1
                missing_seasons = [f"{y}{y+1}" for y in range(last_season_start, curr_start + 1)]
                all_logs = []
                with ThreadPoolExecutor(max_workers=min(max(len(missing_seasons), 1), 4)) as executor:
                    future_to_season = {executor.submit(self._fetch_game_log, player_id, s): s for s in missing_seasons}
                    for future in as_completed(future_to_season):
                        logs = future.result()
                        if logs:
                            all_logs.extend(logs)
                if all_logs:
                    self.db.save_game_logs(player_id, all_logs)
                self.db.touch_player_updated_at(player_id)
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
            res = self.request_with_retry(url, timeout=6)
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
            res = self.request_with_retry(url, timeout=8)
            if res.status_code == 200:
                data = res.json()
                return data.get("skaters", [])
        except Exception as e:
            print(f"Failed to fetch club stats for {team_abbrev}: {e}")
        return []

    def get_team_roster(self, team_abbrev: str) -> List[Dict[str, Any]]:
        url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
        try:
            res = self.request_with_retry(url, timeout=8)
            if res.status_code == 200:
                data = res.json()
                return data.get("forwards", []) + data.get("defensemen", [])
        except Exception as e:
            print(f"Failed to fetch roster for {team_abbrev}: {e}")
        return []

    def get_injuries(self, force: bool = False, max_age_hours: float = 3.0) -> Dict[Tuple[str, str], Dict[str, str]]:
        if not force:
            cached = self.db.get_injuries(max_age_hours=max_age_hours)
            if cached:
                return cached

        url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
        injuries_list = []
        injuries_map = {}
        try:
            res = self.request_with_retry(url, timeout=10)
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
            res = self.request_with_retry(url, timeout=8)
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
                        res = self.request_with_retry(f"https://api-web.nhle.com/v1/gamecenter/{g_id}/landing", timeout=8)
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
            res = self.request_with_retry(url, timeout=8)
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

    def get_team_lines(self, team: str, force: bool = False) -> Dict[Any, Dict[str, Any]]:
        if not force:
            cached = self.db.get_team_lines(team)
            if cached:
                return cached

        slug = TEAM_TO_SLUG.get(team)
        if not slug:
            return TeamLines()

        url = f"https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
        lines_data = TeamLines()
        try:
            res = self.request_with_retry(url, timeout=10)
            if res.status_code == 200:
                m = re.search(r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>', res.text, re.DOTALL)
                if m:
                    data = json.loads(m.group(1))
                    comb = data.get("props", {}).get("pageProps", {}).get("combinations", {})
                    players = comb.get("players", [])
                    if not players:
                        print(f"⚠️ Warning: Daily Faceoff players data missing for {team}")
                    for p in players:
                        p_name = p.get("name")
                        if not p_name:
                            continue
                        nk = name_key(p_name)
                        pos_id = (p.get("positionIdentifier") or p.get("position") or "").strip().upper()
                        grp_id = (p.get("groupIdentifier") or "").strip().upper()
                        if pos_id in ("D", "LD", "RD") or grp_id.startswith("D"):
                            pos_group = "D"
                        elif pos_id in ("G",) or grp_id.startswith("G"):
                            pos_group = "G"
                        else:
                            pos_group = "F"

                        key = (nk, pos_group)
                        if key not in lines_data:
                            lines_data[key] = {
                                "player_name": p_name,
                                "pos_group": pos_group,
                                "ev_line": None,
                                "pp_unit": None,
                                "pk_unit": None,
                                "position": pos_id
                            }
                        entry = lines_data[key]
                        cat = (p.get("categoryIdentifier") or "").lower()
                        grp = (p.get("groupIdentifier") or "").lower()
                        if cat == "ev":
                            if grp in ("f1", "f2", "f3", "f4"):
                                entry["ev_line"] = grp.replace("f", "L").upper()
                            elif grp in ("d1", "d2", "d3", "d4"):
                                entry["ev_line"] = grp.upper()
                        elif cat == "pp":
                            if grp in ("pp1", "pp2"):
                                entry["pp_unit"] = grp.upper()
                        elif cat == "pk":
                            if grp in ("pk1", "pk2"):
                                entry["pk_unit"] = grp.upper()
                        if pos_id and not pos_id.startswith("G") and not pos_id.startswith("IR"):
                            entry["position"] = pos_id

                    # Namesake aliases for 3-word names (e.g. Elias Nils Pettersson -> Elias Pettersson)
                    for (nk_item, p_grp), entry in list(lines_data.items()):
                        words = nk_item.split()
                        if len(words) == 3:
                            alias_key = (f"{words[0]} {words[2]}", p_grp)
                            if alias_key not in lines_data:
                                lines_data[alias_key] = entry
                        if "elias nils pettersson" in nk_item and p_grp == "D":
                            lines_data[("elias pettersson", "D")] = entry

                    team_has_pp = any(v.get("pp_unit") for k, v in lines_data.items() if isinstance(v, dict))
                    lines_data._meta = {"has_pp": team_has_pp, "source": "dailyfaceoff"}

                    if lines_data:
                        self.db.save_team_lines(team, lines_data)
                else:
                    print(f"⚠️ Warning: Daily Faceoff page structure changed (no __NEXT_DATA__) for {team}")
            else:
                print(f"⚠️ Warning: Daily Faceoff returned HTTP {res.status_code} for {team}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to fetch lines for {team}: {e}")

        # If Daily Faceoff scraping failed or returned 0 players, fallback automatically to NHL API TOI profiles!
        if not lines_data:
            print(f"🔄 Triggering automatic NHL API TOI-profile fallback for {team} line combinations...")
            lines_data = self._fallback_team_lines_from_nhl_api(team)

        return lines_data

    def _fallback_team_lines_from_nhl_api(self, team: str) -> TeamLines:
        """
        Reconstructs line combinations (EV Lines L1-L4, D1-D3, PP1/PP2) from official NHL API
        skater profiles, utilizing recent 3-game TOI trends and time-on-ice deployment data.
        Guarantees that the optimizer never runs with empty lines when Daily Faceoff is down.
        """
        lines_data = TeamLines()
        skaters = self._fetch_team_skaters(team)
        if not skaters:
            skaters = self.get_club_stats(team)
        if not skaters:
            return lines_data

        curr_season = f"{current_season_start_year()}{current_season_start_year()+1}"
        toi_url = f"https://api.nhle.com/stats/rest/en/skater/timeonice?cayenneExp=seasonId={curr_season}%20and%20gameTypeId=2%20and%20teamAbbrevs=%27{team}%27"
        pid_to_toi: Dict[int, Dict[str, Any]] = {}
        try:
            res = self.request_with_retry(toi_url, timeout=6)
            if res.status_code == 200:
                for item in res.json().get("data", []):
                    if "playerId" in item:
                        pid_to_toi[item["playerId"]] = item
        except Exception:
            pass

        skater_metrics = []
        for s in skaters:
            pid = s.get("playerId")
            fn = f"{s.get('firstName', {}).get('default', '')} {s.get('lastName', {}).get('default', '')}".strip()
            if not fn:
                fn = s.get("name", "")
            pos = s.get("positionCode", "F")
            pos_group = "D" if pos in ("D", "LD", "RD", "DEF") else "F"

            # Check recent 3 games from database or API
            logs = self.db.get_game_logs(pid) if hasattr(self.db, "get_game_logs") else []
            if not logs:
                try:
                    logs = self._fetch_game_log(pid, curr_season)
                except Exception:
                    logs = []

            recent_logs = logs[-3:] if logs else []
            if recent_logs:
                recent_avg_toi = sum(parse_toi(g.get("toi", "00:00")) for g in recent_logs) / len(recent_logs)
            else:
                recent_avg_toi = float(s.get("avgTimeOnIcePerGame", 0.0) or 0.0) / 60.0

            toi_item = pid_to_toi.get(pid, {})
            ev_toi_stat = float(toi_item.get("evTimeOnIcePerGame", 0.0) or 0.0) / 60.0
            pp_toi_stat = float(toi_item.get("ppTimeOnIcePerGame", 0.0) or 0.0) / 60.0
            tot_toi_stat = float(toi_item.get("timeOnIcePerGame", 0.0) or 0.0) / 60.0

            if tot_toi_stat > 0 and recent_avg_toi > 0:
                scale = recent_avg_toi / tot_toi_stat
                ev_toi = ev_toi_stat * scale
                pp_toi = pp_toi_stat * scale
            elif ev_toi_stat > 0:
                ev_toi = ev_toi_stat
                pp_toi = pp_toi_stat
            else:
                ev_toi = recent_avg_toi * 0.85
                pp_toi = recent_avg_toi * 0.15

            skater_metrics.append({
                "pid": pid,
                "name": fn,
                "pos": pos,
                "pos_group": pos_group,
                "ev_toi": ev_toi,
                "pp_toi": pp_toi,
                "total_toi": recent_avg_toi
            })

        forwards = [x for x in skater_metrics if x["pos_group"] == "F"]
        defense = [x for x in skater_metrics if x["pos_group"] == "D"]

        forwards.sort(key=lambda x: x["ev_toi"], reverse=True)
        defense.sort(key=lambda x: x["ev_toi"], reverse=True)

        ev_assignments: Dict[int, str] = {}
        for idx, f in enumerate(forwards):
            line_idx = min(4, (idx // 3) + 1)
            ev_assignments[f["pid"]] = f"L{line_idx}"

        for idx, d in enumerate(defense):
            pair_idx = min(3, (idx // 2) + 1)
            ev_assignments[d["pid"]] = f"D{pair_idx}"

        # Power play assignments: top 5 by PP TOI -> PP1, next 5 -> PP2
        pp_candidates = sorted(skater_metrics, key=lambda x: x["pp_toi"], reverse=True)
        pp_assignments: Dict[int, str] = {}
        for idx, p in enumerate(pp_candidates):
            if idx < 5 and p["pp_toi"] >= 0.8:
                pp_assignments[p["pid"]] = "PP1"
            elif idx < 10 and p["pp_toi"] >= 0.4:
                pp_assignments[p["pid"]] = "PP2"

        for x in skater_metrics:
            nk = name_key(x["name"])
            key = (nk, x["pos_group"])
            lines_data[key] = {
                "player_name": x["name"],
                "pos_group": x["pos_group"],
                "ev_line": ev_assignments.get(x["pid"]),
                "pp_unit": pp_assignments.get(x["pid"]),
                "pk_unit": None,
                "position": x["pos"]
            }

        team_has_pp = any(v.get("pp_unit") for k, v in lines_data.items() if isinstance(v, dict))
        lines_data._meta = {"has_pp": team_has_pp, "source": "nhl_api_fallback"}

        if lines_data:
            self.db.save_team_lines(team, lines_data)

        return lines_data

    def prefetch_teams_lines(
        self,
        teams: List[str],
        max_workers: int = 6,
        force: bool = False
    ) -> Dict[str, Dict[Any, Dict[str, Any]]]:
        if not teams:
            return {}
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        missing_teams: List[str] = []

        if not force:
            cached_map = self.db.get_all_lines_for_teams(teams)
            for t in teams:
                if t in cached_map and cached_map[t]:
                    result[t] = cached_map[t]
                else:
                    missing_teams.append(t)
        else:
            missing_teams = list(teams)

        if not missing_teams:
            return result

        with ThreadPoolExecutor(max_workers=min(max_workers, len(missing_teams))) as executor:
            future_to_team = {executor.submit(self.get_team_lines, t, force=force): t for t in missing_teams}
            for fut in as_completed(future_to_team):
                t = future_to_team[fut]
                try:
                    res = fut.result()
                    if res:
                        result[t] = res
                except Exception:
                    pass

        return result


TEAM_TO_SLUG = {
    "ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres", "CGY": "calgary-flames",
    "CAR": "carolina-hurricanes", "CHI": "chicago-blackhawks", "COL": "colorado-avalanche", "CBJ": "columbus-blue-jackets",
    "DAL": "dallas-stars", "DET": "detroit-red-wings", "EDM": "edmonton-oilers", "FLA": "florida-panthers",
    "LAK": "los-angeles-kings", "MIN": "minnesota-wild", "MTL": "montreal-canadiens", "NSH": "nashville-predators",
    "NJD": "new-jersey-devils", "NYI": "new-york-islanders", "NYR": "new-york-rangers", "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers", "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks", "SEA": "seattle-kraken",
    "STL": "st-louis-blues", "TBL": "tampa-bay-lightning", "TOR": "toronto-maple-leafs", "UTA": "utah-hockey-club",
    "VAN": "vancouver-canucks", "VGK": "vegas-golden-knights", "WSH": "washington-capitals", "WPG": "winnipeg-jets"
}


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
