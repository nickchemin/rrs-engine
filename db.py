import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from paths import database_path

DB_PATH = str(database_path())


def season_from_date(game_date: str) -> str:
    try:
        parts = game_date.split("-")
        year = int(parts[0])
        month = int(parts[1])
        if month >= 8:
            return f"{year}{year+1}"
        else:
            return f"{year-1}{year}"
    except Exception:
        return "20242025"


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class TeamLines(dict):
    """
    Dict mapping player line records. Supports keying by (player_name, pos_group)
    or player_name fallback, while maintaining exact player count in len(self).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta: Dict[str, Any] = {}

    def get(self, key, default=None):
        if super().__contains__(key):
            return super().__getitem__(key)
        if key == "_meta":
            return getattr(self, "_meta", default)
        if isinstance(key, str):
            for k, v in self.items():
                if isinstance(k, tuple) and k[0] == key:
                    return v
        elif isinstance(key, tuple):
            p_name, pos_group = key
            for k, v in self.items():
                if isinstance(k, str) and k == p_name:
                    return v
        return default

    def __getitem__(self, key):
        val = self.get(key)
        if val is not None:
            return val
        raise KeyError(key)

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if key == "_meta":
            return bool(getattr(self, "_meta", None))
        if isinstance(key, str):
            for k in self.keys():
                if isinstance(k, tuple) and k[0] == key:
                    return True
        elif isinstance(key, tuple):
            p_name, pos_group = key
            for k in self.keys():
                if isinstance(k, str) and k == p_name:
                    return True
        return False


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    team TEXT,
                    position TEXT,
                    updated_at REAL
                );

                CREATE TABLE IF NOT EXISTS game_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    game_id INTEGER,
                    game_date TEXT NOT NULL,
                    season TEXT NOT NULL,
                    is_home INTEGER NOT NULL,
                    opponent TEXT NOT NULL,
                    goals REAL NOT NULL,
                    assists REAL NOT NULL,
                    shots REAL NOT NULL,
                    toi TEXT NOT NULL,
                    toi_minutes REAL NOT NULL,
                    is_b2b INTEGER DEFAULT 0,
                    rest_days REAL DEFAULT 2.0,
                    opp_is_b2b INTEGER DEFAULT 0,
                    UNIQUE(player_id, game_id)
                );

                CREATE INDEX IF NOT EXISTS idx_logs_player_date ON game_logs(player_id, game_date);

                CREATE TABLE IF NOT EXISTS standings (
                    team TEXT PRIMARY KEY,
                    ga_per_game REAL NOT NULL,
                    defense_factor REAL NOT NULL,
                    pk_pct REAL NOT NULL DEFAULT 0.80,
                    times_shorthanded_per_game REAL NOT NULL DEFAULT 3.0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS standings_history (
                    team TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    ga_per_game REAL NOT NULL,
                    defense_factor REAL NOT NULL,
                    pk_pct REAL NOT NULL,
                    times_shorthanded_per_game REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (team, snapshot_date)
                );

                CREATE INDEX IF NOT EXISTS idx_standings_history_date
                    ON standings_history(snapshot_date);

                CREATE TABLE IF NOT EXISTS schedule (
                    game_id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    start_time TEXT,
                    state TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_schedule_date ON schedule(date);

                CREATE TABLE IF NOT EXISTS boosters (
                    player_name TEXT PRIMARY KEY,
                    booster REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_params (
                    name TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS injuries (
                    player_name TEXT NOT NULL,
                    team TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (player_name, team)
                );

                CREATE TABLE IF NOT EXISTS goalies (
                    team TEXT NOT NULL,
                    game_date TEXT NOT NULL DEFAULT '',
                    goalie_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    save_pct REAL NOT NULL,
                    gaa REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (team, game_date)
                );

                CREATE TABLE IF NOT EXISTS team_lines (
                    team TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    pos_group TEXT NOT NULL DEFAULT 'F',
                    ev_line TEXT,
                    pp_unit TEXT,
                    pk_unit TEXT,
                    position TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (team, player_name, pos_group)
                );

                CREATE INDEX IF NOT EXISTS idx_lines_team ON team_lines(team);

                CREATE TABLE IF NOT EXISTS lineup_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_date TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    slot INTEGER NOT NULL,
                    slot_boost REAL NOT NULL,
                    player_id INTEGER,
                    player_name TEXT NOT NULL,
                    team TEXT NOT NULL,
                    opponent TEXT NOT NULL,
                    ev_line TEXT,
                    pp_unit TEXT,
                    booster REAL NOT NULL,
                    total_boost REAL NOT NULL,
                    proj_xrrs REAL NOT NULL,
                    proj_rrs REAL NOT NULL,
                    actual_goals REAL,
                    actual_assists REAL,
                    actual_shots REAL,
                    actual_blocks REAL,
                    actual_toi TEXT,
                    actual_xrrs REAL,
                    actual_rrs REAL,
                    app_rrs REAL,
                    app_payout REAL,
                    settled INTEGER DEFAULT 0,
                    calibrated INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    settled_at REAL,
                    UNIQUE(game_date, mode, slot)
                );

                CREATE INDEX IF NOT EXISTS idx_history_date ON lineup_history(game_date);
            """)

            # Schema migration for existing lineup_history
            cur.execute("PRAGMA table_info(lineup_history)")
            existing_cols = [row["name"] for row in cur.fetchall()]
            if existing_cols:
                if "proj_base_pts" in existing_cols and "proj_xrrs" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history RENAME COLUMN proj_base_pts TO proj_xrrs")
                elif "proj_xfps" in existing_cols and "proj_xrrs" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history RENAME COLUMN proj_xfps TO proj_xrrs")

                if "actual_base_pts" in existing_cols and "actual_xrrs" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history RENAME COLUMN actual_base_pts TO actual_xrrs")
                elif "actual_fps" in existing_cols and "actual_xrrs" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history RENAME COLUMN actual_fps TO actual_xrrs")

                if "proj_score" in existing_cols and "proj_rrs" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history RENAME COLUMN proj_score TO proj_rrs")
                if "actual_score" in existing_cols and "actual_rrs" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history RENAME COLUMN actual_score TO actual_rrs")

                if "proj_rax" in existing_cols:
                    cur.execute("ALTER TABLE lineup_history DROP COLUMN proj_rax")
                if "actual_rax" in existing_cols:
                    cur.execute("ALTER TABLE lineup_history DROP COLUMN actual_rax")

                if "calibrated" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history ADD COLUMN calibrated INTEGER DEFAULT 0")
                    cur.execute("""
                        UPDATE lineup_history
                        SET proj_xrrs = ROUND(proj_xrrs / 1.2, 2),
                            proj_rrs = ROUND(proj_rrs / 1.2, 2),
                            calibrated = 1
                    """)

                if "ev_line" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history ADD COLUMN ev_line TEXT")
                if "pp_unit" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history ADD COLUMN pp_unit TEXT")
                if "actual_blocks" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history ADD COLUMN actual_blocks REAL")
                if "app_rrs" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history ADD COLUMN app_rrs REAL")
                if "app_payout" not in existing_cols:
                    cur.execute("ALTER TABLE lineup_history ADD COLUMN app_payout REAL")

            # Schema migration for injuries composite primary key
            cur.execute("PRAGMA table_info(injuries)")
            inj_info = cur.fetchall()
            if inj_info:
                pk_cols = [c["name"] for c in inj_info if c["pk"] > 0]
                if len(pk_cols) == 1 and pk_cols[0] == "player_name":
                    cur.execute("DROP TABLE injuries")
                    cur.execute("""
                        CREATE TABLE injuries (
                            player_name TEXT NOT NULL,
                            team TEXT NOT NULL,
                            status TEXT NOT NULL,
                            detail TEXT,
                            updated_at REAL NOT NULL,
                            PRIMARY KEY (player_name, team)
                        )
                    """)

            # Schema migration for goalies composite primary key (team, game_date)
            cur.execute("PRAGMA table_info(goalies)")
            g_info = cur.fetchall()
            if g_info:
                pk_cols = [c["name"] for c in g_info if c["pk"] > 0]
                if pk_cols != ["team", "game_date"]:
                    col_names = [c["name"] for c in g_info]
                    date_expr = "COALESCE(game_date, '')" if "game_date" in col_names else "''"
                    cur.execute("""
                        CREATE TABLE goalies_new (
                            team TEXT NOT NULL,
                            game_date TEXT NOT NULL DEFAULT '',
                            goalie_name TEXT NOT NULL,
                            status TEXT NOT NULL,
                            save_pct REAL NOT NULL,
                            gaa REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            PRIMARY KEY (team, game_date)
                        )
                    """)
                    cur.execute(f"""
                        INSERT OR IGNORE INTO goalies_new (team, game_date, goalie_name, status, save_pct, gaa, updated_at)
                        SELECT team, {date_expr}, goalie_name, status, save_pct, gaa, updated_at FROM goalies
                    """)
                    cur.execute("DROP TABLE goalies")
                    cur.execute("ALTER TABLE goalies_new RENAME TO goalies")

            # Schema migration for team_lines (team, player_name, pos_group)
            cur.execute("PRAGMA table_info(team_lines)")
            tl_info = cur.fetchall()
            if tl_info:
                col_names = [c["name"] for c in tl_info]
                if "pos_group" not in col_names:
                    cur.execute("DROP TABLE team_lines")
                    cur.execute("""
                        CREATE TABLE team_lines (
                            team TEXT NOT NULL,
                            player_name TEXT NOT NULL,
                            pos_group TEXT NOT NULL DEFAULT 'F',
                            ev_line TEXT,
                            pp_unit TEXT,
                            pk_unit TEXT,
                            position TEXT,
                            updated_at REAL NOT NULL,
                            PRIMARY KEY (team, player_name, pos_group)
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_lines_team ON team_lines(team)")

            # Schema migration for game_logs context columns
            cur.execute("PRAGMA table_info(game_logs)")
            gl_cols = [row["name"] for row in cur.fetchall()]
            if gl_cols:
                if "is_b2b" not in gl_cols:
                    cur.execute("ALTER TABLE game_logs ADD COLUMN is_b2b INTEGER DEFAULT 0")
                if "rest_days" not in gl_cols:
                    cur.execute("ALTER TABLE game_logs ADD COLUMN rest_days REAL DEFAULT 2.0")
                if "opp_is_b2b" not in gl_cols:
                    cur.execute("ALTER TABLE game_logs ADD COLUMN opp_is_b2b INTEGER DEFAULT 0")

            # Schema migration for standings special teams columns
            cur.execute("PRAGMA table_info(standings)")
            s_cols = [row["name"] for row in cur.fetchall()]
            if s_cols:
                if "pk_pct" not in s_cols:
                    cur.execute("ALTER TABLE standings ADD COLUMN pk_pct REAL DEFAULT 0.80")
                if "times_shorthanded_per_game" not in s_cols:
                    cur.execute("ALTER TABLE standings ADD COLUMN times_shorthanded_per_game REAL DEFAULT 3.0")

            conn.commit()

    # --- Standings ---
    def get_standings(self, max_age_hours: float = 12.0) -> Optional[Dict[str, Dict[str, float]]]:
        min_updated = time.time() - (max_age_hours * 3600)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT team, ga_per_game, defense_factor,
                       COALESCE(pk_pct, 0.80) as pk_pct,
                       COALESCE(times_shorthanded_per_game, 3.0) as times_shorthanded_per_game,
                       updated_at
                FROM standings WHERE updated_at > ?
            """, (min_updated,))
            rows = cur.fetchall()
            if not rows:
                return None
            return {
                r["team"]: {
                    "ga_per_game": float(r["ga_per_game"]),
                    "defense_factor": float(r["defense_factor"]),
                    "pk_pct": float(r["pk_pct"]),
                    "times_shorthanded_per_game": float(r["times_shorthanded_per_game"])
                } for r in rows
            }

    def save_standings(self, metrics: Dict[str, Dict[str, float]], snapshot_date: Optional[str] = None):
        now = time.time()
        snapshot = snapshot_date or datetime.now().strftime("%Y-%m-%d")
        with self._get_connection() as conn:
            cur = conn.cursor()
            for team, data in metrics.items():
                ga_per_game = _as_float(data.get("ga_per_game"), 3.0)
                defense_factor = _as_float(data.get("defense_factor"), 1.0)
                pk_pct = _as_float(data.get("pk_pct"), 0.80)
                times_shorthanded = _as_float(data.get("times_shorthanded_per_game"), 3.0)

                cur.execute("""
                    INSERT INTO standings (team, ga_per_game, defense_factor, pk_pct, times_shorthanded_per_game, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(team) DO UPDATE SET
                        ga_per_game=excluded.ga_per_game,
                        defense_factor=excluded.defense_factor,
                        pk_pct=excluded.pk_pct,
                        times_shorthanded_per_game=excluded.times_shorthanded_per_game,
                        updated_at=excluded.updated_at
                """, (team, ga_per_game, defense_factor, pk_pct, times_shorthanded, now))

                cur.execute("""
                    INSERT INTO standings_history (
                        team, snapshot_date, ga_per_game, defense_factor, pk_pct,
                        times_shorthanded_per_game, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(team, snapshot_date) DO UPDATE SET
                        ga_per_game=excluded.ga_per_game,
                        defense_factor=excluded.defense_factor,
                        pk_pct=excluded.pk_pct,
                        times_shorthanded_per_game=excluded.times_shorthanded_per_game,
                        updated_at=excluded.updated_at
                """, (team, snapshot, ga_per_game, defense_factor, pk_pct, times_shorthanded, now))
            conn.commit()

    def get_standings_as_of(self, game_date: str) -> Dict[str, Dict[str, float]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT h.team, h.ga_per_game, h.defense_factor, h.pk_pct,
                       h.times_shorthanded_per_game
                FROM standings_history h
                JOIN (
                    SELECT team, MAX(snapshot_date) AS snapshot_date
                    FROM standings_history
                    WHERE snapshot_date <= ?
                    GROUP BY team
                ) latest
                  ON latest.team = h.team AND latest.snapshot_date = h.snapshot_date
            """, (game_date,))
            rows = cur.fetchall()
        return {
            r["team"]: {
                "ga_per_game": float(r["ga_per_game"]),
                "defense_factor": float(r["defense_factor"]),
                "pk_pct": float(r["pk_pct"]),
                "times_shorthanded_per_game": float(r["times_shorthanded_per_game"]),
            }
            for r in rows
        }

    # --- Schedule ---
    def get_schedule(self, max_age_hours: float = 3.0) -> List[Dict[str, Any]]:
        min_updated = time.time() - (max_age_hours * 3600)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT game_id, date, home_team, away_team, start_time, state, updated_at FROM schedule WHERE updated_at > ? ORDER BY date, start_time", (min_updated,))
            rows = cur.fetchall()
            if not rows:
                return []
            dates = {}
            for r in rows:
                d = r["date"]
                if d not in dates:
                    dates[d] = []
                dates[d].append({
                    "gameId": r["game_id"],
                    "date": r["date"],
                    "homeTeam": r["home_team"],
                    "awayTeam": r["away_team"],
                    "startTimeUTC": r["start_time"],
                    "gameState": r["state"]
                })
            return [{"date": d, "games": g} for d, g in dates.items()]

    def save_schedule(self, schedule_days: List[Dict[str, Any]]):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            for day in schedule_days:
                d_str = day["date"]
                for g in day.get("games", []):
                    cur.execute("""
                        INSERT INTO schedule (game_id, date, home_team, away_team, start_time, state, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(game_id) DO UPDATE SET
                            date=excluded.date,
                            home_team=excluded.home_team,
                            away_team=excluded.away_team,
                            start_time=excluded.start_time,
                            state=excluded.state,
                            updated_at=excluded.updated_at
                    """, (
                        g.get("gameId"),
                        d_str,
                        g.get("homeTeam"),
                        g.get("awayTeam"),
                        g.get("startTimeUTC"),
                        g.get("gameState"),
                        now
                    ))
            conn.commit()

    def get_schedule_context(self, team: str, opp: str, game_date: str) -> Dict[str, Any]:
        """
        Calculates schedule and fatigue context for team and opponent on game_date:
        - is_home: whether team is at home on game_date
        - is_b2b: 1 if team played on game_date - 1 day, else 0
        - rest_days: days since team's last match (capped at 5.0)
        - opp_is_b2b: 1 if opponent played on game_date - 1 day, else 0
        - opp_rest_days: days since opponent's last match (capped at 5.0)
        Queries across both schedule and game_logs so dates outside current week window
        do not fall back to defaults.
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Team schedule context across schedule UNION game_logs
            cur.execute("""
                SELECT MAX(d) FROM (
                    SELECT date AS d FROM schedule
                    WHERE (home_team = ? OR away_team = ?) AND date < ?
                    UNION
                    SELECT g.game_date AS d FROM game_logs g
                    JOIN players p ON g.player_id = p.player_id
                    WHERE p.team = ? AND g.game_date < ?
                    UNION
                    SELECT game_date AS d FROM game_logs
                    WHERE opponent = ? AND game_date < ?
                )
            """, (team, team, game_date, team, game_date, team, game_date))
            t_row = cur.fetchone()
            prev_t_date = t_row[0] if t_row and t_row[0] else None

            # Opponent schedule context across schedule UNION game_logs
            cur.execute("""
                SELECT MAX(d) FROM (
                    SELECT date AS d FROM schedule
                    WHERE (home_team = ? OR away_team = ?) AND date < ?
                    UNION
                    SELECT g.game_date AS d FROM game_logs g
                    JOIN players p ON g.player_id = p.player_id
                    WHERE p.team = ? AND g.game_date < ?
                    UNION
                    SELECT game_date AS d FROM game_logs
                    WHERE opponent = ? AND game_date < ?
                )
            """, (opp, opp, game_date, opp, game_date, opp, game_date))
            o_row = cur.fetchone()
            prev_o_date = o_row[0] if o_row and o_row[0] else None

            cur.execute("""
                SELECT home_team FROM schedule
                WHERE date = ? AND (home_team = ? OR away_team = ?)
            """, (game_date, team, team))
            h_row = cur.fetchone()
            if h_row and h_row[0]:
                is_home = (h_row[0] == team)
            else:
                cur.execute("""
                    SELECT is_home FROM game_logs g
                    JOIN players p ON g.player_id = p.player_id
                    WHERE p.team = ? AND g.game_date = ?
                    LIMIT 1
                """, (team, game_date))
                gl_h = cur.fetchone()
                is_home = bool(gl_h[0]) if gl_h is not None else True

        try:
            curr_dt = datetime.strptime(game_date, "%Y-%m-%d")
            if prev_t_date:
                t_dt = datetime.strptime(prev_t_date, "%Y-%m-%d")
                rest_days = min(5.0, max(0.0, float((curr_dt - t_dt).days)))
                is_b2b = 1 if rest_days <= 1.0 else 0
            else:
                rest_days = 2.0
                is_b2b = 0

            if prev_o_date:
                o_dt = datetime.strptime(prev_o_date, "%Y-%m-%d")
                opp_rest_days = min(5.0, max(0.0, float((curr_dt - o_dt).days)))
                opp_is_b2b = 1 if opp_rest_days <= 1.0 else 0
            else:
                opp_rest_days = 2.0
                opp_is_b2b = 0
        except Exception:
            rest_days = 2.0
            is_b2b = 0
            opp_rest_days = 2.0
            opp_is_b2b = 0

        return {
            "is_home": is_home,
            "is_b2b": is_b2b,
            "rest_days": rest_days,
            "opp_is_b2b": opp_is_b2b,
            "opp_rest_days": opp_rest_days
        }

    # --- Players & Logs ---
    def get_player(self, player_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT player_id, name, team, position, updated_at FROM players WHERE player_id = ?", (player_id,))
            r = cur.fetchone()
            if r:
                return dict(r)
        return None

    def touch_player_updated_at(self, player_id: int):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE players SET updated_at = ? WHERE player_id = ?", (now, player_id))
            conn.commit()

    def get_all_players(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT player_id, name, team, position FROM players ORDER BY name")
            return [dict(r) for r in cur.fetchall()]

    def save_player(self, player_id: int, name: str, team: str, position: str):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO players (player_id, name, team, position, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    name=excluded.name,
                    team=excluded.team,
                    position=excluded.position,
                    updated_at=excluded.updated_at
            """, (player_id, name, team, position, now))
            conn.commit()

    def save_players_batch(self, players: List[Dict[str, Any]]):
        if not players:
            return
        now = time.time()
        params = [
            (
                p["id"],
                p["name"],
                p.get("team", ""),
                p.get("pos", p.get("position", "F")),
                now
            )
            for p in players
        ]
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.executemany("""
                INSERT INTO players (player_id, name, team, position, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    name=excluded.name,
                    team=excluded.team,
                    position=excluded.position,
                    updated_at=excluded.updated_at
            """, params)
            conn.commit()

    def get_game_logs(self, player_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT game_id, game_date, season, is_home, opponent, goals, assists, shots, toi, toi_minutes,
                       COALESCE(is_b2b, 0) as is_b2b,
                       COALESCE(rest_days, 2.0) as rest_days,
                       COALESCE(opp_is_b2b, 0) as opp_is_b2b
                FROM game_logs
                WHERE player_id = ?
                ORDER BY game_date ASC
            """, (player_id,))
            rows = cur.fetchall()

        logs: List[Dict[str, Any]] = []
        previous_date: Optional[str] = None
        for r in rows:
            rest_days = float(r["rest_days"])
            is_b2b = int(r["is_b2b"])
            if previous_date and r["game_date"]:
                try:
                    gap = (
                        datetime.strptime(r["game_date"], "%Y-%m-%d")
                        - datetime.strptime(previous_date, "%Y-%m-%d")
                    ).days
                    rest_days = min(10.0, max(0.0, float(gap)))
                    is_b2b = 1 if rest_days <= 1.0 else 0
                except ValueError:
                    pass

            logs.append({
                "gameId": r["game_id"],
                "gameDate": r["game_date"],
                "seasonId": r["season"],
                "homeRoadFlag": "H" if r["is_home"] else "R",
                "is_home": 1 if r["is_home"] else 0,
                "opponentAbbrev": r["opponent"],
                "goals": r["goals"],
                "assists": r["assists"],
                "shots": r["shots"],
                "toi": r["toi"],
                "toi_minutes": r["toi_minutes"],
                "is_b2b": is_b2b,
                "rest_days": rest_days,
                "opp_is_b2b": int(r["opp_is_b2b"])
            })
            previous_date = r["game_date"]
        return logs

    def has_game_logs(self, player_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM game_logs WHERE player_id = ? LIMIT 1", (player_id,))
            return cur.fetchone() is not None

    def get_latest_game_date(self, player_id: int) -> Optional[str]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT MAX(game_date) FROM game_logs WHERE player_id = ?", (player_id,))
            r = cur.fetchone()
            return r[0] if r and r[0] else None

    def save_game_logs(self, player_id: int, logs: List[Dict[str, Any]]):
        if not logs:
            return

        # Sort logs chronologically to compute back-to-back and rest accurately
        valid_logs = [g for g in logs if (g.get("gameId") or g.get("id")) and g.get("gameDate")]
        valid_logs.sort(key=lambda x: str(x.get("gameDate", "")))

        # Look up most recent prior game in DB before this batch's earliest game
        prior_game_date = None
        if valid_logs and valid_logs[0].get("gameDate"):
            first_game_date = valid_logs[0]["gameDate"]
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT MAX(game_date) FROM game_logs WHERE player_id = ? AND game_date < ?",
                    (player_id, first_game_date)
                )
                r = cur.fetchone()
                if r and r[0]:
                    prior_game_date = r[0]

        rows = []
        for i, g in enumerate(valid_logs):
            game_id = g.get("gameId") or g.get("id")
            game_date = g.get("gameDate")

            toi_str = str(g.get("toi") or "00:00")
            try:
                parts = toi_str.split(":")
                toi_min = float(parts[0]) + float(parts[1]) / 60.0 if len(parts) == 2 else float(toi_str)
            except Exception:
                toi_min = 0.0

            season_id = str(g.get("seasonId") or season_from_date(game_date))

            # Fatigue & schedule calculation (across batch items and DB historical records)
            if i > 0 and valid_logs[i - 1].get("gameDate"):
                prev_date_str = valid_logs[i - 1]["gameDate"]
            elif i == 0 and prior_game_date:
                prev_date_str = prior_game_date
            else:
                prev_date_str = None

            if prev_date_str:
                try:
                    d_curr = datetime.strptime(game_date, "%Y-%m-%d")
                    d_prev = datetime.strptime(prev_date_str, "%Y-%m-%d")
                    rest = min(10.0, max(0.0, float((d_curr - d_prev).days)))
                    is_b2b = 1 if rest <= 1.0 else 0
                except Exception:
                    rest = float(g.get("rest_days", 2.0))
                    is_b2b = int(g.get("is_b2b", 0))
            else:
                rest = float(g.get("rest_days", 2.0))
                is_b2b = int(g.get("is_b2b", 0))

            opp_is_b2b = int(g.get("opp_is_b2b", 0))

            rows.append((
                player_id,
                game_id,
                game_date,
                season_id,
                1 if g.get("homeRoadFlag") == "H" or g.get("is_home") in (1, True) else 0,
                g.get("opponentAbbrev", ""),
                _as_float(g.get("goals"), 0.0),
                _as_float(g.get("assists"), 0.0),
                _as_float(g.get("shots"), 0.0),
                toi_str,
                toi_min,
                is_b2b,
                rest,
                opp_is_b2b
            ))

        if not rows:
            return

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.executemany("""
                INSERT INTO game_logs (
                    player_id, game_id, game_date, season, is_home, opponent,
                    goals, assists, shots, toi, toi_minutes,
                    is_b2b, rest_days, opp_is_b2b
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, game_id) DO UPDATE SET
                    game_date = excluded.game_date,
                    season = excluded.season,
                    is_home = excluded.is_home,
                    opponent = excluded.opponent,
                    goals = excluded.goals,
                    assists = excluded.assists,
                    shots = excluded.shots,
                    toi = excluded.toi,
                    toi_minutes = excluded.toi_minutes,
                    is_b2b = excluded.is_b2b,
                    rest_days = excluded.rest_days,
                    opp_is_b2b = excluded.opp_is_b2b
            """, rows)
            conn.commit()

    # --- Model Parameters ---
    def save_model_params(self, params: Dict[str, float]):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM model_params")
            for name, value in params.items():
                cur.execute(
                    "INSERT INTO model_params (name, value, updated_at) VALUES (?, ?, ?)",
                    (name, float(value), now),
                )
            conn.commit()

    def get_model_params(self) -> Optional[Dict[str, float]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name, value FROM model_params")
            rows = cur.fetchall()
        if not rows:
            return None
        return {row["name"]: float(row["value"]) for row in rows}

    # --- Boosters ---
    def get_all_boosters(self) -> Dict[str, float]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT player_name, booster FROM boosters")
            return {r["player_name"]: float(r["booster"]) for r in cur.fetchall()}

    def set_booster(self, player_name: str, booster: float):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO boosters (player_name, booster, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(player_name) DO UPDATE SET
                    booster=excluded.booster,
                    updated_at=excluded.updated_at
            """, (player_name, booster, now))
            conn.commit()

    def delete_boosters(self, player_names: List[str]):
        if not player_names:
            return
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.executemany("DELETE FROM boosters WHERE player_name = ?", [(n,) for n in player_names])
            conn.commit()

    # --- Injuries ---
    def get_injuries(self, max_age_hours: float = 3.0) -> Optional[Dict[Tuple[str, str], Dict[str, str]]]:
        min_updated = time.time() - (max_age_hours * 3600)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT player_name, team, status, detail, updated_at FROM injuries WHERE updated_at > ?", (min_updated,))
            rows = cur.fetchall()
            if not rows:
                return None
            return {
                (r["player_name"], r["team"] or ""): {
                    "name": r["player_name"],
                    "team": r["team"] or "",
                    "status": r["status"],
                    "detail": r["detail"] or ""
                } for r in rows
            }

    def save_injuries(self, injuries: List[Dict[str, Any]]):
        if not injuries:
            return
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM injuries")
            for item in injuries:
                cur.execute("""
                    INSERT OR REPLACE INTO injuries (player_name, team, status, detail, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    item.get("name", ""),
                    item.get("team", ""),
                    item.get("status", "Out"),
                    item.get("detail", ""),
                    now
                ))

    def get_player_injury(self, player_name: str, team: str = "") -> Optional[Dict[str, str]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            if team:
                cur.execute("SELECT team, status, detail FROM injuries WHERE player_name = ? AND team = ?", (player_name, team))
            else:
                cur.execute("SELECT team, status, detail FROM injuries WHERE player_name = ?", (player_name,))
            r = cur.fetchone()
            if r:
                return {
                    "team": r["team"] or "",
                    "status": r["status"],
                    "detail": r["detail"] or ""
                }
            return None

    # --- Goalies ---
    def get_starting_goalies(self, game_date: str = "", max_age_hours: float = 4.0) -> Optional[Dict[str, Dict[str, Any]]]:
        min_updated = time.time() - (max_age_hours * 3600)
        with self._get_connection() as conn:
            cur = conn.cursor()
            if game_date:
                cur.execute("""
                    SELECT team, game_date, goalie_name, status, save_pct, gaa, updated_at
                    FROM goalies
                    WHERE updated_at > ? AND game_date = ?
                """, (min_updated, game_date))
            else:
                cur.execute("""
                    SELECT team, game_date, goalie_name, status, save_pct, gaa, updated_at
                    FROM goalies
                    WHERE updated_at > ?
                """, (min_updated,))
            rows = cur.fetchall()
            if not rows:
                return None
            return {
                r["team"]: {
                    "goalie_name": r["goalie_name"],
                    "status": r["status"],
                    "save_pct": float(r["save_pct"]),
                    "gaa": float(r["gaa"]),
                    "game_date": r["game_date"] or ""
                } for r in rows
            }

    def save_starting_goalies(self, goalies: Dict[str, Dict[str, Any]], game_date: str = ""):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            for team, data in goalies.items():
                g_date = data.get("game_date") or game_date
                cur.execute("""
                    INSERT INTO goalies (team, game_date, goalie_name, status, save_pct, gaa, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(team, game_date) DO UPDATE SET
                        goalie_name=excluded.goalie_name,
                        status=excluded.status,
                        save_pct=excluded.save_pct,
                        gaa=excluded.gaa,
                        updated_at=excluded.updated_at
                """, (
                    team,
                    g_date,
                    data.get("goalie_name", ""),
                    data.get("status", "Projected"),
                    _as_float(data.get("save_pct"), 0.902),
                    _as_float(data.get("gaa"), 3.00),
                    now
                ))

    # --- Team Lines & Power Play Units ---
    def get_team_lines(self, team: str, max_age_hours: float = 12.0) -> Optional[TeamLines]:
        min_updated = time.time() - (max_age_hours * 3600)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT player_name, pos_group, ev_line, pp_unit, pk_unit, position, updated_at
                FROM team_lines
                WHERE team = ? AND updated_at > ?
            """, (team, min_updated))
            rows = cur.fetchall()
            if not rows:
                return None
            res = TeamLines()
            for r in rows:
                data = {
                    "player_name": r["player_name"],
                    "ev_line": r["ev_line"],
                    "pp_unit": r["pp_unit"],
                    "pk_unit": r["pk_unit"],
                    "position": r["position"],
                    "pos_group": r["pos_group"]
                }
                res[(r["player_name"], r["pos_group"])] = data
            has_pp = any(v.get("pp_unit") for k, v in res.items() if isinstance(v, dict))
            res._meta = {"has_pp": has_pp}
            return res

    def save_team_lines(self, team: str, lines_data: Dict[Any, Dict[str, Any]]):
        if not lines_data:
            return
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Clean stale records for this team before inserting fresh lines
            cur.execute("DELETE FROM team_lines WHERE team = ?", (team,))
            for k, data in lines_data.items():
                if k == "_meta" or not isinstance(data, dict):
                    continue
                if isinstance(k, tuple):
                    p_name, pos_group = k
                else:
                    p_name = k
                    pos_group = data.get("pos_group", "D" if str(data.get("position", "")).startswith("D") else "F")
                cur.execute("""
                    INSERT INTO team_lines (team, player_name, pos_group, ev_line, pp_unit, pk_unit, position, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(team, player_name, pos_group) DO UPDATE SET
                        ev_line=excluded.ev_line,
                        pp_unit=excluded.pp_unit,
                        pk_unit=excluded.pk_unit,
                        position=excluded.position,
                        updated_at=excluded.updated_at
                """, (
                    team,
                    p_name,
                    pos_group,
                    data.get("ev_line"),
                    data.get("pp_unit"),
                    data.get("pk_unit"),
                    data.get("position"),
                    now
                ))
            conn.commit()

    def get_all_lines_for_teams(self, teams: List[str], max_age_hours: float = 12.0) -> Dict[str, TeamLines]:
        if not teams:
            return {}
        min_updated = time.time() - (max_age_hours * 3600)
        placeholders = ",".join(["?"] * len(teams))
        result: Dict[str, TeamLines] = {}
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT team, player_name, pos_group, ev_line, pp_unit, pk_unit, position, updated_at
                FROM team_lines
                WHERE team IN ({placeholders}) AND updated_at > ?
            """, (*teams, min_updated))
            rows = cur.fetchall()
            for r in rows:
                t = r["team"]
                data = {
                    "player_name": r["player_name"],
                    "ev_line": r["ev_line"],
                    "pp_unit": r["pp_unit"],
                    "pk_unit": r["pk_unit"],
                    "position": r["position"],
                    "pos_group": r["pos_group"]
                }
                if t not in result:
                    result[t] = TeamLines()
                result[t][(r["player_name"], r["pos_group"])] = data
        for t, tl in result.items():
            has_pp = any(v.get("pp_unit") for k, v in tl.items() if isinstance(v, dict))
            tl._meta = {"has_pp": has_pp}
        return result

    # --- Lineup History & Tracking ---
    def save_lineup_history(self, game_date: str, mode: str, lineup: List[Dict[str, Any]]):
        if not lineup:
            return
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM lineup_history WHERE game_date = ? AND mode = ? AND settled = 0 AND slot > ?",
                (game_date, mode, len(lineup)),
            )
            for item in lineup:
                cur.execute("""
                    INSERT INTO lineup_history (
                        game_date, mode, slot, slot_boost, player_id, player_name,
                        team, opponent, ev_line, pp_unit, booster, total_boost, proj_xrrs, proj_rrs,
                        calibrated, settled, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?)
                    ON CONFLICT(game_date, mode, slot) DO UPDATE SET
                        slot_boost = excluded.slot_boost,
                        player_id = excluded.player_id,
                        player_name = excluded.player_name,
                        team = excluded.team,
                        opponent = excluded.opponent,
                        ev_line = excluded.ev_line,
                        pp_unit = excluded.pp_unit,
                        booster = excluded.booster,
                        total_boost = excluded.total_boost,
                        proj_xrrs = excluded.proj_xrrs,
                        proj_rrs = excluded.proj_rrs
                """, (
                    game_date,
                    mode,
                    item["slot"],
                    item["slot_boost"],
                    item.get("id"),
                    item["name"],
                    item["team"],
                    item.get("opp", ""),
                    item.get("ev_line"),
                    item.get("pp_unit"),
                    item.get("booster", 0.0),
                    item.get("total_boost", item["slot_boost"]),
                    item.get("xRRS", item.get("base_pts", 0.0)),
                    item.get("score", 0.0),
                    now
                ))
            conn.commit()

    def get_unsettled_history(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, game_date, mode, slot, player_id, player_name, team, total_boost, proj_rrs
                FROM lineup_history
                WHERE settled = 0
            """)
            return [dict(r) for r in cur.fetchall()]

    def settle_history_player(
        self,
        entry_id: int,
        goals: float,
        assists: float,
        shots: float,
        toi: str,
        actual_xrrs: float,
        actual_rrs: float,
        blocks: float = 0.0,
        app_rrs: Optional[float] = None,
        app_payout: Optional[float] = None
    ):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE lineup_history SET
                    actual_goals = ?,
                    actual_assists = ?,
                    actual_shots = ?,
                    actual_blocks = ?,
                    actual_toi = ?,
                    actual_xrrs = ?,
                    actual_rrs = ?,
                    app_rrs = COALESCE(?, app_rrs),
                    app_payout = COALESCE(?, app_payout),
                    settled = 1,
                    settled_at = ?
                WHERE id = ?
            """, (goals, assists, shots, blocks, toi, actual_xrrs, actual_rrs, app_rrs, app_payout, now, entry_id))
            conn.commit()

    def record_app_results(self, updates: List[Dict[str, Any]]):
        """
        Updates actual Real Sports mobile app scores (app_rrs, app_payout) for lineup_history.
        Each dict in updates must contain 'id' and 'app_rrs' (and optionally 'app_payout').
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            for u in updates:
                cur.execute("""
                    UPDATE lineup_history SET
                        app_rrs = ?,
                        app_payout = COALESCE(?, app_payout)
                    WHERE id = ?
                """, (u["app_rrs"], u.get("app_payout"), u["id"]))
            conn.commit()

    def get_lineup_for_date(self, game_date: str, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            if mode:
                cur.execute("""
                    SELECT id, game_date, mode, slot, slot_boost, player_id, player_name,
                           team, opponent, ev_line, pp_unit, booster, total_boost,
                           proj_xrrs, proj_rrs, actual_goals, actual_assists, actual_shots,
                           actual_blocks, actual_toi, actual_xrrs, actual_rrs, app_rrs, app_payout,
                           settled
                    FROM lineup_history
                    WHERE game_date = ? AND mode = ?
                    ORDER BY slot ASC
                """, (game_date, mode))
            else:
                cur.execute("""
                    SELECT id, game_date, mode, slot, slot_boost, player_id, player_name,
                           team, opponent, ev_line, pp_unit, booster, total_boost,
                           proj_xrrs, proj_rrs, actual_goals, actual_assists, actual_shots,
                           actual_blocks, actual_toi, actual_xrrs, actual_rrs, app_rrs, app_payout,
                           settled
                    FROM lineup_history
                    WHERE game_date = ?
                    ORDER BY mode ASC, slot ASC
                """, (game_date,))
            return [dict(r) for r in cur.fetchall()]

    def get_history_slates(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    game_date,
                    mode,
                    COUNT(*) as players_count,
                    SUM(proj_rrs) as total_proj_rrs,
                    SUM(CASE WHEN settled = 1 THEN actual_rrs ELSE 0 END) as total_actual_rrs,
                    SUM(CASE WHEN app_payout IS NOT NULL THEN app_payout ELSE 0 END) as total_app_payout,
                    COUNT(app_rrs) as app_rrs_count,
                    COUNT(app_payout) as app_payout_count,
                    AVG(proj_rrs) as avg_proj_rrs,
                    AVG(CASE WHEN settled = 1 THEN actual_rrs ELSE NULL END) as avg_actual_rrs,
                    AVG(CASE WHEN app_payout IS NOT NULL THEN (app_payout - proj_rrs) ELSE NULL END) as avg_app_bias,
                    AVG(CASE WHEN settled = 1 THEN (actual_rrs - proj_rrs) ELSE NULL END) as avg_bias_rrs,
                    MIN(settled) as all_settled
                FROM lineup_history
                GROUP BY game_date, mode
                ORDER BY game_date DESC, mode ASC
            """)
            return [dict(r) for r in cur.fetchall()]

    def get_history_slate_details(self, game_date: str, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            if mode:
                cur.execute("""
                    SELECT * FROM lineup_history
                    WHERE game_date = ? AND mode = ?
                    ORDER BY slot ASC
                """, (game_date, mode))
            else:
                cur.execute("""
                    SELECT * FROM lineup_history
                    WHERE game_date = ?
                    ORDER BY mode ASC, slot ASC
                """, (game_date,))
            return [dict(r) for r in cur.fetchall()]

    def get_role_bias_summary(self) -> Dict[str, List[float]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT pp_unit, proj_rrs, actual_rrs FROM lineup_history WHERE settled = 1")
            pp1_diffs = []
            other_diffs = []
            for row in cur.fetchall():
                p_rrs = float(row["proj_rrs"] or 0.0)
                a_rrs = float(row["actual_rrs"] or 0.0)
                diff = a_rrs - p_rrs
                if row["pp_unit"] == "PP1":
                    pp1_diffs.append(diff)
                else:
                    other_diffs.append(diff)
            return {
                "pp1_diffs": pp1_diffs,
                "other_diffs": other_diffs
            }

    def get_settled_stats_for_calibration(self, app_scores_only: bool = False) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            query = """
                SELECT
                    id, game_date, mode, slot, slot_boost, player_id, player_name,
                    team, opponent, ev_line, pp_unit, booster, total_boost,
                    proj_xrrs, proj_rrs, actual_goals, actual_assists, actual_shots,
                    actual_blocks, actual_toi, actual_xrrs, actual_rrs,
                    app_rrs, app_payout
                FROM lineup_history
                WHERE settled = 1 AND actual_rrs IS NOT NULL
            """
            if app_scores_only:
                query += " AND app_rrs IS NOT NULL"
            query += " ORDER BY game_date ASC, slot ASC"
            cur.execute(query)
            return [dict(r) for r in cur.fetchall()]

    def get_lineup_level_validation_summary(self) -> Dict[str, Any]:
        slates = self.get_history_slates()
        settled_slates = [s for s in slates if s.get("all_settled")]
        if not settled_slates:
            return {"settled_count": 0}

        diffs = []
        actuals = []
        projs = []
        hit_count = 0

        for s in settled_slates:
            p = float(s.get("total_proj_rrs") or 0.0)
            a = float(s.get("total_actual_rrs") or 0.0)
            diff = a - p
            diffs.append(diff)
            actuals.append(a)
            projs.append(p)
            if a >= p:
                hit_count += 1

        n_slates = len(settled_slates)
        mae = sum(abs(d) for d in diffs) / n_slates
        rmse = (sum(d ** 2 for d in diffs) / n_slates) ** 0.5
        bias = sum(diffs) / n_slates
        hit_rate = hit_count / n_slates

        stats = self.get_settled_stats_for_calibration()
        slot_errors: Dict[int, List[float]] = {}
        booster_errors: List[float] = []
        no_booster_errors: List[float] = []

        for item in stats:
            p_rrs = float(item.get("proj_rrs") or 0.0)
            a_rrs = float(item.get("actual_rrs") or 0.0)
            err = abs(a_rrs - p_rrs)
            slot = int(item.get("slot", 1))
            slot_errors.setdefault(slot, []).append(err)
            b = float(item.get("booster") or 0.0)
            if b > 0.0:
                booster_errors.append(err)
            else:
                no_booster_errors.append(err)

        return {
            "settled_count": n_slates,
            "total_players": len(stats),
            "lineup_mae": round(mae, 2),
            "lineup_rmse": round(rmse, 2),
            "lineup_bias": round(bias, 2),
            "hit_rate": round(hit_rate, 4),
            "slot_avg_errors": {s: round(sum(errs) / len(errs), 2) for s, errs in sorted(slot_errors.items()) if errs},
            "booster_avg_error": round(sum(booster_errors) / len(booster_errors), 2) if booster_errors else None,
            "no_booster_avg_error": round(sum(no_booster_errors) / len(no_booster_errors), 2) if no_booster_errors else None,
        }
