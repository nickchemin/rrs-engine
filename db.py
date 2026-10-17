import sqlite3
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db")


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


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
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
                    UNIQUE(player_id, game_id)
                );

                CREATE INDEX IF NOT EXISTS idx_logs_player_date ON game_logs(player_id, game_date);

                CREATE TABLE IF NOT EXISTS standings (
                    team TEXT PRIMARY KEY,
                    ga_per_game REAL NOT NULL,
                    defense_factor REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

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

                CREATE TABLE IF NOT EXISTS injuries (
                    player_name TEXT PRIMARY KEY,
                    team TEXT,
                    status TEXT NOT NULL,
                    detail TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS goalies (
                    team TEXT PRIMARY KEY,
                    goalie_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    save_pct REAL NOT NULL,
                    gaa REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

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
                    booster REAL NOT NULL,
                    total_boost REAL NOT NULL,
                    proj_xrrs REAL NOT NULL,
                    proj_rrs REAL NOT NULL,
                    actual_goals REAL,
                    actual_assists REAL,
                    actual_shots REAL,
                    actual_toi TEXT,
                    actual_xrrs REAL,
                    actual_rrs REAL,
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

            conn.commit()

    # --- Standings ---
    def get_standings(self, max_age_hours: float = 12.0) -> Optional[Dict[str, Dict[str, float]]]:
        min_updated = time.time() - (max_age_hours * 3600)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT team, ga_per_game, defense_factor, updated_at FROM standings WHERE updated_at > ?", (min_updated,))
            rows = cur.fetchall()
            if not rows:
                return None
            return {
                r["team"]: {
                    "ga_per_game": float(r["ga_per_game"]),
                    "defense_factor": float(r["defense_factor"])
                } for r in rows
            }

    def save_standings(self, metrics: Dict[str, Dict[str, float]]):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            for team, data in metrics.items():
                cur.execute("""
                    INSERT INTO standings (team, ga_per_game, defense_factor, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(team) DO UPDATE SET
                        ga_per_game=excluded.ga_per_game,
                        defense_factor=excluded.defense_factor,
                        updated_at=excluded.updated_at
                """, (team, data["ga_per_game"], data["defense_factor"], now))
            conn.commit()

    # --- Schedule ---
    def get_schedule(self, max_age_hours: float = 12.0) -> List[Dict[str, Any]]:
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

    # --- Players & Logs ---
    def get_player(self, player_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT player_id, name, team, position FROM players WHERE player_id = ?", (player_id,))
            r = cur.fetchone()
            if r:
                return dict(r)
        return None

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
                SELECT game_id, game_date, season, is_home, opponent, goals, assists, shots, toi, toi_minutes
                FROM game_logs
                WHERE player_id = ?
                ORDER BY game_date ASC
            """, (player_id,))
            rows = cur.fetchall()
            return [{
                "gameId": r["game_id"],
                "gameDate": r["game_date"],
                "seasonId": r["season"],
                "homeRoadFlag": "H" if r["is_home"] else "R",
                "opponentAbbrev": r["opponent"],
                "goals": r["goals"],
                "assists": r["assists"],
                "shots": r["shots"],
                "toi": r["toi"],
                "toi_minutes": r["toi_minutes"]
            } for r in rows]

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
        rows = []
        for g in logs:
            game_id = g.get("gameId") or g.get("id")
            game_date = g.get("gameDate")
            # Skip invalid records without gameId or date (prevents NULL integrity issues)
            if not game_id or not game_date:
                continue

            toi_str = str(g.get("toi") or "00:00")
            try:
                parts = toi_str.split(":")
                toi_min = float(parts[0]) + float(parts[1]) / 60.0 if len(parts) == 2 else float(toi_str)
            except Exception:
                toi_min = 0.0

            season_id = str(g.get("seasonId") or season_from_date(game_date))

            rows.append((
                player_id,
                game_id,
                game_date,
                season_id,
                1 if g.get("homeRoadFlag") == "H" else 0,
                g.get("opponentAbbrev", ""),
                float(g.get("goals", 0)),
                float(g.get("assists", 0)),
                float(g.get("shots", 0)),
                toi_str,
                toi_min
            ))

        if not rows:
            return

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.executemany("""
                INSERT OR IGNORE INTO game_logs (
                    player_id, game_id, game_date, season, is_home, opponent, goals, assists, shots, toi, toi_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

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
    def get_injuries(self, max_age_hours: float = 4.0) -> Optional[Dict[str, Dict[str, str]]]:
        min_updated = time.time() - (max_age_hours * 3600)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT player_name, team, status, detail, updated_at FROM injuries WHERE updated_at > ?", (min_updated,))
            rows = cur.fetchall()
            if not rows:
                return None
            return {
                r["player_name"]: {
                    "team": r["team"] or "",
                    "status": r["status"],
                    "detail": r["detail"] or ""
                } for r in rows
            }

    def save_injuries(self, injuries: List[Dict[str, Any]]):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            for item in injuries:
                cur.execute("""
                    INSERT INTO injuries (player_name, team, status, detail, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(player_name) DO UPDATE SET
                        team=excluded.team,
                        status=excluded.status,
                        detail=excluded.detail,
                        updated_at=excluded.updated_at
                """, (
                    item.get("name", ""),
                    item.get("team", ""),
                    item.get("status", "Out"),
                    item.get("detail", ""),
                    now
                ))
            conn.commit()

    def get_player_injury(self, player_name: str) -> Optional[Dict[str, str]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
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
    def get_starting_goalies(self, max_age_hours: float = 4.0) -> Optional[Dict[str, Dict[str, Any]]]:
        min_updated = time.time() - (max_age_hours * 3600)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT team, goalie_name, status, save_pct, gaa, updated_at FROM goalies WHERE updated_at > ?", (min_updated,))
            rows = cur.fetchall()
            if not rows:
                return None
            return {
                r["team"]: {
                    "goalie_name": r["goalie_name"],
                    "status": r["status"],
                    "save_pct": float(r["save_pct"]),
                    "gaa": float(r["gaa"])
                } for r in rows
            }

    def save_starting_goalies(self, goalies: Dict[str, Dict[str, Any]]):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            for team, data in goalies.items():
                cur.execute("""
                    INSERT INTO goalies (team, goalie_name, status, save_pct, gaa, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(team) DO UPDATE SET
                        goalie_name=excluded.goalie_name,
                        status=excluded.status,
                        save_pct=excluded.save_pct,
                        gaa=excluded.gaa,
                        updated_at=excluded.updated_at
                """, (
                    team,
                    data.get("goalie_name", ""),
                    data.get("status", "Projected"),
                    float(data.get("save_pct", 0.902)),
                    float(data.get("gaa", 3.00)),
                    now
                ))
            conn.commit()

    # --- Lineup History & Tracking ---
    def save_lineup_history(self, game_date: str, mode: str, lineup: List[Dict[str, Any]]):
        if not lineup:
            return
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM lineup_history WHERE game_date = ? AND mode = ?", (game_date, mode))
            for item in lineup:
                cur.execute("""
                    INSERT INTO lineup_history (
                        game_date, mode, slot, slot_boost, player_id, player_name,
                        team, opponent, booster, total_boost, proj_xrrs, proj_rrs,
                        calibrated, settled, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?)
                """, (
                    game_date,
                    mode,
                    item["slot"],
                    item["slot_boost"],
                    item.get("id"),
                    item["name"],
                    item["team"],
                    item.get("opp", ""),
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
        actual_rrs: float
    ):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE lineup_history SET
                    actual_goals = ?,
                    actual_assists = ?,
                    actual_shots = ?,
                    actual_toi = ?,
                    actual_xrrs = ?,
                    actual_rrs = ?,
                    settled = 1,
                    settled_at = ?
                WHERE id = ?
            """, (goals, assists, shots, toi, actual_xrrs, actual_rrs, now, entry_id))
            conn.commit()

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
                    AVG(proj_rrs) as avg_proj_rrs,
                    AVG(CASE WHEN settled = 1 THEN actual_rrs ELSE NULL END) as avg_actual_rrs,
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
