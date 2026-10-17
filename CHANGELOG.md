# Changelog

All notable changes to the RRS Engine project are documented in this file.

## [2.1.0] - 2026-10-17

### Added
- **NHL Timezone Support**: Added `nhl_now()`, `nhl_today()`, and `current_season_start_year()` in `nhl_api.py` using `zoneinfo.ZoneInfo("America/New_York")` with UTC-5 fallback, ensuring slate dates are accurate regardless of user local timezone.
- **Exact Dynamic Programming Optimizer**: Replaced combinatorial $C(25, 5)$ search with an exact $O(5N)$ dynamic programming solver in `solve_optimal_lineup`, maximizing EV under the rearrangement inequality in sub-millisecond execution time.
- **Gamecenter Boxscore Settlement**: Added `get_boxscore_skaters` in `nhl_api.py` fetching official NHL boxscore statistics (goals, assists, SOG, blocked shots, TOI) from `/v1/gamecenter/{id}/boxscore` when gameState is `FINAL` or `OFF`.
- **DNP Handling & Repeat Query Caching**: In `settle_past_lineups`, added automated detection for players who Did Not Play (slate $\ge 3$ days old or player has completed subsequent games), assigning 0.0 RRS without hanging as pending. Added `logs_cache` to avoid redundant API queries.
- **Roster Filtering & Duplicate Prevention**: In `get_all_club_stats`, added parallel fetching of current rosters (`/v1/roster/{team}/current`) to filter out traded players who have left their club, along with global `playerId` deduplication.
- **Enhanced Booster Resolution**:
  - Added `name_key` supporting NFKD diacritic removal and lowercasing.
  - Implemented `resolve_booster` with dedicated duplicate and non-duplicate priority chains (position -> group -> center -> bare name).
  - Added percent booster parsing (`parse_booster_val`) ensuring `%` values are properly divided by 100.
  - Added warning for unused boosters ($> 0.0$) to detect user typos in `boosters.txt`.
- **ESPN Team Mapping for Injuries**: Added `ESPN_TEAM_MAP` (`NJ` $\to$ `NJD`, `TB` $\to$ `TBL`, `LA` $\to$ `LAK`, `SJ` $\to$ `SJS`, `UTAH` $\to$ `UTA`) and injury disambiguation for players with identical names.
- **Offline Unit Test Suite**: Added 25 unit tests under `tests/test_engine.py` covering timezones, boosters, optimizer, database transactions, boxscores, and goalie factors.

### Changed
- **Goalie Factor Calibration**: In `PerformancePredictor.predict`, removed duplicate application of `goalie_factor` to `opp_def` input vector; applied `goalie_factor` once post-prediction to goals and assists (leaving shots unscaled).
- **Daily Faceoff Verification**: Daily Faceoff goalies are now validated against the slate's active matchups to prevent using stale previous-day listings.
- **Database Connection Management**: Converted `Database._get_connection` into a `@contextmanager` with 30s timeout, automatic transaction commit on success, rollback on error, and guaranteed closure.
- **History Atomic Re-writes**: Updated `save_lineup_history` to delete prior records for the slate date and mode in one transaction before inserting updated lineups with `settled = 0`.
- **CLI Validation**: Added ISO date validation (`parse_iso_date_arg`) for `--date` argument.
- **Codebase Deduplication**: Unified player collection for slate generation and `boosters.txt` generation into `collect_slate_skaters`.
- **Documentation**: Stripped UTF-8 BOM from `README.md` and added comprehensive command documentation.
