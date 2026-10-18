# Changelog

All notable changes to the RRS Engine project are documented in this file.

## [2.2.1] - 2026-10-18

### Added
- **Bayesian Shrinkage for Low-GP Projections**: Implemented Bayesian regression shrinkage in `PerformancePredictor.predict` and baseline slate collection for skaters with $gp < 10$: $(stat \cdot gp + pos\_avg \cdot 10) / (gp + 10)$. Skaters with $gp = 0$ receive positional baseline averages, eliminating inflated small-sample forecasts for call-ups and allowing debutants with boosters to be drafted.
- **Robust Settle Fallback for Missing Boxscores**: In `settle_past_lineups`, matches older than 3 days whose Gamecenter boxscore protocol remains unavailable or empty are settled using official stats from the player's game log rather than remaining pending forever.
- **Automatic Missing Season Ingestion**: In `nhl_api.get_player_game_logs`, if a player's latest cached log predates the current season (prior to `YYYY-09-01`), all intervening and missing seasons are concurrently fetched and persisted to the database.
- **Activation Hints for Commented Boosters**: `load_boosters_file` now detects non-zero booster assignments in `# [DEPTH]` and `# [DTD]` commented lines, emitting an actionable console hint reminding users to remove `#` to activate the player.
- **Cross-Version Python Typing Fix**: Explicitly imported `Tuple` in `db.py`, ensuring backwards compatibility without `NameError` on Python 3.12 and 3.13 where PEP 649 deferred annotation evaluation is unavailable.
- **59-Test Offline Test Suite**: Added `TestQAThirdReview` covering all edge cases across Blocks A and B with real function invocations (`is_slate_started`, `build_skater_projection`, `collect_slate_skaters`, `sync_boosters_to_db`).

### Changed
- **Flexible Booster Value Parsing**: `parse_booster_val` now supports European comma decimal separators (`1,5`), leading decimals (`.5`), and anchored pattern matching `^\+?(\d*\.?\d+)\s*([x%])?$`. Values $> 5.0$ without explicit unit are converted from percentages ($\div 100$) with a descriptive warning and exact decimal representation (e.g. `10.5` $\to$ `0.105`).
- **Out-Status Namesake Disambiguation**: Global duplicate detection in `collect_slate_skaters` runs across the full slate prior to injury exclusion. Players with identical names whose counterparts are Out/IR (e.g., Sebastian Aho CAR vs NYI, Elias Pettersson C vs D) retain their position tag in `save_boosters_file`, preventing accidental ambiguity warnings on reload.
- **Multi-Date Goalie Cache Partitioning**: Migrated the `goalies` table to a composite primary key `(team, game_date)` with an automatic table migration, preventing next-day syncs from overwriting current-day goalie data.
- **Booster Sync Safety Guard**: `sync_boosters_to_db` now guards against empty inputs, preventing schedule misses or empty slates from purging previously stored player boosters in SQLite.
- **Pettersson Position Tag Preservation**: `normalize_booster_name` now respects specified parentheses position tags (e.g., `Elias Pettersson (D, VAN)`) instead of defaulting to Center.
- **Precise Booster Multiplier Formatting**: `save_boosters_file` outputs numbers using `:g` (e.g., `0.25`) to prevent loss of precision from fixed decimal truncation.
- **Depth Player TOI Formatting**: Time on ice formatting now utilizes `divmod(round(avg_toi * 60), 60)`, avoiding edge-case `12:60` outputs.
- **CLI Argument Date Normalization**: `parse_iso_date_arg` and `parse_history_arg` reformat input dates to strict `YYYY-MM-DD` (`2026-10-1` $\to$ `2026-10-01`).
- **Safe Injury Upserts**: `save_injuries` uses `INSERT OR REPLACE` to avoid `IntegrityError` on duplicate records, and empties the local cache when ESPN reports zero injuries.
- **Past-Date Backtest Leakage Prevention**: In `build_skater_projection`, rookie projections on historical dates without past game logs return `(0.0, 0.0)` rather than leaking full-season club aggregates.
- **Banner and CLI Versioning**: Updated CLI banner and argparse description to `v2.2`.

## [2.2.0] - 2026-10-18

### Added
- **Time-Based Slate Started Check**: Slate is marked started if earliest `startTimeUTC <= datetime.now(timezone.utc)`, preventing active lineup overwrites caused by schedule caching.
- **Deep Historical Settlement**: Merged fresh `/now` logs with `db.get_game_logs(p_id)` in `settle_past_lineups`, settling past season and playoff games with actual boxscores rather than DNP.
- **Resilient Boxscore Protocol Fetching**: When boxscore request returns empty `{}` for an existing `gameId`, the settlement is deferred to subsequent runs rather than finalized with default blocks.
- **Rookie & Zero-History Fallback**: Skaters without prior game logs or empty `train_logs` fall back to baseline `xRRS` and are retained as draft candidates.
- **Same-Position Namesake Disambiguation**: Added `Name (POS, TEAM)` formatting and resolution in `normalize_booster_name`, `resolve_booster`, and `save_boosters_file`.
- **Depth Skater Discovery**: Skaters with `avg_toi < 14.5` without a custom booster are appended to team sections in `boosters.txt` as commented `# [DEPTH] Player Name = 0.0  # TEAM POS TOI`.
- **45-Test Offline Unit Suite**: Expanded offline unit test suite to 45 tests across `TestTimezoneAndDates`, `TestBoosterManagement`, `TestGoaliesAndInjuries`, `TestLineupOptimizer`, `TestDatabaseAndHistory`, and `TestNHLClient` with temporary database isolation.

### Changed
- **Leak-Free ML Projections**: `predictor.predict` now receives `game_logs=train_logs` strictly prior to the slate date, preventing data leakage during retrospective backtests.
- **Zero-GP Standings Safety**: Teams with 0 games played receive `defense_factor = 1.0`, `ga_per_game = 3.0`, and are excluded from league-wide average GA calculations.
- **Ambiguous Booster Detection**: Duplicate names without position specification do not fall back to bare names; an explicit warning is output in console.
- **Composite Key Injury Storage**: ESPN injuries are stored under `(player_name, team)` composite primary key, with atomic table replacement upon update to clear recovered players.
- **Booster Value Parsing**: `parse_booster_val` uses regex `^[+]?(\d+(?:\.\d+)?)\s*([x%])?`, divides `%` by 100, warns on multipliers $> 5.0$, and reports line numbers on unparsable values.
- **Pre-Season Filtering & Cache Invalidation**: `gameTypeId == 1` pre-season logs are rejected at top-level; stale logs older than `f"{start_year - 1}-09-01"` trigger full season refresh.
- **Daily Goalie Cache Partitioning**: `goalies` table migrated to include `game_date TEXT`, isolating goalie predictions per slate date.
- **Peripheral Weight Alignment**: Synchronized `calculate_xrrs_projection` peripheral weights to `xsog * 0.25 + xblk * 0.50` matching actual `calculate_rrs_from_stats`.
- **Stack Synergy Formula & Defensemen Pairing**: Calibrated synergy bonus to `base_val * (1.0 + 0.08 + 0.12 * (g_factor - 1.0))`; teams with 2 defensemen as top options pair the top defenseman with the first available forward.
- **Boosters Single Source of Truth**: When `boosters.txt` exists, it acts as the sole source of truth; `sync_boosters_to_db` deletes database keys absent from the file.
- **CLI Argument Architecture**: Placed `--sync` and `--history` in a mutually exclusive group, adding format validation for `--history` (`all` or `YYYY-MM-DD`).
- **Cleaned Imports**: Removed unused `nhl_now` and `parse_toi` from `main.py`, and cleaned unused imports in tests.

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
