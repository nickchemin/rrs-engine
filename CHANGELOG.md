# Changelog

All notable changes to the RRS Engine project are documented in this file.

## [2.4.0] - 2026-10-19

### Added
- **Slot-Aligned Stack Synergy Computation**: In `solve_stacked_lineup`, pair synergy bonuses are strictly computed from each player's real assigned slot score (`sum(p["score"] for p in assigned_stack) * syn_rate`), eliminating discrepancies from the previous hardcoded 1.6x slot estimate.
- **Dynamic US Eastern Daylight Saving Time**: Implemented `USEasternTz` in `nhl_api.py` with standard US DST transitions (2nd Sunday of March to 1st Sunday of November), ensuring correct slate dates and times across platforms lacking `zoneinfo`/`tzdata` (EDT UTC-4 in summer, EST UTC-5 in winter).
- **Expanded ESPN Injury Status Classification**: Defined `is_out_status` and `is_dtd_status` helper functions in `main.py` recognizing `Questionable`, `Doubtful`, `GTD`, `Game-Time Decision`, `Suspension`, and `Injured Reserve`, routing unboostered questionable players to DTD comments instead of false active status.
- **Thread-Isolated HTTP Sessions**: Converted `NHLClient.session` to thread-local storage (`threading.local`) preventing socket sharing and race conditions during parallel requests while maintaining mockability for tests.
- **Schedule and Injury Cache TTL**: Added default 3.0-hour cache TTL (`max_age_hours=3.0`) to `get_upcoming_schedule()` and `get_injuries()` in `NHLClient` and `Database`.
- **Generic 3-Word Name Alias Resolution**: Added automatic middle-name elimination (`words[0] words[2]`) in `NHLClient.get_team_lines` for robust mapping of Daily Faceoff rosters to NHL API namesakes.
- **Public Database Role Bias Summary API**: Added `Database.get_role_bias_summary()` returning settled empirical differences for PP1 and other deployment units, removing private `_get_connection()` calls from `show_history_report`.
- **Empirical Scoring Formula Calibration (`--calibrate`)**: Added `calibrate_rrs_formula` in `model.py` and `Database.get_settled_stats_for_calibration` in `db.py`, running Ordinary Least Squares (OLS) regression on settled `lineup_history` outcomes to reverse-engineer actual mobile app scoring weights ($w_G, w_A, w_S$), reporting $R^2$, MAE, and RMSE against model heuristics.
- **Look-Ahead Bias Elimination in Validation (`--train`)**: Strict point-in-time validation mode in `evaluate_model` eliminates leakage of future end-of-season standings into historical out-of-sample backtest splits.
- **Lineup-Level Financial Validation & Compounded Error Analysis**: Added `Database.get_lineup_level_validation_summary` reporting 5-player total Lineup MAE, Lineup RMSE, overall projection bias, beat-projection hit rate, and slot-by-slot / booster-compounded multiplier error breakdown.
- **93 Unit Tests**: Suite expanded to 93 tests passing with 100% success.

## [2.3.1] - 2026-10-18

### Added
- **Intra-Team Namesake Deployment Disambiguation**: Team line deployment in `db.py` and `nhl_api.py` is keyed by composite `(name_key, pos_group)` with database primary key `(team, player_name, pos_group)`. Created a custom `TeamLines(dict)` subclass supporting dual tuple and name-based lookups, guaranteeing distinct line and power play assignments for namesakes on the same club (e.g. Vancouver's Elias Pettersson C on L1/PP1 vs D on D3).
- **Conditional Power Play Penalty Guard**: In `compute_line_factor` and `PerformancePredictor.predict`, added `team_has_pp: bool` preventing the -8% goals / -12% assists penalty from being applied when the team's data lacks power play records.
- **Booster File Overwrite & Partial Sync Protection**: In `save_boosters_file`, guarded against overwriting `boosters.txt` when 0 skater stats are retrieved from the NHL API. Added section-level preservation for teams present in the existing file that fail to load during partial syncs.
- **Tournament Stack EV Floor Guarantee**: Initialized `solve_stacked_lineup` candidate pool with the DP optimal lineup (including naturally occurring teammate synergy), guaranteeing that tournament stack upside score $\ge$ standard DP lineup base score.
- **Stale Team Lines Invalidation**: `save_team_lines` executes `DELETE FROM team_lines WHERE team = ?` prior to inserting freshly fetched lines to clear scratched or waived players.
- **Historical Slate Backtest Isolation**: In `collect_slate_skaters` and `build_skater_projection`, line multipliers and tags are suppressed when `game_date < today_str`, preventing contemporary line deployment leakage into historical backtests.
- **Diagnostic Deployment Reporting**: Terminal output reports loaded and missing team deployment counts (`Lines: loaded N of M teams (missing: ...)`).
- **Flexible Value & Comment Parsing**: `parse_booster_val` and `load_boosters_file` strip `#` comments strictly to the right of `=` while preserving `#` in player names (`#25 Jack Eichel` or `Jack Eichel #25`). Added regex support for trailing decimal points (`1.`) and leading decimal points (`.5`).
- **Defensive Settle Fallback Real Blocks**: In `settle_past_lineups`, matches $\ge 3$ days old lacking Gamecenter boxscore protocols resolve blocked shots from `blockedShots` in the player's official game log (falling back to 0.0), eliminating arbitrary default blocks (1.0 for D / 0.3 for F).
- **Line Role Tracking in History**: Added `ev_line` and `pp_unit` to `lineup_history` with automatic table migration, displaying each skater's line tag in `--history [date]` and tracking empirical PP1 vs non-PP1 predictive bias.
- **Synergy Multiplier Calibration**: Explicitly calibrated teammate synergy bonuses: L1 gives +10% (`[{team} L1]`), L2 gives +8% (`[{team} L2]`), while L3/L4 pairs fall back to generic 6% (`[{team} Pair]`).
- **0 GP Season Query Throttling**: Added `players.updated_at` tracking in `touch_player_updated_at` to avoid re-querying missing historical seasons for zero-GP players within 24 hours.
- **81 Offline Unit Tests**: Expanded test suite to 81 tests, adding `TestQAFourthReview` covering all items in Blocks A and B as well as all 6 regression test cases with real function executions.

## [2.3.0] - 2026-10-18

### Added
- **Line Combinations & Power Play Deployment (PP1/PP2)**: Scrapes confirmed and projected line combinations and power play units from Daily Faceoff (`/teams/{slug}/line-combinations`) concurrently in parallel across all slate teams.
- **Role Factor Model Integration**: `compute_line_factor` modulates projected goals, assists, and ice time:
  - `PP1`: +12% goals, +18% assists, +1.5 min TOI.
  - `PP2`: +4% goals, +5% assists, +0.5 min TOI.
  - No PP (when EV line is confirmed): -8% goals, -12% assists, -0.5 min TOI.
  - EV Lines: `L1`/`D1` boost, `L3`/`L4`/`D3` realistic workload deflation.
- **Correlated Line & Power Play Stacking**: `solve_stacked_lineup` searches all candidate combinations per team and evaluates pairwise correlation:
  - Both on PP1 & same EV line (`[TEAM L1+PP1]`): +15% synergy.
  - Both on PP1 (`[TEAM PP1 QB+F]` or `[TEAM PP1]`): +13% synergy.
  - Same EV top line (`[TEAM L1]`): +10% synergy.
  - Same PP2 (`[TEAM PP2]`): +8% synergy.
  - Generic team stack (`[TEAM Pair]`): +6% synergy.
- **`Line` Column in Terminal Output**: Optimizer lineup tables display each skater's line tag (e.g. `L1/PP1`, `D1/PP1`, `L2/PP2`, `L3`) for immediate visibility.
- **Line Tags in `boosters.txt`**: Saved booster files append line and power play unit comments (`# L1/PP1`, `# D1/PP1`, `# L3`), accelerating manual multiplier entry.
- **`team_lines` SQLite Table & Cache**: Persists line combination data per team with a 12-hour TTL and batch lookup (`get_all_lines_for_teams`).
- **65 Offline Unit Tests**: Expanded test suite from 59 to 65 tests in `TestTeamLinesAndDeployment`, testing line caching, DFO JSON parsing, line multipliers, projection effects, PP1 stack synergy, and terminal table formatting.

## [2.2.1] - 2026-10-18

### Added
- **Bayesian Shrinkage for Low-GP Projections**: Implemented Bayesian regression shrinkage in `PerformancePredictor.predict` and baseline slate collection for skaters with $gp < 10$: $(stat \cdot gp + pos\_avg \cdot 10) / (gp + 10)$. Skaters with $gp = 0$ receive positional baseline averages, eliminating inflated small-sample forecasts for call-ups and allowing debutants with boosters to be drafted.
- **Robust Settle Fallback for Missing Boxscores**: In `settle_past_lineups`, matches older than 3 days whose Gamecenter boxscore protocol remains unavailable or empty are settled using official stats from the player's game log rather than remaining pending forever.
- **Automatic Missing Season Ingestion**: In `nhl_api.get_player_game_logs`, if a player's latest cached log predates the current season (prior to `YYYY-09-01`), all intervening and missing seasons are concurrently fetched and persisted to the database.
- **Activation Hints for Commented Boosters**: `load_boosters_file` now detects non-zero booster assignments in `# [DEPTH]` and `# [DTD]` commented lines, emitting an actionable console hint reminding users to remove `#` to activate the player.
- **Cross-Version Python Typing Fix**: Explicitly imported `Tuple` in `db.py`, ensuring backwards compatibility without `NameError` on Python 3.12 and 3.13 where type annotations are evaluated eagerly at import time (PEP 649 deferred annotation evaluation only applies starting in Python 3.14).
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
