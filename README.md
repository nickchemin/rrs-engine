# RRS Engine

Quantitative draft optimizer and sports analytics terminal for NHL contests on the **Real Sports** mobile app. Engineered with Generalized Linear Models (Poisson & Ridge Regression), schedule fatigue analytics, opponent penalty kill modulation, vectorized Monte-Carlo covariance simulation, exact $O(5N)$ dynamic programming lineup optimization, fault-tolerant scraping with TOI line fallback, and automatic draft screenshot OCR parsing.

---

## Key Features

- **Advanced Feature Engineering & xRRS Projections**:
  - Predicts individual skater performance using Poisson GLM (log-link) for discrete scoring counts (goals and assists) and Ridge Regression for continuous time-on-ice (`xTOI`).
  - Incorporates schedule fatigue metrics: home ice advantage (`is_home`), back-to-back fatigue (`is_b2b`), rest days differential (`rest_days`), and opponent fatigue status (`opp_is_b2b`).
  - Modulates power play units (`PP1`/`PP2`) dynamically based on opponent penalty kill efficiency (`opp_pk_pct`) and penalty frequency (`opp_times_shorthanded`).
  - Applies Bayesian shrinkage against positional priors for skaters with small sample sizes ($GP < 10$).
- **Vectorized Monte-Carlo Covariance Simulation (`--stack`)**:
  - Simulates 10,000 correlated slate iterations in $< 20$ ms via Cholesky decomposition of empirical covariance matrix $\Sigma = D R D$.
  - Models offensive teammate co-occurrence across deployment tiers (`L1+PP1`, `PP1 QB + Forward`, `L1 linemates`).
  - Computes actionable distribution quantiles:
    - **Expected Score (Mean)**: mathematical payout expectation.
    - **Floor (25th percentile)**: downside risk scenario for Cash games.
    - **Ceiling (90th / 95th percentile)**: explosive upside potential for GPP tournaments.
- **Exact Dynamic Programming Optimizer**:
  - Solves the optimal 5-slot booster configuration in $O(5N)$ using the rearrangement inequality in $< 1$ ms.
  - Supports configurable slot multipliers (e.g. `--slots 2.0,1.8,1.6,1.4,1.2`).
- **Scraping Fault Tolerance & Network Resilience**:
  - Centralized retry mechanism with exponential backoff on transient HTTP status codes (`429`, `500`, `502`, `503`, `504`) and network dropouts.
  - Automatic line combination fallback: if Daily Faceoff fails or alters HTML, dynamically reconstructs forward lines (`L1-L4`), defense pairings (`D1-D3`), and power play units (`PP1/PP2`) from official NHL stats API time-on-ice profiles (`skater/timeonice`).
- **Draft Screenshot OCR & Booster Auto-Fill (`--parse-images`)**:
  - Automatically extracts player names and booster multipliers (`+0.4`, `1.2x`, `+20%`, `0.8`, `3.0`) from screenshots of the Real Sports mobile app draft screen.
  - High-performance fuzzy matching via `RapidFuzz` (with `difflib` fallback) matching noisy OCR text to active game-day skaters.
  - Handles namesakes and duplicate player disambiguation (e.g., `Elias Pettersson (D)` vs `Elias Pettersson (C)`).
  - Automatically updates `boosters.txt` preserving team section formatting and syncs directly to the SQLite database.
- **Settlement & Scoring Calibration**:
  - Fetches official NHL boxscores to settle historical contests.
  - Re-calibrates scoring formula weights via OLS regression (`--calibrate`) against ground-truth mobile app payouts (`app_rrs` and `app_payout`).
  - Out-of-sample backtest suite (`--train`) benchmarking rolling averages, Ridge regression, and Poisson GLM side-by-side with zero look-ahead bias.

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/nickchemin/rrs-engine.git
cd rrs-engine
pip install -r requirements.txt
```

### 2. Sync Slate
```bash
python main.py --sync
```
Fetches today's NHL schedule, matches starting goalies, synchronizes line combinations, filters injured players, and generates partitioned `boosters.txt`.

### 3. Set Boosters

#### Option A: Automatic Screenshot Parsing (Recommended)
Take screenshots of the draft screen in the Real Sports app, place them in a folder (e.g. `screenshots/`), and run:
```bash
python main.py --parse-images screenshots
```
The engine will run OCR, match players via fuzzy search, and automatically populate `boosters.txt` and the database.

#### Option B: Manual Editing (`boosters.txt`)
Edit `boosters.txt` directly (supports percentages, multipliers, decimals, and commas):
```text
Nathan MacKinnon         = 0.0  # L1/PP1
Cale Makar               = 0.4  # D1/PP1
Rickard Rakell           = 2.2x # L2/PP2
Egor Chinakhov           = 80%  # Evaluates to 0.80
Brent Burns              = 1,5  # Evaluates to 1.50 | D2
Sam Malinski             = .5   # Evaluates to 0.50
Elias Pettersson (D)     = 0.0  # VAN D | D2
```

### 4. Optimize Lineups
**Windows**:
```bash
python main.py
```
*(or double-click `run.bat`)*

**Linux / macOS**:
```bash
chmod +x run.sh && ./run.sh
```

### 5. Run Offline Test Suite
```bash
pytest -v
# or
python -m unittest discover -s tests -v
```

---

## Command Reference

| Command | Description |
| :--- | :--- |
| `python main.py` | Generates standard optimal lineup and correlated tournament stack with Monte-Carlo distribution metrics. |
| `python main.py --stack` | Solves exclusively for correlated tournament stacks. |
| `python main.py --parse-images <folder>` | Runs OCR on mobile app screenshots from folder, fuzzy matches active players, and auto-populates `boosters.txt`. |
| `python main.py --sync` | Fetches active NHL roster, starting goalies, lines, and initializes/refreshes `boosters.txt`. |
| `python main.py --date YYYY-MM-DD` | Runs optimizer for a specific slate date. |
| `python main.py --sync --date YYYY-MM-DD` | Syncs rosters and generates `boosters.txt` for a specific historical or future slate date. |
| `python main.py --history [date\|all]` | Performance tracking report with cumulative payout, ROI, and settled contest statistics. |
| `python main.py --record [date]` | Records verified ground-truth Real Sports mobile app ratings (`app_rrs`) and payouts (`app_payout`). |
| `python main.py --slots 2.0,1.8,...` | Runs optimizer with custom slot multipliers (default: `2.0,1.8,1.6,1.4,1.2`). |
| `python main.py --calibrate` | OLS empirical regression reverse-engineering app scoring weights and verifying additive vs multiplicative mechanics. |
| `python main.py --train` | Runs comparative multi-model benchmark (Rolling Average, Ridge, Poisson GLM) with strict point-in-time cross-validation. |

---

## License

This project is licensed under the [MIT License](LICENSE).
