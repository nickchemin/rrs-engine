# RRS Engine

Quantitative draft optimizer and sports analytics terminal for NHL contests on the **Real Sports** app. Built with Ridge Regression machine learning, exact dynamic programming lineup optimization, starting goalie matchup tracking, and correlated teammate stacking.

## Features

- **xRRS Calculation**: Predicts player performance based on historical stats, opponent defense metrics, starting goalie save percentages, line combinations (EV Line 1-4, D1-3), and Power Play deployment (PP1/PP2), with Bayesian shrinkage for small-sample skaters ($gp < 10$).
- **Exact DP Optimizer**: Evaluates optimal 5-slot lineups using exact $O(5N)$ dynamic programming obeying the rearrangement inequality in $<1$ ms.
- **Line Deployment Intel**: Scrapes and caches live line combinations and power play units (PP1, PP2, EV Lines) from Daily Faceoff in parallel across all slate teams.
- **Correlated Teammate Stacks**: Solves for highly correlated combinations (same PP1 unit, PP1 QB + Forward, L1+PP1 linemates) with dynamic synergy multipliers to maximize tournament ceiling.
- **Starting Goalie Intel**: Scrapes confirmed starting netminders from Daily Faceoff with slate matchup verification, multi-date composite caching, and official NHL API fallback.
- **Automated Settlement**: Fetches completed match boxscores from NHL Gamecenter (`/boxscore`) to track actual goals, assists, shots, and blocked shots, with automatic DNP handling and resilient game log fallback for games $\ge 3$ days old.
- **Partitioned Boosters**: Formats `boosters.txt` grouped by team with line combination comments (e.g. `# L1/PP1`), automatic name deduplication (e.g., Sebastian Aho F vs. D, Elias Pettersson C vs. D), activation hints for depth players, and flexible booster parsing (supports `80%`, `0.8`, `1,5`, `1.4x`).
- **NHL Timezone Handling**: Accurately aligns slate dates using America/New_York time across European and Asian timezones.
- **Out-of-sample ML Validation**: Tests prediction models against a 20-game rolling average baseline.

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
Downloads today's NHL schedule, matches starting goalies, synchronizes line combinations, filters injuries, and generates `boosters.txt`.

### 3. Set Boosters (`boosters.txt`)
Fill in player multipliers from the Real Sports app (supports percentages, multipliers, and decimals; comments indicate player lines and PP units):
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

### 5. Run Offline Tests
```bash
python -m unittest discover -s tests -v
```

## Command Reference

| Command | Description |
| :--- | :--- |
| `python main.py` | Generates standard optimal and tournament stack lineups. |
| `python main.py --stack` | Solves exclusively for correlated teammate stacks. |
| `python main.py --sync` | Fetches active NHL roster, starting goalies, and refreshes `boosters.txt`. |
| `python main.py --date YYYY-MM-DD` | Runs optimizer for a specific slate date. |
| `python main.py --sync --date YYYY-MM-DD` | Syncs rosters and generates `boosters.txt` for a specific date. |
| `python main.py --history` | Performance tracking table with cumulative profit and settled status. |
| `python main.py --history YYYY-MM-DD` | Detailed box score breakdown for a specific date. |
| `python main.py --calibrate` | Runs OLS empirical regression to reverse-engineer app scoring weights from settled history. |
| `python main.py --train` | Runs out-of-sample ML and lineup-level validation (strictly point-in-time, no look-ahead bias). |

## License

This project is licensed under the [MIT License](LICENSE).
