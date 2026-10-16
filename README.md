# RRS Engine

Quantitative draft optimizer and sports analytics terminal for NHL contests on the **Real Sports** app. Built with Ridge Regression machine learning, starting goalie matchup tracking, and correlated teammate stacking.

## Features

- **xRRS Calculation**: Predicts player performance based on historical stats, opponent defense metrics, and starting goalie save percentages.
- **Starting Goalie Intel**: Scrapes confirmed starting netminders from Daily Faceoff with official NHL API fallback.
- **Teammate Stacks**: Solves for correlated 2-man and 2+2 combinations to maximize tournament ceiling.
- **Automated Settlement**: Fetches completed match boxscores and calculates actual vs. projected payouts.

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
Downloads today's NHL schedule, matches starting goalies, filters injuries, and generates `boosters.txt`.

### 3. Set Boosters (`boosters.txt`)
Fill in player multipliers from the Real Sports app:
```text
Nathan MacKinnon         = 0.0
Rickard Rakell           = 2.2
Egor Chinakhov           = 2.7
Brent Burns              = 2.8
Sam Malinski             = 3.0
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

## Command Reference

| Command | Description |
| :--- | :--- |
| `python main.py` | Generates standard optimal and tournament stack lineups. |
| `python main.py --stack` | Solves exclusively for correlated teammate stacks. |
| `python main.py --sync` | Fetches active NHL roster, starting goalies, and refreshes `boosters.txt`. |
| `python main.py --sync --date YYYY-MM-DD` | Syncs rosters for a specific date. |
| `python main.py --history` | Performance tracking table with cumulative profit. |
| `python main.py --history YYYY-MM-DD` | Detailed box score breakdown for a specific date. |
| `python main.py --train` | Runs out-of-sample machine learning validation on historical games. |

## License

This project is licensed under the [MIT License](LICENSE).
