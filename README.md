<div align="center">

```text
  ██████╗ ██████╗ ███████╗   RRS ENGINE v2.0
  ██╔══██╗██╔══██╗██╔════╝   NHL Quant Draft Optimizer • Real Rating System
```

# RRS Engine — Real Rating System for NHL Drafts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Real%20Sports%20App-orange.svg)](https://realsports.io)
[![Status: Production](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

**RRS Engine** is a high-performance quantitative sports analytics terminal and lineup optimizer engineered specifically for the **Real Sports** mobile app. Powered by Ridge Regression machine learning, starting goalie matchup tracking, and covariance teammate stacking.

</div>

---

## ⚡ Why RRS Engine?

Unlike generic fantasy tools calibrated for legacy DFS platforms, **RRS Engine** is natively built for **Real Sports**:

* **Calibrated xRRS Scale**: Predicts exact unmultiplied base points ($G \approx 1.67$, $A \approx 1.11$, $S \approx 0.22$, $BLK \approx 0.22$) with a built-in variance haircut, ensuring projected payouts match real leaderboard reality ($xRRS \times \text{Total Boost} = \text{RRS Payout}$).
* **Starting Goalie Intelligence**: Scrapes confirmed starting netminders and save percentages from Daily Faceoff with official NHL API fallback. Adjusts scoring expectations according to opposing goaltender strength.
* **Correlated Teammate Stacks**: Identifies high-upside 2-man and 2+2 teammate combinations against vulnerable goalies to maximize tournament ceiling.
* **Automated Result Settlement**: Pulls boxscores from completed matches, settles player stats ($G, A, S, TOI$), and calculates actual vs. projected profit.
* **Zero Configuration Setup**: Auto-initializes local SQLite database and game-day rosters on first run.

---

## 📊 Terminal Preview

```text
==============================================================================================================
Date: 2026-10-16 | Projected: 34.9 RRS | Mode: Lineup #1: Standard Optimal (Max Base EV)
==============================================================================================================
Slot  Slot Boost  Player                 Team  Opp (Goalie)           xRRS     Booster   Total   RRS Payout    Synergy
--------------------------------------------------------------------------------------------------------------
#1    2.0x        Nathan MacKinnon       COL   vsPIT (A. Silovs)      4.0      +0.0x     2.0x    8.1 RRS       [COL Pair]
#2    1.8x        Rickard Rakell         PIT   @COL (M. Blackwood)    1.8      +2.2x     4.0x    7.2 RRS       [PIT Pair]
#3    1.6x        Egor Chinakhov         PIT   @COL (M. Blackwood)    1.7      +2.7x     4.3x    7.2 RRS       [PIT Pair]
#4    1.4x        Brent Burns            COL   vsPIT (A. Silovs)      1.6      +2.8x     4.2x    6.5 RRS       [COL Pair]
#5    1.2x        Sam Malinski           COL   vsPIT (A. Silovs)      1.4      +3.0x     4.2x    5.8 RRS       [COL Pair]
--------------------------------------------------------------------------------------------------------------
⚡ Quant Summary: Top EV driver Nathan MacKinnon (2.0x Boost) | Synergistic [COL Pair], [PIT Pair] upside stack
==============================================================================================================
```

---

## 🚀 Quickstart (1 Minute)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/nickchemin/rrs-engine.git
cd rrs-engine
pip install -r requirements.txt
```

### 2. Sync Today's Active Slate
```bash
python main.py --sync
```
*This downloads the daily NHL schedule, matches starting goalies, filters injuries, and saves active skaters to `boosters.txt`.*

### 3. Input Your Boosters (`boosters.txt`)
Open `boosters.txt` and fill in the player multipliers from your Real Sports app:
```text
Nathan MacKinnon         = 0.0
Rickard Rakell           = 2.2
Egor Chinakhov           = 2.7
Brent Burns              = 2.8
Sam Malinski             = 3.0
```
*(Supports `=`, `:`, whitespace, `+`, `x`, or `%` notation).*

### 4. Generate Winning Lineups
**Windows**: Double-click `run.bat` or run:
```bash
python main.py
```
**macOS / Linux**:
```bash
chmod +x run.sh && ./run.sh
```

---

## 🛠️ Command Reference

| Command | Description |
| :--- | :--- |
| `python main.py` | Generates Standard Optimal and Tournament Stack lineups; auto-saves to tracking history. |
| `python main.py --stack` | Solves exclusively for correlated teammate stacks with maximum upside synergy. |
| `python main.py --sync` | Fetches today's live NHL roster, starting goalies, injuries, and refreshes `boosters.txt`. |
| `python main.py --sync --date YYYY-MM-DD` | Syncs rosters for a specific target game date. |
| `python main.py --history` | Performance tracking table: settles finished games and displays cumulative RRS profit. |
| `python main.py --history YYYY-MM-DD` | Detailed box score breakdown ($G, A, S, TOI$) for every drafted player on that date. |
| `python main.py --train` | Runs out-of-sample machine learning validation (MAE metrics on historical games). |

---

## 📐 Mathematical Formulation

### 1. Multiplier Stacking
In Real Sports, each draft slot has a base multiplier ($2.0x, 1.8x, 1.6x, 1.4x, 1.2x$). The effective multiplier is:
$$\text{Total Boost} = \text{Slot Boost} + \text{Player Booster}$$

### 2. Calibrated Base Rating ($xRRS$)
$$xRRS = \frac{12.0 \cdot xG + 8.0 \cdot xA + 1.6 \cdot xSOG + 1.6 \cdot xBLK}{7.2}$$
*Where components are predicted via Ridge Regression trained on player game history, adjusted for opponent defense metrics, home/away splits, and starting goalie save percentages.*

### 3. Slot Payout
$$\text{RRS Payout} = xRRS \times \text{Total Boost}$$
*Total lineup score is the direct sum across all 5 slots.*

---

## 🔮 Roadmap & Pro Features

We are actively developing premium extensions for serious players:

- [ ] **Telegram & Discord Live Bot**: Automated morning slate briefings, confirmed goalie alerts, and optimal picks directly to your phone.
- [ ] **Interactive Web UI**: Visual dashboard for drag-and-drop lineup building and booster management.
- [ ] **Monte Carlo Simulation Engine**: 10,000+ game simulations per slate for percentile distribution modeling.
- [ ] **Multi-Entry Portfolio Optimizer**: Solves for 5–20 diversified lineups with controlled player exposure.

Interested in early access or partnership? Open an Issue or reach out via GitHub Discussions.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
