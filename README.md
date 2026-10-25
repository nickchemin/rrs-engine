# RRS Engine

Command-line projection and lineup optimizer for NHL drafts on the Real Sports app.
It collects schedule, roster, injury, goalie, and line data, projects player
performance, solves a five-slot lineup, and tracks settled results over time.

## Requirements

- Python 3.10 or newer.
- Network access to the NHL API, ESPN, and Daily Faceoff.
- Tesseract OCR is required only for `--parse-images`.

Tesseract installation:

- Windows: install the UB Mannheim build and add its directory to `PATH`, or set
  `TESSERACT_CMD` to the full path of `tesseract.exe`.
- macOS: `brew install tesseract`
- Debian/Ubuntu: `sudo apt install tesseract-ocr`

The `pytesseract` package is a wrapper; it does not include the binary.

## Installation

```bash
git clone https://github.com/nickchemin/rrs-engine.git
cd rrs-engine
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux and macOS
pip install -r requirements.txt
```

## Quickstart

1. Sync the current slate and generate a boosters file:

```bash
python main.py --sync
```

2. Fill in booster multipliers. Either edit `boosters.txt` directly (use
   `boosters.example.txt` as the format reference) or parse screenshots of the
   draft screen:

```bash
python main.py --parse-images screenshots
```

3. Generate lineups:

```bash
python main.py
```

The command prints a standard lineup and a correlated tournament lineup with
projected payout quantiles.

## Data

Runtime state is stored in the user data directory:

- Windows: `%LOCALAPPDATA%\rrs-engine`
- macOS: `~/Library/Application Support/rrs-engine`
- Linux: `$XDG_DATA_HOME/rrs-engine` or `~/.local/share/rrs-engine`

The directory contains:

- `data.db` - SQLite database with players, game logs, lines, standings, and
  lineup history.
- `boosters.txt` - player booster values, partitioned by team.

Set `RRS_DATA_DIR` to use a different directory. On first start, `data.db` and
`boosters.txt` found next to the source files are copied into the data
directory once; existing files are never overwritten.

Both data files are ignored by git. Back them up before changing machines.

## Commands

| Command | Description |
| :--- | :--- |
| `python main.py` | Standard lineup plus tournament stack with payout quantiles. |
| `python main.py --stack` | Tournament stack only. |
| `python main.py --sync` | Fetch schedule, rosters, goalies, injuries, and lines; write `boosters.txt`. |
| `python main.py --parse-images <folder>` | OCR draft screenshots and merge detected boosters. |
| `python main.py --date YYYY-MM-DD` | Target a specific slate date available in the cached or current schedule window. |
| `python main.py --history [date\|all]` | Print settled results, app payouts, and projection bias. |
| `python main.py --record [date]` | Record Real Sports app ratings and payouts for a saved lineup. |
| `python main.py --calibrate` | Fit scoring weights by OLS against recorded app ratings. |
| `python main.py --train` | Run the out-of-sample model benchmark. |
| `python main.py --slots 2.0,1.8,1.6,1.4,1.2` | Override slot multipliers. |
| `python main.py --verbose` | Enable debug logging for data fetching. |
| `python main.py --quiet` | Suppress warning and info logs. |

Diagnostic logs are written to stderr. Lineup tables and reports stay on stdout,
so piping the report does not mix in log lines.

`--date` is limited to dates present in the current schedule payload or the
local cache. Older historical dates require previously synced data.

## Model Notes

- Player projections use per-player Poisson and Ridge models trained on prior
  games. Poisson targets are events per 60 minutes weighted by time on ice;
  the predicted TOI converts rates back to per-game counts.
- Rolling form, opponent defense, home ice, and deployment are inputs.
- The default xRRS formula is a heuristic. `--calibrate` fits an OLS model
  against recorded app ratings; fitted weights are stored in the database and
  used by later projections only when at least 15 ground-truth records exist.
- The tournament simulation uses a configured correlation matrix and
  role-based coefficients of variation. Its quantiles are model output, not
  empirical guarantees.
- The backtest uses standings snapshots recorded during sync. When no snapshot
  exists for a cutoff date, opponent features are neutralized and the report
  says so.
- Settlement estimates actual scores from boxscore events. When app payouts
  are recorded with `--record`, those values take precedence for calibration.

## Development

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m coverage run --source=db,image_parser,main,model,nhl_api,optimizer,paths,slate -m unittest discover -s tests
python -m coverage report
```

CI runs the test suite on Python 3.10-3.13 for Linux and Windows and enforces a
minimum coverage level defined in `pyproject.toml`.

## License

MIT. See [LICENSE](LICENSE).
