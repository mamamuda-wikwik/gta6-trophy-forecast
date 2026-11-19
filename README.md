# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mamamuda-wikwik/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mamamuda-wikwik/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI PS5 trophy's earn rate on Exophase at 2026-11-26 (one week after launch) as five quantiles per trophy, from the case's own read history. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

A trophy's earn rate is earners / players-tracked. Exophase refreshes its
counts on its own schedule, so a read shows the rates as of its last
refresh: between refreshes the tracked base grows while earners only
grow, which is why rates can fall at a refresh. `main.py` fits the two
processes separately — earners and tracked base, both as power laws in
log-time since launch — then divides at the target. Per-trophy slopes
shrink toward class priors (story chapters rise fast, grind slow,
platinum slowest). With a flat history (one distinct page) it uses a
horizon-aware dilution prior instead: a short horizon from launch night
expects the next refresh to dilute rates.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=tdgxqn9n&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
