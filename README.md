<!-- logo -->

# rl-market-maker

A PPO market maker trained in a limit order book simulator with real queue position and latency, and an honest comparison against Avellaneda-Stoikov and simpler quoting rules.

I've worked around market making for a long time, and I've read plenty of papers where an RL agent beats Avellaneda-Stoikov. Most of them train in a simulator where a passive order fills whenever the price touches it, with no queue and no latency. I wanted to know whether an agent trained in something closer to a real venue learns anything beyond what the closed-form models already say, or whether it just finds the quirks of its simulator. So I built one on top of my [lob](https://github.com/PDiTO/lob) matching engine, wrote PPO from scratch, calibrated the baselines to the same simulator, and measured all of them on markets none of them had seen.

Short version of what I found: <<SUMMARY>>

## Contents

- [Quickstart](#quickstart)
- [How it sits on top of lob](#how-it-sits-on-top-of-lob)
- [The environment](#the-environment)
- [PPO](#ppo)
- [Baselines and calibration](#baselines-and-calibration)
- [Training](#training)
- [Results](#results)
- [What the agent actually learned](#what-the-agent-actually-learned)
- [Out-of-regime stress tests](#out-of-regime-stress-tests)
- [Is this skill or a simulator quirk?](#is-this-skill-or-a-simulator-quirk)
- [Limitations](#limitations)
- [Tests](#tests)
- [Layout](#layout)

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and a Rust toolchain, because `uv sync` builds the simulator from source with maturin. The simulator is a sibling checkout:

```sh
git clone https://github.com/PDiTO/lob
git clone <this repo> rl-market-maker
cd rl-market-maker
uv sync                  # builds ../lob and installs torch, gymnasium, numpy, matplotlib
uv run pytest            # 66 tests, a few seconds on a laptop CPU
```

The dependency on lob is a uv path source. To use the GitHub repo instead of a sibling checkout, change one line in `pyproject.toml`:

```toml
lob = { path = "../lob", package = true }   # now
lob = { git = "https://github.com/PDiTO/lob" }   # instead
```

(`package = true` is only there because lob's own `pyproject.toml` marks it as a virtual project for its development workflow; a git source is built as a package anyway.)

Reproducing everything:

```sh
uv run rlmm train --config configs/ppo.toml            # 3 seeds, 20M steps each
uv run rlmm train --config configs/ppo_ablation.toml   # the ablation, 3 more seeds
uv run rlmm calibrate --config configs/eval.toml       # sigma, A, k, baseline tuning
uv run rlmm eval --config configs/eval.toml            # held-out comparison and stress tests
uv run rlmm plot --config configs/eval.toml            # figures into results/figures
```

`rlmm train --seed 2` trains a single seed, which is how I ran the seeds in parallel. Using the environment directly:

```python
import numpy as np
from rlmm.env import EnvConfig, LobMarketMakingEnv

env = LobMarketMakingEnv(EnvConfig(horizon_s=60.0))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(np.array([2, 2]))  # join both sides
```

The trained policy from seed 1 is in `results/policy/ppo_seed1.pt` (<<POLICY_SIZE>>). `rlmm.ppo.Agent.load` reads it.

## How it sits on top of lob

lob gives me three things through its Python bindings: `lob.OrderBook`, the price-time matching engine; `lob.generate_feed`, the synthetic order-by-order market; and `lob.run_backtest`, an event-driven backtester with order-entry and market-data latency.

The gap is that `run_backtest` owns its event loop. It runs the whole feed and calls a Python strategy back on timers and market data. An RL environment needs the opposite: the caller decides when the simulation advances, one decision at a time. There's also no way to see your own queue position through the strategy context, which only exposes L2.

I didn't want to fork lob, so `rlmm.sim.ReplayEngine` re-implements the backtester's loop in Python on top of `lob.OrderBook` and `lob.generate_feed`, with the same rules:

- feed messages reach the exchange at their own timestamps, our orders `order_latency_ns` after we send them;
- we see public data and our own acks and fills `market_data_latency_ns` after the exchange produced them;
- ties go to the feed first, then everything else in the order it was scheduled;
- accounting is lob's exact integer micro-tick scheme with average-cost realized PnL, ported line for line (including Rust's truncating integer division).

The part I care most about is that it's tested against the real thing. `tests/test_engine.py` runs the same quoting logic twice, once as a Python strategy inside `lob.run_backtest` and once on the replay engine, under five latency settings, and requires every fill (time, order id, side, price, size, mid before) and the final cash, position, fees and realized PnL to match exactly. They do.

On top of lob's semantics the engine tracks the exact queue position of each of our resting orders, meaning lots ahead of it at its price. It updates that from the matching events rather than by scanning L3, which costs about 75 microseconds per call through the bindings. A test checks the incremental number against a full L3 recount at every decision. The whole environment runs at roughly 20,000 steps a second on one core.

## The environment

`LobMarketMakingEnv` is a Gymnasium environment. One step is 100 ms of market time. An episode is a 60-second session, so 600 decisions, after a 5-second warm-up so the book is in steady state. Each episode gets a fresh synthetic market from a seed, and everything is deterministic given `reset(seed=...)`. Training, calibration and evaluation draw their market seeds from disjoint ranges.

The market is lob's default synthetic market: Poisson limit orders around a latent fair value that follows a random walk with volatility 2 ticks per square-root second, 10 market orders a second of which half are informed (they trade toward fair value), and per-order cancels. Latency is 50 microseconds each way, the maker rebate 0.1 ticks per lot and the taker fee 0.3. Our orders always join the back of their queue.

### Observation

Everything the agent sees is what a real system would know at that moment: the book as of market-data latency ago, and its own position and orders rebuilt only from acks and fills that have arrived. 24 features, each clipped to a fixed range:

| group | features |
|---|---|
| own state | inventory / max position, share of the session left |
| book | spread, top-of-book imbalance, five-level imbalance, microprice minus mid |
| recent prices | mid change over the last 1, 10 and 50 steps, EWMA volatility of 1-step changes |
| trade flow | aggressor imbalance and log volume over the last step |
| own quotes, per side | quoting or not, ticks behind the touch, log of lots ahead in the queue, share of the level ahead, open size |
| own fills | lots bought and sold in the last step |

Queue position is in there because it's the thing a real market maker watches and a touch-fill simulator can't represent. An order at the back of a 40-lot queue and one at the front of a 3-lot queue at the same price are very different bets.

### Action

`MultiDiscrete([5, 5])`: one choice per side from {no quote, one tick inside the touch, at the touch, one tick behind, two ticks behind}, at a fixed size of 5 lots. "Inside" joins the touch when the spread is one tick, since it would otherwise cross.

I went with discrete levels relative to the touch rather than continuous offsets from mid for three reasons. The tick is coarse here (the spread is one tick about two thirds of the time), so continuous offsets would be rounded onto the same handful of prices anyway. Levels relative to the touch map directly onto queue decisions: stay where you are, rejoin the back of the touch, or step away. And two small categorical heads are easy for PPO; skew falls out of choosing the two sides differently. I left size fixed. With a hard inventory cap and an inventory penalty, size mostly scales PnL and risk together, and adding it would square the action space.

An order already resting at the chosen price is left alone, so the agent keeps its queue place by choosing the same level again. Anything else is cancelled and replaced with a post-only order at the back of the new queue. A side is not quoted if a fill would take the known position past 50 lots. That limit uses the delayed position, so in-flight fills can overshoot by one quote, as they would in production.

### Reward

$$
r_t = \Delta \mathrm{PnL}_t \;-\; \phi\, q_t^2\, \Delta t \;-\; \eta \max\!\left(0,\; q_{t-1}\, \Delta m_t\right) \;-\; \mathbb{1}[t = T]\; C_{\mathrm{liq}}(q_T)
$$

- $\Delta \mathrm{PnL}_t$ is the change in mark-to-market PnL at the exchange mid, including rebates, from the simulator's own accounting.
- $\phi q^2 \Delta t$ is a running quadratic inventory penalty, the discrete form of the $\phi \int q^2\,dt$ term in Cartea and Jaimungal's formulation. I used $\phi = 0.005$ ticks per lot squared per second.
- The $\eta$ term is asymmetric dampening (Spooner et al., 2018). It removes a share of the gains that came from the mid moving in favour of the inventory, but none of the losses, which discourages the agent from speculating with inventory. It's implemented and tested but off ($\eta = 0$) in the runs below.
- $C_{\mathrm{liq}}$ is the cost of flattening the final position with a market order against the visible book at the end, measured from mid and including the taker fee.

Since time left is in the observation, the end of a session is a true terminal state, not a truncation.

## PPO

PPO is written out in `src/rlmm/ppo/`, with no RL library. For a batch of transitions with probability ratio $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)$ and advantage estimate $\hat A_t$, the loss minimised is

$$
L(\theta) = -\,\mathbb{E}_t\!\left[\min\!\big(\rho_t \hat A_t,\; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, \hat A_t\big)\right] \;+\; c_v\, L^V(\theta) \;-\; c_e\, \mathbb{E}_t\!\left[\mathcal{H}\big(\pi_\theta(\cdot \mid s_t)\big)\right]
$$

with the value loss optionally clipped,

$$
L^V = \tfrac12\, \mathbb{E}_t\!\left[\max\!\Big(\big(V_\theta(s_t) - R_t\big)^2,\; \big(V_{\mathrm{old}}(s_t) + \mathrm{clip}(V_\theta(s_t) - V_{\mathrm{old}}(s_t), -\epsilon_v, \epsilon_v) - R_t\big)^2\Big)\right],
$$

and advantages from GAE($\lambda$), where termination and truncation are handled differently:

$$
\delta_t = r_t + \gamma\,(1 - d^{\mathrm{term}}_t)\, V(s_{t+1}) - V(s_t), \qquad
\hat A_t = \delta_t + \gamma \lambda\, (1 - d^{\mathrm{end}}_t)\, \hat A_{t+1}, \qquad R_t = \hat A_t + V(s_t).
$$

On a truncation, $V(s_{t+1})$ is the value of the final observation, not of the reset observation that replaced it. The vector env returns final observations separately for exactly this.

The rest of the standard recipe (Huang et al., "The 37 implementation details of PPO", is a good checklist):

- 16 synchronous environments with same-step autoreset;
- separate policy and value MLPs, 2 x 128 tanh, orthogonal init, policy head at gain 0.01 so it starts near uniform;
- a factorised categorical policy, one head per side, log-probabilities and entropies summed;
- running mean and variance normalisation of observations (Chan et al.'s parallel update) and scaling of rewards by the running std of the discounted return;
- per-minibatch advantage normalisation, minibatch epochs, global gradient-norm clipping, Adam with linear learning-rate annealing;
- logging of approximate KL ($\mathbb{E}[(\rho - 1) - \log \rho]$), clip fraction, explained variance, entropy and gradient norm, plus a validation run of the greedy policy on 16 fixed markets every 5 updates.

| setting | value |
|---|---|
| steps per seed | 20M (2,441 updates of 8,192) |
| rollout | 16 envs x 512 steps |
| minibatches x epochs | 8 x 5 |
| learning rate | 3e-4, annealed to 0 |
| $\gamma$, $\lambda$ | 0.99, 0.95 |
| clip $\epsilon$ | 0.2 (value clipping off) |
| entropy, value coefficients | 0.005, 0.5 |
| max gradient norm | 0.5 |

## Baselines and calibration

### Avellaneda-Stoikov

In Avellaneda and Stoikov (2008), the mid follows $dS = \sigma\, dW$, the market maker maximises CARA utility of terminal wealth with risk aversion $\gamma$, and a quote at distance $\delta$ from the mid fills at Poisson rate $\lambda(\delta) = A e^{-k\delta}$. The approximate solution is a reservation price and a spread around it:

$$
r_t = s_t - q_t\, \gamma \sigma^2 (T - t), \qquad
\delta^a_t + \delta^b_t = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\!\left(1 + \frac{\gamma}{k}\right).
$$

The skew per lot, $\gamma\sigma^2(T-t)$, is largest at the start of a session and vanishes at the end. Guéant, Lehalle and Fernandez-Tapia (2013) give the long-horizon limit, where the quotes stop depending on $T - t$:

$$
\delta^b(q) = c_1 + \frac{2q + 1}{2} c_2, \qquad \delta^a(q) = c_1 - \frac{2q - 1}{2} c_2, \qquad
c_1 = \frac{1}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right), \quad
c_2 = \sqrt{\frac{\sigma^2 \gamma}{2kA}\left(1 + \frac{\gamma}{k}\right)^{1 + k/\gamma}}.
$$

I run both: `avellaneda_stoikov` on the 60-second session horizon, and `as_stationary` with the constant skew. Model prices are rounded to the nearest tick, ties to the passive side, and kept from crossing the visible touch because the orders are post-only.

### Two simpler rules

- `symmetric` quotes a fixed number of ticks behind the best bid and ask whatever the inventory.
- `linear_skew` quotes half a tick either side of $m - \kappa q$, which joins the touch when flat and backs off the side that would add to inventory. `microprice_skew` is the same rule centred on the size-weighted microprice instead of the mid, which is the simplest way to give a hand-written quoter the book-imbalance signal the agent can see.

### Calibration

All calibration runs on its own range of market seeds (`rlmm calibrate`, output in `results/calibration.json`).

**Volatility.** The realised volatility of the exchange mid depends on the horizon you measure it at:

| horizon | 0.1 s | 1 s | 5 s | 10 s |
|---|---|---|---|---|
| $\hat\sigma$, ticks per $\sqrt{\mathrm{s}}$ | 1.23 | 1.35 | 1.80 | 1.96 |

The generator's fair value moves at 2.0. The mid is much quieter at short horizons because it lags the fair value and only moves in whole ticks, and the rising ratio is positive autocorrelation: moves continue. That will matter later. AS assumes a diffusive mid, so I use the 5-second figure, $\sigma = 1.80$.

**Fill intensity.** I didn't want to estimate $A$ and $k$ from trade prints, because that ignores the queue: a print at your price doesn't mean you got filled. Instead a probe strategy quotes a fixed number of ticks behind the touch on both sides, in the full simulator, across 30 markets per level, and I measure the fill rate in lots per quoted lot per second against the time-averaged distance of the quote from the mid:

| ticks behind the touch | mean distance from mid | fill rate | lots filled |
|---|---|---|---|
| 0 | 0.65 | 1.22 | 21,973 |
| 1 | 1.72 | 0.077 | 1,383 |
| 2 | 2.73 | 0.0022 | 39 |
| 3 | 3.73 | 0 | 0 |

A weighted least-squares fit of $\log \lambda = \log A - k\delta$ (weights equal to the fill counts, since the variance of a log Poisson rate is about one over the count) gives $A = 6.84$ and $k = 2.63$ per tick.

![fill intensity](results/figures/calibration.png)

**Tuning.** Each baseline has one free parameter, which I picked by grid search on 200 calibration markets using mean training reward, the same objective the agent optimises:

<<TUNING_TABLE>>

## Training

<<TRAINING>>

## Results

<<RESULTS>>

## What the agent actually learned

<<LEARNED>>

## Out-of-regime stress tests

<<STRESS>>

## Is this skill or a simulator quirk?

<<ANALYSIS>>

## Limitations

<<LIMITATIONS>>

## Tests

<<TESTS>>

## Layout

```text
src/rlmm/
  sim/            replay engine on lob, exact accounting port, agent-side order view
  env/            LobMarketMakingEnv, observation features, config
  ppo/            GAE, losses, normalisers, actor-critic, vector env, trainer, agent
  baselines/      Avellaneda-Stoikov, stationary AS, heuristics, calibration
  evaluation/     episode metrics, bootstrap CIs, parallel runner, reports
  cli.py          rlmm train | calibrate | eval | plot
  plotting.py     figures
configs/          ppo.toml, ppo_ablation.toml, eval.toml
results/          calibration, evaluation tables, training logs, figures, one policy
tests/            pytest suite
```

## License

MIT. See [LICENSE](LICENSE).
