Evaluated on 500 held-out markets, 60 s each. PnL in ticks x lots, net of fees and of flattening the final position. Brackets are 95% bootstrap confidence intervals for the mean.

| policy | PnL mean [95% CI] | PnL sd | PnL mean/sd | Sharpe, 1 s | max drawdown | mean abs inventory | reward [95% CI] |
|---|---|---|---|---|---|---|---|
| symmetric | 44.1 [30.9, 57.8] | 153.4 | 0.29 | 0.081 ± 0.187 | 119.8 ± 107.3 | 7.8 ± 4.9 | 10.2 [-3.5, 24.0] |
| linear_skew | 139.7 [134.8, 144.3] | 52.5 | 2.66 | 0.452 ± 0.184 | 29.5 ± 19.6 | 2.8 ± 0.2 | 136.0 [131.2, 140.6] |
| microprice_skew | 98.7 [95.0, 102.5] | 43.0 | 2.30 | 0.377 ± 0.174 | 26.5 ± 15.6 | 2.3 ± 0.2 | 96.1 [92.3, 99.8] |
| avellaneda_stoikov | 37.6 [32.1, 43.0] | 62.3 | 0.60 | 0.197 ± 0.184 | 41.8 ± 34.9 | 3.2 ± 0.3 | 32.3 [26.7, 37.9] |
| as_stationary | 66.4 [62.6, 70.0] | 42.4 | 1.57 | 0.269 ± 0.169 | 29.3 ± 16.5 | 2.9 ± 0.2 | 62.6 [58.9, 66.3] |
| ppo_seed1 | 368.3 [348.4, 387.9] | 225.5 | 1.63 | 0.324 ± 0.142 | 96.9 ± 53.3 | 11.6 ± 2.8 | 302.7 [285.0, 320.3] |
| ppo_seed2 | 372.8 [352.7, 393.2] | 231.8 | 1.61 | 0.313 ± 0.141 | 101.8 ± 52.8 | 12.1 ± 2.8 | 302.5 [284.3, 320.8] |
| ppo_seed3 | 366.6 [346.1, 387.3] | 235.1 | 1.56 | 0.299 ± 0.138 | 106.7 ± 56.6 | 12.5 ± 3.0 | 291.5 [273.2, 309.9] |
| ppo_ablation_seed1 | 244.7 [228.0, 261.6] | 187.7 | 1.30 | 0.266 ± 0.174 | 92.9 ± 49.2 | 10.2 ± 2.2 | 197.3 [181.6, 213.1] |
| ppo_ablation_seed2 | 248.7 [230.6, 267.3] | 207.8 | 1.20 | 0.240 ± 0.169 | 109.6 ± 58.0 | 11.5 ± 2.8 | 187.9 [171.0, 205.2] |
| ppo_ablation_seed3 | 243.5 [224.9, 262.4] | 211.5 | 1.15 | 0.227 ± 0.167 | 119.2 ± 67.2 | 11.8 ± 2.9 | 179.8 [162.2, 197.4] |
| ppo_risk_seed1 | 209.4 [205.5, 213.3] | 45.1 | 4.65 | 0.715 ± 0.151 | 15.2 ± 7.7 | 2.3 ± 0.2 | 121.3 [117.8, 124.6] |
| ppo_risk_seed2 | 212.0 [207.6, 216.2] | 48.7 | 4.35 | 0.708 ± 0.156 | 15.2 ± 7.6 | 2.4 ± 0.2 | 125.3 [121.7, 128.9] |
| ppo_risk_seed3 | 209.1 [205.2, 213.0] | 44.9 | 4.66 | 0.716 ± 0.154 | 15.3 ± 7.3 | 2.5 ± 0.2 | 115.0 [111.7, 118.2] |
| ppo (pooled) | 369.2 [357.7, 381.1] | 230.7 | 1.60 | 0.312 ± 0.141 | 101.8 ± 54.4 | 12.1 ± 2.9 | 298.9 [288.7, 309.6] |
| ppo_ablation (pooled) | 245.6 [235.6, 256.0] | 202.5 | 1.21 | 0.244 ± 0.171 | 107.2 ± 59.5 | 11.2 ± 2.7 | 188.4 [179.0, 198.0] |
| ppo_risk (pooled) | 210.2 [207.9, 212.5] | 46.2 | 4.55 | 0.713 ± 0.154 | 15.2 ± 7.5 | 2.4 ± 0.2 | 120.5 [118.6, 122.5] |

| policy | fills | fill rate | bid / ask quoted | markout 0.1 s | markout 1 s | markout 5 s |
|---|---|---|---|---|---|---|
| symmetric | 13 | 0.038 | 1.00 / 1.00 | -0.383 | -0.158 | +0.115 |
| linear_skew | 156 | 0.375 | 1.00 / 1.00 | -0.389 | -0.451 | -0.503 |
| microprice_skew | 143 | 0.283 | 1.00 / 1.00 | -0.501 | -0.482 | -0.555 |
| avellaneda_stoikov | 231 | 0.426 | 1.00 / 1.00 | -0.562 | -0.516 | -0.558 |
| as_stationary | 240 | 0.446 | 1.00 / 1.00 | -0.533 | -0.494 | -0.535 |
| ppo_seed1 | 154 | 0.226 | 0.86 / 0.80 | -0.298 | -0.184 | -0.011 |
| ppo_seed2 | 156 | 0.205 | 0.95 / 0.83 | -0.282 | -0.195 | -0.017 |
| ppo_seed3 | 151 | 0.209 | 0.94 / 0.86 | -0.277 | -0.184 | +0.016 |
| ppo_ablation_seed1 | 137 | 0.185 | 0.90 / 0.82 | -0.289 | -0.270 | -0.175 |
| ppo_ablation_seed2 | 133 | 0.177 | 0.81 / 0.94 | -0.277 | -0.243 | -0.129 |
| ppo_ablation_seed3 | 145 | 0.228 | 0.79 / 0.96 | -0.287 | -0.292 | -0.192 |
| ppo_risk_seed1 | 148 | 0.271 | 0.78 / 0.94 | -0.375 | -0.334 | -0.333 |
| ppo_risk_seed2 | 157 | 0.313 | 0.76 / 0.80 | -0.385 | -0.342 | -0.349 |
| ppo_risk_seed3 | 162 | 0.278 | 0.85 / 0.76 | -0.400 | -0.350 | -0.343 |
| ppo (pooled) | 154 | 0.213 | 0.92 / 0.83 | -0.286 | -0.188 | -0.004 |
| ppo_ablation (pooled) | 138 | 0.197 | 0.84 / 0.91 | -0.284 | -0.269 | -0.166 |
| ppo_risk (pooled) | 156 | 0.287 | 0.79 / 0.83 | -0.387 | -0.342 | -0.342 |

| policy | spread capture | inventory PnL | rebates | liquidation | net PnL |
|---|---|---|---|---|---|
| symmetric | 59.8 [58.0, 61.7] | -5.2 [-18.5, 8.6] | 4.3 | -14.8 | 44.1 |
| linear_skew | 351.6 [347.5, 355.6] | -260.4 [-265.0, -255.9] | 51.3 | -2.8 | 139.7 |
| microprice_skew | 334.0 [330.1, 337.9] | -283.1 [-287.3, -278.8] | 50.1 | -2.3 | 98.7 |
| avellaneda_stoikov | 422.2 [419.0, 425.5] | -451.0 [-456.8, -445.4] | 80.1 | -13.7 | 37.6 |
| as_stationary | 434.7 [431.4, 438.1] | -448.7 [-453.1, -444.4] | 83.1 | -2.8 | 66.4 |
| ppo_seed1 | 314.0 [311.0, 317.1] | 15.2 [-4.5, 34.9] | 53.3 | -14.2 | 368.3 |
| ppo_seed2 | 316.9 [313.7, 320.1] | 18.4 [-1.6, 39.1] | 54.1 | -16.6 | 372.8 |
| ppo_seed3 | 303.6 [300.8, 306.5] | 27.5 [7.0, 48.5] | 51.8 | -16.3 | 366.6 |
| ppo_ablation_seed1 | 276.6 [274.0, 279.2] | -65.4 [-82.0, -48.9] | 44.0 | -10.5 | 244.7 |
| ppo_ablation_seed2 | 267.0 [264.7, 269.4] | -47.9 [-65.9, -29.5] | 42.6 | -13.0 | 248.7 |
| ppo_ablation_seed3 | 290.7 [288.0, 293.4] | -82.0 [-100.5, -63.4] | 46.6 | -11.9 | 243.5 |
| ppo_risk_seed1 | 326.4 [322.9, 329.8] | -164.5 [-167.4, -161.6] | 49.9 | -2.3 | 209.4 |
| ppo_risk_seed2 | 342.8 [339.1, 346.5] | -181.0 [-183.8, -178.2] | 52.5 | -2.4 | 212.0 |
| ppo_risk_seed3 | 342.9 [339.5, 346.3] | -186.9 [-189.7, -183.9] | 55.4 | -2.4 | 209.1 |
| ppo (pooled) | 311.5 [309.7, 313.4] | 20.4 [8.8, 32.5] | 53.0 | -15.7 | 369.2 |
| ppo_ablation (pooled) | 278.1 [276.5, 279.7] | -65.1 [-75.0, -54.9] | 44.4 | -11.8 | 245.6 |
| ppo_risk (pooled) | 337.3 [335.3, 339.4] | -177.4 [-179.2, -175.7] | 52.6 | -2.4 | 210.2 |

Paired PnL differences on the same markets:

| agent | baseline | mean difference | 95% CI |
|---|---|---|---|
| ppo_seed1 | symmetric | +324.1 | [+302.4, +346.3] |
| ppo_seed1 | linear_skew | +228.6 | [+208.4, +248.9] |
| ppo_seed1 | microprice_skew | +269.5 | [+248.8, +290.3] |
| ppo_seed1 | avellaneda_stoikov | +330.7 | [+309.6, +351.8] |
| ppo_seed1 | as_stationary | +301.9 | [+281.9, +322.2] |
| ppo_seed2 | symmetric | +328.6 | [+306.3, +351.5] |
| ppo_seed2 | linear_skew | +233.1 | [+212.6, +254.3] |
| ppo_seed2 | microprice_skew | +274.1 | [+252.8, +295.7] |
| ppo_seed2 | avellaneda_stoikov | +335.2 | [+313.6, +357.3] |
| ppo_seed2 | as_stationary | +306.4 | [+285.8, +327.2] |
| ppo_seed3 | symmetric | +322.4 | [+300.1, +346.0] |
| ppo_seed3 | linear_skew | +226.9 | [+205.6, +248.2] |
| ppo_seed3 | microprice_skew | +267.8 | [+246.2, +289.9] |
| ppo_seed3 | avellaneda_stoikov | +329.0 | [+307.2, +351.2] |
| ppo_seed3 | as_stationary | +300.2 | [+279.2, +321.7] |
| ppo_ablation_seed1 | symmetric | +200.5 | [+182.2, +219.3] |
| ppo_ablation_seed1 | linear_skew | +105.0 | [+88.0, +122.0] |
| ppo_ablation_seed1 | microprice_skew | +145.9 | [+128.3, +163.7] |
| ppo_ablation_seed1 | avellaneda_stoikov | +207.1 | [+189.4, +225.1] |
| ppo_ablation_seed1 | as_stationary | +178.3 | [+161.5, +195.3] |
| ppo_ablation_seed2 | symmetric | +204.6 | [+185.0, +224.7] |
| ppo_ablation_seed2 | linear_skew | +109.1 | [+91.0, +127.6] |
| ppo_ablation_seed2 | microprice_skew | +150.0 | [+131.1, +169.4] |
| ppo_ablation_seed2 | avellaneda_stoikov | +211.1 | [+192.5, +230.4] |
| ppo_ablation_seed2 | as_stationary | +182.4 | [+164.2, +200.9] |
| ppo_ablation_seed3 | symmetric | +199.4 | [+180.3, +218.9] |
| ppo_ablation_seed3 | linear_skew | +103.8 | [+85.2, +122.6] |
| ppo_ablation_seed3 | microprice_skew | +144.8 | [+125.3, +164.3] |
| ppo_ablation_seed3 | avellaneda_stoikov | +205.9 | [+186.3, +225.5] |
| ppo_ablation_seed3 | as_stationary | +177.1 | [+158.2, +196.2] |
| ppo_risk_seed1 | symmetric | +165.3 | [+151.5, +178.7] |
| ppo_risk_seed1 | linear_skew | +69.8 | [+65.2, +74.4] |
| ppo_risk_seed1 | microprice_skew | +110.7 | [+106.3, +115.1] |
| ppo_risk_seed1 | avellaneda_stoikov | +171.8 | [+165.6, +178.3] |
| ppo_risk_seed1 | as_stationary | +143.1 | [+138.5, +147.8] |
| ppo_risk_seed2 | symmetric | +167.8 | [+154.1, +181.2] |
| ppo_risk_seed2 | linear_skew | +72.3 | [+67.6, +77.1] |
| ppo_risk_seed2 | microprice_skew | +113.2 | [+108.5, +118.0] |
| ppo_risk_seed2 | avellaneda_stoikov | +174.3 | [+167.9, +181.1] |
| ppo_risk_seed2 | as_stationary | +145.6 | [+140.8, +150.4] |
| ppo_risk_seed3 | symmetric | +164.9 | [+151.0, +178.4] |
| ppo_risk_seed3 | linear_skew | +69.4 | [+64.6, +74.3] |
| ppo_risk_seed3 | microprice_skew | +110.3 | [+105.5, +115.2] |
| ppo_risk_seed3 | avellaneda_stoikov | +171.5 | [+165.2, +178.1] |
| ppo_risk_seed3 | as_stationary | +142.7 | [+138.0, +147.6] |
| ppo, mean of 3 seeds | symmetric | +325.1 | [+303.5, +347.4] |
| ppo, mean of 3 seeds | linear_skew | +229.6 | [+209.6, +250.0] |
| ppo, mean of 3 seeds | microprice_skew | +270.5 | [+249.7, +291.4] |
| ppo, mean of 3 seeds | avellaneda_stoikov | +331.6 | [+310.8, +353.1] |
| ppo, mean of 3 seeds | as_stationary | +302.8 | [+282.9, +323.1] |
| ppo_ablation, mean of 3 seeds | symmetric | +201.5 | [+183.5, +220.1] |
| ppo_ablation, mean of 3 seeds | linear_skew | +106.0 | [+89.2, +122.9] |
| ppo_ablation, mean of 3 seeds | microprice_skew | +146.9 | [+129.3, +164.7] |
| ppo_ablation, mean of 3 seeds | avellaneda_stoikov | +208.0 | [+190.4, +226.2] |
| ppo_ablation, mean of 3 seeds | as_stationary | +179.3 | [+162.3, +196.5] |
| ppo_risk, mean of 3 seeds | symmetric | +166.0 | [+152.3, +179.3] |
| ppo_risk, mean of 3 seeds | linear_skew | +70.5 | [+66.1, +74.9] |
| ppo_risk, mean of 3 seeds | microprice_skew | +111.4 | [+107.1, +115.8] |
| ppo_risk, mean of 3 seeds | avellaneda_stoikov | +172.6 | [+166.4, +179.0] |
| ppo_risk, mean of 3 seeds | as_stationary | +143.8 | [+139.3, +148.3] |

Spread across training seeds:

- ppo: per-seed mean PnL 368.3, 372.8, 366.6; between-seed sd 3.2
- ppo_ablation: per-seed mean PnL 244.7, 248.7, 243.5; between-seed sd 2.7
- ppo_risk: per-seed mean PnL 209.4, 212.0, 209.1; between-seed sd 1.6
