# Reefer Demand Forecasting Handover

This document summarizes the key findings, modeling assumptions, feature relevance, and data handling decisions discovered during analysis of the reefer demand challenge package.

## Problem framing

- The challenge is to forecast hourly combined reefer electricity demand.
- The modeling target should be treated at the hourly aggregate level.
- The most practical hourly target proxy from the reefer dataset is the sum of `AvPowerCons` per hour.
- The submission requires:
  - `pred_power_kw`
  - `pred_p90_kw`

## Core modeling perspective

- The strongest predictive signals are not primarily weather features.
- The most important drivers are operational structure features:
  - how many reefer units are active
  - what hardware mix is present
  - what time pattern the hour belongs to
  - what happened in previous hours and previous days

## Most relevant features

### 1. Active reefer volume

These are the strongest structural drivers of demand:

- `active_rows`
- `unique_containers`
- `unique_visits`

Interpretation:
- More active reefer containers in a given hour strongly increases total demand.
- These features were consistently more informative than many raw environmental variables.

### 2. Lagged target features

Highly recommended:

- demand at `t-24`
- demand at `t-48`
- demand at `t-168`

Interpretation:
- Reefer demand clearly shows daily and weekly repetition.
- Lagged target features should be treated as top-priority model inputs.

### 3. Time features

Important recurring structure:

- `hour_of_day`
- `day_of_week`
- `is_weekend`
- `month` or `season`

Findings:
- Demand is generally higher around midday and early afternoon than during the night.
- Weekend hours tend to be higher than many weekday hours.
- There is a clear seasonal pattern across the year.

### 4. Hardware mix per hour

Hourly counts of hardware types are relevant.

Particularly informative hardware families:

- `SCC6`
- `DecosVb`
- `ML5`
- `DecosIIIh`
- `DecosIIIj`
- `DecosVa`
- `MP4000`
- `ML3`

Interpretation:
- Peak hours and high-demand periods often coincide with a different hardware composition.
- `SCC6` is especially overrepresented in peak hours.

### 5. Temperature

Temperature is useful, but not the single dominant signal.

Useful temperature-related features:

- external weather temperature when available
- internal reefer ambient temperature
- internal setpoint / return / supply temperatures

Interpretation:
- Temperature contributes signal, but container count and operational structure are usually stronger.

### 6. Stack tier

- `avg_stack_tier` showed surprisingly usable signal at hourly aggregate level.
- It should be treated as a secondary but potentially useful feature.

## Features to use carefully

### 1. Same-hour energy values

Use carefully or avoid as direct predictors for the same target hour:

- `sum_energy_wh`
- `avg_energy_wh`

Reason:
- These are too close to the target and may introduce leakage if used for the hour being predicted.

Correct usage:
- acceptable as lagged features from earlier hours
- acceptable as evaluation/training target references
- not acceptable as direct same-hour inputs for forecasting the same hour

### 2. Raw IDs

Do not feed raw IDs directly into the model:

- `container_visit_uuid`
- `container_uuid`
- `customer_uuid`

Reason:
- They are high-cardinality identifiers.
- They are more useful after aggregation, for example:
  - number of active visits
  - number of active containers
  - number of distinct customers per hour

## Features that appear relatively weak or low priority

### Container size

Container size does not appear to be a strong standalone modeling feature.

Distribution:
- `40ft`: about `93.81%`
- `20ft`: about `6.10%`
- `45ft`: about `0.04%`

Findings:
- `40ft` dominates the dataset so strongly that raw `ContainerSize` carries limited independent information.
- The count of `40ft` containers correlates with demand mainly because it behaves like a proxy for total active container count.
- Peak and non-peak hours have almost identical size composition.

Conclusion:
- `ContainerSize` is not a high-value raw feature.
- If used at all, it should only be treated as a minor supporting aggregate.

## Temperature handling rules

These decisions were explicitly agreed as project rules:

- Use temperature values when available.
- Fill small local gaps.
- For larger gaps, include a missingness flag.
- If possible, use a second temperature source.
- Do not fully invent long missing blocks.

### Temperature gap observations

External weather temperature series:
- contain meaningful gaps
- include some very large missing blocks

Internal reefer temperature columns:
- `TemperatureAmbient`
- `TemperatureReturn`
- `RemperatureSupply`
- `TemperatureSetPoint`

These internal reefer temperature fields showed no missing values in the reefer dataset during inspection.

### Small-gap imputation rule

For short temperature gaps, the agreed strategy is to fill within the same daypart:

- night: `00-05`
- morning: `06-11`
- afternoon: `12-17`
- evening: `18-23`

Within a daypart:
- forward fill first
- backward fill if needed

This rule should only be used for small gaps, not for large missing blocks.

## Seasonal findings

### Summer vs autumn

Autumn demand is higher than summer demand.

Interpretation:
- The main reason is not that each individual container draws much more power.
- The stronger driver is that more reefer units are active at the same time in autumn.
- The hardware mix also shifts, especially toward more `SCC6`.

Key insight:
- Autumn is operationally heavier, not just thermally different.

### Summer vs winter

- Summer demand is higher than winter demand.
- Autumn is the strongest season in the observed dataset.

## Special anomalous window

The period from `2025-03-30` to `2025-04-24` stands out strongly from the rest of the dataset.

Observed properties:
- substantially lower hourly demand than the rest of the year
- lower number of active containers
- lower average power per active container
- much lower variability
- almost no peak hours

Why this matters:
- This window often looks visually strange in plots because it appears flatter and more compressed than the rest.
- It likely reflects a different operating regime or a special period, not just ordinary seasonality.

Project interpretation:
- Treat this period as a special segment worth separate consideration.
- Be cautious when using global plots or statistics that may be distorted by this regime difference.

## Peak-hour patterns

Peak hours were defined as the top `10%` of hourly demand.

### Common peak-hour patterns

- Peaks occur much more often in autumn than in the other seasons.
- Peaks are concentrated around midday and afternoon.
- Peaks occur more often on Friday, Saturday, and Sunday, especially Saturday and Sunday.
- Peak hours have:
  - more active containers
  - higher mean power per container
  - somewhat warmer ambient conditions
  - colder internal reefer temperature states

### Hardware patterns in peak hours

Hardware types overrepresented in peak hours include:

- `SCC6`
- `ML3`
- `DecosVb`
- `ML5`
- `DecosIIIj`
- `DecosVa`
- `MP4000`
- `DecosIIIh`

Main interpretation:
- Peak hours are driven by both higher volume and a shifted hardware composition.
- `SCC6` is the clearest recurring signal.

## Practical feature priority for modeling

Suggested priority order:

1. active reefer count features
2. lagged target features
3. hour-of-day features
4. weekday / weekend features
5. hardware mix features
6. temperature features
7. stack-tier aggregates
8. month / season indicators
9. internal reefer temperature aggregates
10. distinct customer count

## Reproducibility and submission requirements

According to the challenge files, the final hand-in should include:

1. `predictions.csv`
2. `approach.md`
3. code or notebook

The code or notebook must:

- be easy to run end-to-end
- be reproducible by organizers
- work on the hidden full timestamp list and complete reefer release data

## Recommended modeling caution

- Separate truly predictive features from descriptive or leakage-prone variables.
- Do not use same-hour target-like energy values as direct forecast inputs.
- Prefer operational aggregates over raw identifiers.
- Treat weather as useful but incomplete.
- Account for anomalous low-demand windows explicitly during analysis and evaluation.
