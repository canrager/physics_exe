# Weather Impact Analysis

## Scope
- Same-hour weather versus same-hour reefer power shows actual observed weather impact.
- History-window analysis measures how much past observed weather at forecast time is informative for the next 24 hourly power values.
- This is still an upper-bound analysis because a production day-ahead model would use weather forecasts, not the future realized weather.

## Overlap
- Label range: 2025-01-01T00:00:00Z to 2026-01-10T06:00:00Z
- Weather range: 2025-09-24T10:00:00Z to 2026-02-23T14:00:00Z
- Usable overlap: 2025-09-24T10:00:00Z to 2026-01-10T06:00:00Z

## Top Same-Hour Weather Correlations
- `weather_temperature_vc_halle3_c`: 0.557
- `weather_temperature_mean_c`: 0.377
- `weather_temperature_zentralgate_c`: 0.231
- `weather_wind_direction_vc_halle3_cos`: -0.210
- `weather_wind_direction_mean_cos`: -0.193
- `weather_wind_direction_zentralgate_cos`: -0.172
- `weather_wind_direction_zentralgate_consistency`: -0.098
- `weather_wind_speed_vc_halle3`: -0.081
- `weather_wind_direction_vc_halle3_consistency`: 0.069
- `weather_wind_direction_mean_consistency`: -0.039

## Recommended Weather History Windows
- `weather_temperature_vc_halle3_c`: efficient window 51 h, peak window 72 h, peak score 0.623
- `weather_temperature_mean_c`: efficient window 44 h, peak window 72 h, peak score 0.620
- `weather_temperature_zentralgate_c`: efficient window 46 h, peak window 67 h, peak score 0.574
- `weather_wind_direction_vc_halle3_cos`: efficient window 60 h, peak window 72 h, peak score 0.293
- `weather_wind_direction_mean_cos`: efficient window 62 h, peak window 72 h, peak score 0.290
- `weather_wind_direction_zentralgate_cos`: efficient window 63 h, peak window 72 h, peak score 0.285
- `weather_wind_direction_zentralgate_consistency`: efficient window 26 h, peak window 40 h, peak score 0.261
- `weather_wind_speed_vc_halle3`: efficient window 57 h, peak window 71 h, peak score 0.223
- `weather_wind_speed_spread`: efficient window 43 h, peak window 50 h, peak score 0.169
- `weather_temperature_spread_c`: efficient window 43 h, peak window 51 h, peak score 0.164
