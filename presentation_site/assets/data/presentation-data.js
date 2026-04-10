window.presentationData = {
  "headline": {
    "windowLabel": "January 1-10, 2026",
    "windowHours": 223,
    "historyHours": 336,
    "forecastHorizonHours": 24,
    "datasetRows": 8983,
    "peakThresholdKw": 970.2,
    "bestComposite": 32.43,
    "baselineComposite": 63.54,
    "anchorComposite": 38.27,
    "previousWinnerComposite": 47.03,
    "improvementVsBaselinePct": 49.0,
    "improvementVsAnchorPct": 15.3,
    "tailHours": 15
  },
  "heroMetrics": [
    {
      "label": "Public Composite",
      "value": "32.43",
      "detail": "Measured on the released public target window"
    },
    {
      "label": "Gain vs Anchor",
      "value": "15.3%",
      "detail": "Improvement over the already-strong anchor forecast"
    },
    {
      "label": "Gain vs Old Winner",
      "value": "31.0%",
      "detail": "Improvement over the previous best public result"
    },
    {
      "label": "Tail-Routed Hours",
      "value": "15",
      "detail": "Only the highest anchor hours are handed to the peak specialist"
    }
  ],
  "challengeFacts": [
    "Target window: January 1, 2026 to January 10, 2026 (223 hours).",
    "Scoring: 0.5 * MAE_all + 0.3 * MAE_peak + 0.2 * pinball_p90.",
    "Peak-hour cutoff on the public window: 970.2 kW.",
    "The routed tail hours are concentrated at the end of the window, from Jan 08 21:00 to Jan 10 06:00 UTC.",
    "Best day in the public window: Jan 08 with 19.5 kW MAE.",
    "Hardest day in the public window: Jan 09 with 49.0 kW MAE."
  ],
  "blendChips": [
    "Anchor forecast = outputs/results/predictions.csv",
    "Tail specialist = predictions_tabular_emergency.csv",
    "Top 15 highest anchor hours are rerouted",
    "Tail point blend = 27.5% anchor + 72.5% tail",
    "Tail p90 blend = 95% anchor + 5% tail",
    "Final score = 32.43 on the public window"
  ],
  "experimentRows": [
    {
      "label": "Organizer baseline",
      "composite": 63.54,
      "maeAll": 55.73,
      "maePeak": 110.61,
      "pinball": 12.46,
      "highlight": false
    },
    {
      "label": "Tail specialist",
      "composite": 55.06,
      "maeAll": 72.27,
      "maePeak": 50.86,
      "pinball": 18.34,
      "highlight": false
    },
    {
      "label": "Previous winner",
      "composite": 47.03,
      "maeAll": 54.6,
      "maePeak": 59.1,
      "pinball": 10.0,
      "highlight": false
    },
    {
      "label": "Strong anchor forecast",
      "composite": 38.27,
      "maeAll": 34.06,
      "maePeak": 65.9,
      "pinball": 7.35,
      "highlight": false
    },
    {
      "label": "Winning anchor-tail hybrid",
      "composite": 32.43,
      "maeAll": 32.75,
      "maePeak": 48.68,
      "pinball": 7.28,
      "highlight": true
    }
  ],
  "featureNotes": [
    "The tail specialist is not the best overall model, but it is much better on the hardest high-load hours.",
    "Strong anchor forecast: 38.27 composite, 65.90 peak MAE.",
    "Tail specialist: 55.06 composite, 50.86 peak MAE.",
    "The hybrid keeps the anchor on 208 hours and only hands 15 hours to the peak specialist.",
    "The tabular tail specialist still uses the strict T-24 setup with 336h history and 24h horizon."
  ],
  "outlookMetrics": [
    {
      "label": "Realized Temp Correlation",
      "value": "0.557",
      "detail": "VC Halle 3 temperature vs load on the full weather-overlap window"
    },
    {
      "label": "Noisy Forecast Correlation",
      "value": "0.520",
      "detail": "Synthetic 24h forecast proxy with 2.0\u00b0C noise"
    },
    {
      "label": "Signal Retained",
      "value": "93.4%",
      "detail": "Share of the realized weather signal that survives the forecast-noise stress test"
    }
  ],
  "outlookPoints": [
    "Temperature is already the strongest same-hour weather signal in the existing analysis, led by VC Halle 3.",
    "Because archived forecast files are not part of the participant package, we build a proxy forecast by taking realized temperature and adding Gaussian noise.",
    "We then treat that noisy value as if it were a 24-hour-ahead forecast available at prediction time and measure how much correlation to load survives.",
    "Across 2121 weather-overlap hours from 2025-09-24 to 2026-01-10, the proxy still keeps a strong signal.",
    "This is a sensitivity analysis, not a true historical forecast backtest, so it should be read as evidence of likely usefulness rather than proof of final leaderboard gain."
  ],
  "outlookSteps": [
    "Ingest 24h site weather forecasts for temperature, wind, and spread as first-class T-24 features.",
    "Feed forecast temperatures into both the anchor model and the tail specialist, especially for the late-window peaks that trigger rerouting.",
    "Add forecast-minus-recent-history features so the model can react to tomorrow being warmer or colder than the recent reefer regime.",
    "Re-run the anchor-tail routing experiment with weather-aware features and compare whether fewer hours need specialist handoff."
  ],
  "takeaways": [
    "The new winner is a routed hybrid: a strong anchor forecast for almost every hour, plus a peak specialist only where the anchor itself predicts the heaviest load.",
    "This is why the score jumps so sharply: the anchor protects the broad shape of the curve, while the tail specialist only touches the handful of hours where peak sensitivity matters most.",
    "Container count remains the clearest operational driver, and temperature history still explains why the hardest hours cluster late in the January window.",
    "The most natural next feature family is a 24h weather forecast, because even a noisy proxy forecast still preserves most of the temperature-to-load relationship.",
    "The site is generated from repo artifacts, so future improvements in `hackathon_reefer_dl` can be pushed into the presentation by rerunning one script."
  ],
  "generatedAt": "2026-04-10 09:13 UTC"
};
