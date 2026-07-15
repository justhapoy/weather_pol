"""
ENTRY-BAND GATE.

The ledger analysis is unambiguous about WHERE each strategy makes and loses
money by ENTRY PRICE:

  late_observed_no  — the engine, but ONLY in the 0.50–0.85 band (69–79% WR,
                      +$271). The <0.50 band is a net loser (-$96, 0–45% WR).
  late_observed_yes — a lottery that only pays when bought CHEAP (<~$0.10
                      band +$109). Above ~$0.20 it bleeds.

Neither of these bands can be gated from Telegram (LATE_OBSERVED_NO_MIN_PRICE
caps at 0.20), so this overlay adds a hard entry-price gate at the single
placement path. It is consulted alongside strategy_gate.trade_allowed():
a blocked leg is dropped BEFORE any order is built.

All thresholds are live-tunable (registered in settings). Master switch
ENTRY_BAND_GATE_ENABLED (default ON). Fail-open on any error.
"""

try:
    from config import Config
except Exception:  # pragma: no cover
    Config = None

SETTING_DEFAULTS = {
    "ENTRY_BAND_GATE_ENABLED": True,
    "LATE_OBS_NO_MIN_ENTRY": 0.50,   # skip late_observed_no below this price
    "LATE_OBS_NO_MAX_ENTRY": 0.97,   # skip the near-1.0 no-edge tail
    "LATE_OBS_YES_MAX_ENTRY": 0.15,  # skip late_observed_yes above this price
}


def ensure_defaults():
    if Config is None:
        return
    for key, default in SETTING_DEFAULTS.items():
        if not hasattr(Config, key):
            setattr(Config, key, default)


def _f(name, default):
    if Config is None:
        return default
    try:
        return float(getattr(Config, name, default))
    except (TypeError, ValueError):
        return default


def entry_allowed(strategy, entry_price):
    """Return (ok, reason). ok=False => price outside the profitable band."""
    if Config is None or not bool(getattr(Config, "ENTRY_BAND_GATE_ENABLED", True)):
        return True, "ok"
    try:
        p = float(entry_price)
    except (TypeError, ValueError):
        return True, "ok"
    tag = (strategy or "").strip().lower()
    if tag == "late_observed_no":
        lo = _f("LATE_OBS_NO_MIN_ENTRY", 0.50)
        hi = _f("LATE_OBS_NO_MAX_ENTRY", 0.97)
        if p < lo:
            return False, f"late_observed_no @ {p:.2f} < {lo:.2f} (junk band)"
        if p > hi:
            return False, f"late_observed_no @ {p:.2f} > {hi:.2f} (no-edge tail)"
        return True, "ok"
    if tag == "late_observed_yes":
        hi = _f("LATE_OBS_YES_MAX_ENTRY", 0.15)
        if p > hi:
            return False, f"late_observed_yes @ {p:.2f} > {hi:.2f} (lottery only pays cheap)"
        return True, "ok"
    return True, "ok"


ensure_defaults()
