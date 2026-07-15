"""
ML Decision Engine - LLM (Freemodel/OpenAI-compatible) for fast trading decisions.

The ML is used for:
1. Signal validation (confirm/reject entry signals)
2. Position review (hold/sell open positions)
3. Mid-trade PROFIT decisions (book now vs run for more; 300%+ cap handling)
4. Market selection (which cities to prioritize today)
5. Narrative trade reports for /mlanalysis

Design:
- MARKET-SCOPED CONTEXT, FAST single API call.
- ALWAYS degrades gracefully to the local model / rules when no API key or the
  API fails, so the bot never blocks on the network.

Req-32: validate_signal no longer fabricates a 0.0C forecast. The trade path
did not pass a forecast value, so the old forecast_temp=0.0 default made the
reasoning model 'see' a freezing forecast that contradicted every warm market
and veto essentially everything. forecast_temp now defaults to None ('unknown'),
the prompt says so explicitly, and a SKIP rationalised on the (absent) forecast
is neutralised so a missing forecast can never silently kill every signal again.

Req-31: the Freemodel gpt-5.x models are REASONING models - they emit a
<think>...</think> block (~7s) BEFORE the JSON answer. The old engine used an
8s timeout, a 40-60 token cap and a fences-only parser, so every reasoning
reply truncated / timed out and silently fell back to the local model even with
a valid key. This version reads timeout + token budgets from config, strips the
<think> preamble, and tolerantly extracts the JSON answer.
"""

import re
import time
import json
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from config import Config
from logger import log


class MLDecisionEngine:
    """Fast ML-powered trading decisions using an OpenAI-compatible LLM."""

    def __init__(self):
        self.base_url = (Config.ML_API_URL or '').rstrip('/')
        self.api_key = Config.ML_API_KEY
        self.model = Config.ML_MODEL
        self.analysis_model = getattr(Config, 'ML_ANALYSIS_MODEL', self.model)
        # Req-31: reasoning-model aware budgets, read from config (no more 8s/60tok
        # hardcodes that silently truncated every reply).
        self.timeout = float(getattr(Config, 'ML_QUERY_TIMEOUT', 30))
        self.decision_max_tokens = int(getattr(Config, 'ML_DECISION_MAX_TOKENS', 700))
        self.analysis_max_tokens = int(getattr(Config, 'ML_ANALYSIS_MAX_TOKENS', 1200))
        self.analysis_enabled = bool(getattr(Config, 'ML_ANALYSIS_ENABLED', True))
        self.enabled = bool(self.api_key)
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        })
        self._cache: Dict[str, Tuple[float, Dict]] = {}
        self._cache_ttl = 120  # 2 minutes
        self._total_tokens_used = 0
        self._total_calls = 0
        self._last_ok_ts = 0.0
        self._last_error = ''

        # Local model fallback (no API needed)
        self._local_model = None
        self._api_failures = 0
        self._max_api_failures = 5

        if self.enabled:
            log.info(f"  ML Engine: {self.model} (analysis {self.analysis_model}) "
                     f"via {self.base_url[:34]}...")
        else:
            log.info("  ML Engine: API key not set - using LOCAL model only")

    # ------------------------------------------------------------------ #
    # Local model (lazy)
    # ------------------------------------------------------------------ #
    @property
    def local_model(self):
        if self._local_model is None:
            try:
                from ml.local_model import get_local_model
                self._local_model = get_local_model()
            except Exception as e:
                log.warning(f"  Local model init failed: {e}")
        return self._local_model

    def _city_wr(self, city: str) -> Optional[float]:
        try:
            from ml.local_model import CITY_WIN_RATES
            key = (city or '').strip().lower().replace(' ', '-')
            return CITY_WIN_RATES.get(key)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # 1) Entry validation
    # ------------------------------------------------------------------ #
    def validate_signal(self, city: str, bucket_label: str, entry_price: float,
                        our_prob: float, edge: float, forecast_temp=None,
                        n_models: int = 0, weekly_context: str = '') -> Dict:
        """Validate a trading signal. Returns {action:'BUY'|'SKIP', confidence, reason}.

        forecast_temp may be None when the caller has no live forecast value
        handy (observation- or price-driven strategies). Req-32: in that case we
        MUST NOT present a fabricated 0.0C to the model - that made the reasoning
        model veto essentially every warm/cold market. When the forecast is
        unknown we say so and tell the model to judge on price / edge / observed
        lock only, and we neutralise any SKIP it rationalises on the absent
        forecast.
        """
        has_forecast = forecast_temp is not None
        if has_forecast:
            try:
                forecast_temp = float(forecast_temp)
            except (ValueError, TypeError):
                forecast_temp = None
                has_forecast = False

        cwr = self._city_wr(city)
        if not self.enabled:
            if self.local_model is not None:
                return self.local_model.predict_entry({
                    "edge": edge, "edge_ratio": our_prob / max(entry_price, 0.01),
                    "n_models": n_models or 3, "confidence": 0.6,
                    "market_spread_bps": 500, "market_price": entry_price,
                    "forecast_std": 1.5, "lead_hours": 24,
                    "city_win_rate": cwr if cwr is not None else 0.07,
                    "bucket_position": entry_price,
                    "hour_of_day": datetime.now(timezone.utc).hour,
                    "day_of_week": datetime.now(timezone.utc).weekday(),
                })
            return {'action': 'BUY', 'confidence': 0.7, 'reason': 'ML disabled'}

        cache_key = f"sig_{city}_{bucket_label}_{entry_price:.3f}"
        now = time.time()
        if cache_key in self._cache:
            ts, result = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return result

        cwr_txt = f" CityWR:{cwr:.0%}" if cwr is not None else ""
        if has_forecast:
            fc_txt = f"Forecast:{forecast_temp:.1f}C Models:{n_models}"
            guide = ''
        else:
            # Req-32: NEVER imply 0C. The bucket label already carries the
            # temperature range; judge on price/edge/lock instead.
            fc_txt = f"Forecast:n/a Models:{n_models}"
            guide = ("No live forecast value is provided; the bucket label carries the "
                     "temperature range. Judge ONLY on price, edge, our probability and "
                     "city history. Do NOT skip merely because the forecast is missing.\n")
        prompt = (
            'Weather trade entry. Reply JSON only: '
            '{"action":"BUY"|"SKIP","conf":0-1,"why":"short"}\n'
            + guide +
            f"City:{city} Bucket:{bucket_label} Price:${entry_price:.3f} "
            f"OurProb:{our_prob:.0%} Edge:{edge:+.0%} "
            f"{fc_txt}{cwr_txt}\n"
            f"History:{weekly_context[:80]}"
        )
        result = self._query(prompt)
        # Req-32 safety net: when no forecast was available, a SKIP that is
        # clearly rationalised on the (absent) forecast is NOT a valid veto -
        # neutralise it so a missing forecast can never silently kill every
        # signal again. Price-based SKIPs (reason not citing the forecast) pass.
        if not has_forecast and str(result.get('action', '')).upper() == 'SKIP':
            why = str(result.get('reason', '')).lower()
            if ('forecast' in why or '0.0' in why or '0c' in why
                    or 'no data' in why or 'contradict' in why):
                result = {'action': 'BUY', 'confidence': 0.55,
                          'reason': 'forecast n/a -> defer to price/edge'}
        self._cache[cache_key] = (now, result)
        return result

    # ------------------------------------------------------------------ #
    # 2) Position review (generic hold/sell)
    # ------------------------------------------------------------------ #
    def review_position(self, city: str, bucket_label: str, entry_price: float,
                        current_price: float, hold_hours: float,
                        resolution_hours: float, *, strategy: str = '',
                        roi_pct: Optional[float] = None,
                        peak_roi: Optional[float] = None,
                        edge: Optional[float] = None) -> Dict:
        """Ask ML whether to HOLD or SELL an open position.
        Returns {action:'HOLD'|'SELL', confidence, reason}."""
        if roi_pct is None:
            roi_pct = ((current_price - entry_price) / max(entry_price, 1e-9)) * 100
        if not self.enabled:
            if self.local_model is not None:
                return self.local_model.predict_exit({
                    'pnl_pct': roi_pct, 'hours_remaining': resolution_hours,
                    'forecast_changed': False,
                })
            return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'ML disabled'}

        cwr = self._city_wr(city)
        ctx = []
        if strategy:
            ctx.append(f"Strat:{strategy}")
        if peak_roi is not None:
            ctx.append(f"Peak:{peak_roi:+.0f}%")
        if edge is not None:
            ctx.append(f"Edge:{edge:+.0%}")
        if cwr is not None:
            ctx.append(f"CityWR:{cwr:.0%}")
        prompt = (
            'Position review. Reply JSON: '
            '{"action":"HOLD"|"SELL","conf":0-1,"why":"short"}\n'
            f"City:{city} {bucket_label} Entry:${entry_price:.3f} "
            f"Now:${current_price:.3f} ROI:{roi_pct:+.0f}% "
            f"Held:{hold_hours:.0f}h Left:{resolution_hours:.0f}h " + ' '.join(ctx)
        )
        return self._query(prompt)

    # ------------------------------------------------------------------ #
    # 3) Mid-trade PROFIT decision (book vs run; 300% cap)
    # ------------------------------------------------------------------ #
    def decide_profit_hold(self, *, city: str, bucket_label: str, strategy: str,
                           entry_price: float, current_price: float,
                           roi_pct: float, peak_roi: Optional[float] = None,
                           hold_hours: float = 0.0,
                           resolution_hours: float = 24.0,
                           edge: Optional[float] = None,
                           mode: str = 'ladder') -> Dict:
        """Decide whether to BOOK now or HOLD for more upside on an in-profit position.

        mode='ladder' -> hit the +profit target; book now or run to 20/40/50/200%.
                         If ML unavailable -> BOOK (lock the target, never round-trip).
        mode='cap'    -> position is above the 300% cap; HOLD to settle for even more
                         or BOOK now. If ML unavailable -> HOLD (let it settle), per spec.

        Returns {action:'BOOK'|'HOLD', target_roi: float, confidence, reason, source}.
        """
        if peak_roi is None:
            peak_roi = roi_pct
        fallback_hold = (mode == 'cap')

        if not self.enabled:
            lm = self.local_model
            if lm is not None:
                return lm.decide_hold({
                    'roi_pct': roi_pct, 'peak_roi': peak_roi,
                    'hours_remaining': resolution_hours, 'strategy': strategy,
                    'edge': edge or 0.0, 'mode': mode,
                })
            return {'action': 'HOLD' if fallback_hold else 'BOOK',
                    'target_roi': roi_pct, 'confidence': 0.5,
                    'reason': 'no ML; ' + ('settle' if fallback_hold else 'book target'),
                    'source': 'fallback'}

        cwr = self._city_wr(city)
        ctx = [f"Peak:{peak_roi:+.0f}%"]
        if edge is not None:
            ctx.append(f"Edge:{edge:+.0%}")
        if cwr is not None:
            ctx.append(f"CityWR:{cwr:.0%}")
        if mode == 'cap':
            guide = ("Position is ABOVE the 300% cap. It already round-tripped to 0 once "
                     "in the past. Decide HOLD only if strong reason it climbs further or "
                     "settles a winner; else BOOK now to lock the gain.")
        else:
            guide = ("Position hit its profit target. Decide BOOK now to lock +profit, or "
                     "HOLD for more (realistic next targets 20/40/50/200%). Prefer BOOK if "
                     "it is fading from peak or little time/edge remains.")
        prompt = (
            'Profit decision. Reply JSON only: '
            '{"action":"BOOK"|"HOLD","target":<pct number>,"conf":0-1,"why":"short"}\n'
            f"{guide}\n"
            f"City:{city} {bucket_label} Strat:{strategy} Entry:${entry_price:.3f} "
            f"Now:${current_price:.3f} ROI:{roi_pct:+.0f}% Held:{hold_hours:.0f}h "
            f"Left:{resolution_hours:.0f}h " + ' '.join(ctx)
        )
        res = self._query(prompt)
        act = str(res.get('action', '')).upper()
        if act not in ('BOOK', 'HOLD'):
            act = 'HOLD' if fallback_hold else 'BOOK'
        res['action'] = act
        res.setdefault('target_roi', res.get('target', roi_pct))
        try:
            res['target_roi'] = float(res['target_roi'])
        except (ValueError, TypeError):
            res['target_roi'] = roi_pct
        return res

    # ------------------------------------------------------------------ #
    # 4) Market selection
    # ------------------------------------------------------------------ #
    def select_markets(self, available_cities: List[str],
                       weekly_context: str = '') -> List[str]:
        if not self.enabled:
            return available_cities[:8]
        prompt = (
            'Rank cities for weather trading today. Reply JSON array of top 5: '
            '["city1","city2",...]\n'
            f"Available: {','.join(available_cities[:15])}\n"
            f"Performance: {weekly_context[:100]}"
        )
        result = self._query(prompt)
        if isinstance(result.get('raw'), list):
            return [str(c) for c in result['raw'] if isinstance(c, (str, int, float))]
        return available_cities[:8]

    # ------------------------------------------------------------------ #
    # 5) Narrative trade report for /mlanalysis
    # ------------------------------------------------------------------ #
    def write_trade_report(self, stats: Dict, by_strat: Dict,
                           by_city: Optional[Dict] = None,
                           recent: Optional[List[Dict]] = None) -> str:
        """Produce a narrative report of how trading is going. Uses the LLM when
        available AND /mlanalysis is enabled, otherwise a heuristic summary.
        Always returns a string."""
        heur = self._report_heuristic(stats, by_strat, by_city)
        if not self.enabled or not self.analysis_enabled:
            return heur

        s = stats or {}
        lines = [
            f"Bal:${s.get('balance', 0):.0f} PV:${s.get('portfolio_value', 0):.0f} "
            f"PnL:${s.get('total_pnl', 0):+.0f} ROI:{s.get('roi_pct', 0):+.1f}% "
            f"Trades:{s.get('total_trades', 0)} W:{s.get('wins', 0)} "
            f"L:{s.get('losses', 0)} WR:{s.get('win_rate', 0):.0f}% "
            f"Redeemed:${s.get('total_redeemed', 0):.0f}"
        ]
        for st, d in (by_strat or {}).items():
            lines.append(f"{st}: {d.get('wins', 0)}W/{d.get('losses', 0)}L "
                         f"pnl${d.get('pnl', 0):+.0f} ({d.get('trades', 0)} trades)")
        if by_city:
            for c, d in list(by_city.items())[:10]:
                lines.append(f"city {c}: {d.get('wins', 0)}W/{d.get('losses', 0)}L "
                             f"pnl${d.get('pnl', 0):+.0f}")
        for t in (recent or [])[:20]:
            lines.append(
                f"- {t.get('strategy', '?')} {t.get('city', '')} "
                f"{t.get('exit_reason', t.get('status', ''))} "
                f"pnl${t.get('pnl', 0):+.1f} roi{t.get('roi_pct', 0):+.0f}%"
            )
        prompt = (
            "You are a quantitative trading analyst reviewing a paper-trading "
            "weather prediction-market bot. Write a concise report (<170 words, "
            "plain text, no markdown headers) covering: how it is going, what is "
            "failing, what you observe per strategy, and 3 concrete improvements.\n"
            + "\n".join(lines[:40])
        )
        try:
            self._total_calls += 1
            resp = self._session.post(
                f"{self.base_url}/chat/completions",
                json={
                    'model': self.analysis_model,
                    'messages': [
                        {'role': 'system', 'content': 'You are a concise quantitative trading analyst.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    'max_tokens': self.analysis_max_tokens, 'temperature': 0.3,
                },
                timeout=max(self.timeout, 30),
            )
            if resp.status_code == 200:
                data = resp.json()
                self._total_tokens_used += data.get('usage', {}).get('total_tokens', 0)
                txt = (data.get('choices', [{}])[0].get('message', {})
                       .get('content', '') or '').strip()
                txt = self._strip_reasoning(txt).strip()
                self._last_ok_ts = time.time()
                if txt:
                    return txt
            else:
                self._last_error = f"HTTP {resp.status_code}: {resp.text[:80]}"
                log.warning(f"  ML report {self._last_error}")
        except Exception as e:
            self._last_error = str(e)[:80]
            log.warning(f"  ML report error: {self._last_error}")
        return heur

    def _report_heuristic(self, stats: Dict, by_strat: Dict,
                          by_city: Optional[Dict] = None) -> str:
        s = stats or {}
        wr = s.get('win_rate', 0)
        parts = [
            f"Trading status: balance ${s.get('balance', 0):.0f}, "
            f"portfolio ${s.get('portfolio_value', 0):.0f}, "
            f"net PnL ${s.get('total_pnl', 0):+.0f} "
            f"({s.get('roi_pct', 0):+.1f}% ROI) over {s.get('total_trades', 0)} "
            f"trades at {wr:.0f}% win rate."
        ]
        bs = by_strat or {}
        if bs:
            best = max(bs.items(), key=lambda kv: kv[1].get('pnl', 0), default=None)
            worst = min(bs.items(), key=lambda kv: kv[1].get('pnl', 0), default=None)
            if best:
                parts.append(f"Best strategy: {best[0]} (${best[1].get('pnl', 0):+.0f}).")
            if worst and worst[0] != (best[0] if best else None):
                parts.append(f"Weakest: {worst[0]} (${worst[1].get('pnl', 0):+.0f}).")
        if wr < 30:
            parts.append("Win rate is far below the 80-90% target wallets; entries "
                         "are likely too speculative - tighten edge/liquidity gates "
                         "and lean on the observed/hold-to-resolution book.")
        parts.append("(Heuristic report - enable /mlanalysis + set ML_API_KEY for an LLM-written analysis.)")
        return ' '.join(parts)

    # ------------------------------------------------------------------ #
    # Core query + parsing
    # ------------------------------------------------------------------ #
    def _query(self, prompt: str, max_tokens: Optional[int] = None) -> Dict:
        mt = int(max_tokens or self.decision_max_tokens)
        try:
            self._total_calls += 1
            resp = self._session.post(
                f"{self.base_url}/chat/completions",
                json={
                    'model': self.model,
                    'messages': [
                        {'role': 'system', 'content': 'You are a weather trading assistant. Think briefly if needed, then reply with the JSON answer only.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    'max_tokens': mt,
                    'temperature': 0.1,
                },
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                self._api_failures += 1
                self._last_error = f"HTTP {resp.status_code}: {resp.text[:80]}"
                log.warning(f"  ML API FAIL [{self._api_failures}]: {self._last_error}")
                return self._local_fallback('BUY', f'API HTTP {resp.status_code}')
            data = resp.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            self._total_tokens_used += data.get('usage', {}).get('total_tokens', 0)
            self._last_ok_ts = time.time()
            return self._parse_response(content)
        except requests.Timeout:
            self._api_failures += 1
            self._last_error = 'timeout'
            log.warning(f"  ML API TIMEOUT [{self._api_failures}] - using local model")
            return self._local_fallback('BUY', 'API timeout')
        except Exception as e:
            self._api_failures += 1
            self._last_error = str(e)[:80]
            log.warning(f"  ML API ERROR [{self._api_failures}]: {self._last_error}")
            return self._local_fallback('BUY', f'API: {str(e)[:30]}')

    def _local_fallback(self, default_action: str, reason: str) -> dict:
        if self.local_model is not None:
            result = self.local_model._rules_predict({
                "edge": 0.05, "edge_ratio": 2.0, "n_models": 3,
                "confidence": 0.6, "market_spread_bps": 500, "market_price": 0.1,
            })
            result["reason"] = f"local_fallback: {reason}"
            return result
        return {"action": default_action, "confidence": 0.5,
                "reason": f"fallback: {reason}", "source": "fallback"}

    # --- Req-31: reasoning-model aware parsing ------------------------- #
    _THINK_BLOCK_RE = re.compile(r'(?is)<think>.*?</think>')
    _THINK_OPEN_RE = re.compile(r'(?is)<think>.*$')

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove a reasoning model's <think>...</think> preamble (and any
        dangling, unclosed <think> tail) so only the final answer remains."""
        if not text:
            return ''
        text = MLDecisionEngine._THINK_BLOCK_RE.sub('', text)
        text = MLDecisionEngine._THINK_OPEN_RE.sub('', text)
        return text.strip()

    @staticmethod
    def _loads_json(s: str):
        """Parse JSON, tolerating extra prose by extracting the first balanced
        {...} object or [...] array. Returns the parsed value or None."""
        s = (s or '').strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            pass
        for open_ch, close_ch in (('{', '}'), ('[', ']')):
            start = s.find(open_ch)
            if start < 0:
                continue
            depth = 0
            for i in range(start, len(s)):
                c = s[i]
                if c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        frag = s[start:i + 1]
                        try:
                            return json.loads(frag)
                        except (json.JSONDecodeError, ValueError):
                            break
        return None

    def _parse_response(self, content: str) -> Dict:
        content = (content or '').strip()
        # Req-31: strip the reasoning model's <think>...</think> preamble first.
        content = self._strip_reasoning(content)
        if content.startswith('```'):
            content = content.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        parsed = self._loads_json(content)
        if isinstance(parsed, list):
            return {'raw': parsed, 'action': 'SELECT', 'confidence': 0.8, 'reason': ''}
        if isinstance(parsed, dict):
            action = str(parsed.get('action', parsed.get('act', 'BUY'))).upper()
            try:
                confidence = float(parsed.get('conf', parsed.get('confidence', 0.5)))
            except (ValueError, TypeError):
                confidence = 0.5
            reason = parsed.get('why', parsed.get('reason', ''))
            out = {
                'action': action,
                'confidence': min(1.0, max(0.0, confidence)),
                'reason': str(reason)[:60],
            }
            target = parsed.get('target', parsed.get('tp', parsed.get('target_roi')))
            if target is not None:
                try:
                    out['target_roi'] = float(target)
                except (ValueError, TypeError):
                    pass
            return out
        # Fallback: keyword scan on the cleaned text.
        cu = content.upper()
        if 'SKIP' in cu:
            return {'action': 'SKIP', 'confidence': 0.5, 'reason': 'parsed from text'}
        if 'SELL' in cu or 'BOOK' in cu:
            return {'action': 'SELL', 'confidence': 0.5, 'reason': 'parsed from text'}
        if 'HOLD' in cu:
            return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'parsed from text'}
        return {'action': 'BUY', 'confidence': 0.5, 'reason': 'parse failed'}

    # ------------------------------------------------------------------ #
    # Status / self-test
    # ------------------------------------------------------------------ #
    def get_token_usage(self) -> int:
        return self._total_tokens_used

    def get_status(self) -> Dict:
        local_status = self.local_model.get_status() if self.local_model else {"model": "none"}
        return {
            'enabled': self.enabled,
            'model': self.model if self.enabled else 'local',
            'analysis_model': self.analysis_model if self.enabled else 'local',
            'analysis_enabled': self.analysis_enabled,
            'local_model': local_status.get("model", "none"),
            'tokens_used': self._total_tokens_used,
            'calls': self._total_calls,
            'api_failures': self._api_failures,
            'timeout_s': self.timeout,
            'last_error': self._last_error,
            'cache_size': len(self._cache),
        }

    def self_test(self) -> Dict:
        """Live round-trip test of the API key/URL/model. Returns a dict with ok/latency/reply."""
        if not self.enabled:
            return {'ok': False, 'reason': 'ML_API_KEY not set', 'url': self.base_url,
                    'model': self.model}
        t0 = time.time()
        try:
            resp = self._session.post(
                f"{self.base_url}/chat/completions",
                json={'model': self.model,
                      'messages': [{'role': 'user', 'content': 'Reply with the single word OK.'}],
                      'max_tokens': self.decision_max_tokens, 'temperature': 0.0},
                timeout=max(self.timeout, 15),
            )
            dt = time.time() - t0
            ok = resp.status_code == 200
            body = ''
            if ok:
                body = (resp.json().get('choices', [{}])[0].get('message', {})
                        .get('content', '') or '').strip()
                body = self._strip_reasoning(body)
            return {'ok': ok, 'status': resp.status_code, 'latency_s': round(dt, 2),
                    'reply': body[:60], 'url': self.base_url, 'model': self.model,
                    'error': '' if ok else resp.text[:120]}
        except Exception as e:
            return {'ok': False, 'latency_s': round(time.time() - t0, 2),
                    'url': self.base_url, 'model': self.model, 'error': str(e)[:120]}


if __name__ == '__main__':
    # Run on a networked host (e.g. Railway shell) to verify the key works:
    #   python -m ml.decision_engine
    eng = MLDecisionEngine()
    print('status:', eng.get_status())
    print('self_test:', eng.self_test())
