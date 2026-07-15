"""
Telegram Bot Integration — Notifications + Commands.

Features:
- Send trade alerts (entry, exit, win, loss)
- Detailed redemption alerts (full market, entry/exit, profit/PnL)
- ONE grouped "Peak Cluster Box N" alert per basket (not one per leg)
- Send daily summary
- Commands: /status, /positions, /balance, /pnl, /markets, /stop
- Paginated + sortable positions view (10 per page), peak-cluster legs grouped
- Non-blocking (runs in background thread)
"""

import os
import csv
import json
import html
import time
import logging
import threading
import requests
from collections import deque
from typing import Optional, Dict, List
from datetime import datetime, timezone

from config import Config
from logger import log


class TelegramBot:
    """Telegram bot for notifications and commands."""

    PAGE_SIZE = 10
    _SORT_NAMES = {
        'pnl': 'Top PnL', 'loss': 'Biggest losers',
        'roi': 'Top ROI', 'recent': 'Most recent',
    }

    def __init__(self, position_manager=None, scanner=None):
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        self.pm = position_manager
        self.scanner = scanner
        self._session = requests.Session()
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_update_id = 0
        # Optional dashboard hook: restart_fresh(starting_balance=None) clears
        # ALL positions and resets the paper balance for a fresh start. Set by
        # the dashboard; the inline Restart button / /restart invoke it.
        self._on_restart = None
        self._restart_pending = False
        # Req-29 settings UX: capture typed input (e.g. a new starting balance),
        # and log human-readable changes so the OK button can summarise them.
        self._awaiting = None          # None | 'balance'
        self._session_changes = []     # ["STARTING_BALANCE = 300", ...]
        # Req-29 mlanalysis: optional ML engine handle (set by the dashboard via
        # attach_ml); None keeps mlanalysis on its heuristic fallback.
        self.ml = None
        # Req-29 ai-summary: capture WARNING+ log lines into a ring buffer so
        # /aisummary can surface recent runtime errors for sharing.
        self._error_log = deque(maxlen=300)
        self._install_error_capture()
        # Seed with already-redeemed ids so a restart doesn't re-announce the
        # whole backlog — only NEW redemptions after startup are sent.
        self._announced_redeemed = set(
            p.id for p in self.pm.positions if p.status == 'redeemed'
        ) if self.pm else set()

        if not self.enabled:
            log.info("Telegram: disabled (no token/chat_id set)")
        else:
            log.info(f"Telegram: enabled → chat {self.chat_id}")

    @property
    def base_url(self):
        return "https" + "://api.telegram.org/bot" + str(self.token)

    @staticmethod
    def _esc(s) -> str:
        """HTML-escape dynamic text so market names with &/</> don't break parse."""
        return html.escape(str(s if s is not None else ''))

    # ==============================================================
    # SEND MESSAGES
    # ==============================================================

    def send(self, text: str, parse_mode: str = 'HTML', reply_markup: dict = None) -> bool:
        """Send a message to the configured chat (optionally with an inline keyboard)."""
        if not self.enabled:
            return False
        # Intercept the legacy blind "Redeemed N winning positions!" message and
        # replace it with the detailed per-position breakdown (full market, entry/
        # exit, cost/payout, PnL). The detailed header starts with "<b>REDEEMED",
        # so it never matches this guard and there is no recursion.
        stripped = text.strip()
        if stripped.startswith('\U0001F4B0 Redeemed ') and stripped.endswith('positions!'):
            self.notify_redeems_recent()
            return True
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True,
            }
            if reply_markup is not None:
                payload['reply_markup'] = reply_markup
            resp = self._session.post(
                f"{self.base_url}/sendMessage", json=payload, timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            log.debug(f"Telegram send failed: {e}")
            return False

    def _edit(self, message_id: int, text: str, reply_markup: dict = None) -> bool:
        """Edit an existing message (used to refresh panels in place)."""
        if not self.enabled:
            return False
        try:
            payload = {
                'chat_id': self.chat_id, 'message_id': message_id,
                'text': text, 'parse_mode': 'HTML', 'disable_web_page_preview': True,
            }
            if reply_markup is not None:
                payload['reply_markup'] = reply_markup
            r = self._session.post(f"{self.base_url}/editMessageText", json=payload, timeout=10)
            return r.status_code == 200
        except Exception as e:
            log.debug(f"Telegram edit failed: {e}")
            return False

    def _answer_callback(self, callback_id: str, text: str = ''):
        try:
            self._session.post(f"{self.base_url}/answerCallbackQuery",
                               json={'callback_query_id': callback_id, 'text': text},
                               timeout=10)
        except Exception:
            pass

    def _install_error_capture(self):
        """Attach a handler to the root logger that records WARNING+ lines into
        an in-memory ring buffer (for /aisummary). Idempotent + defensive."""
        try:
            buf = self._error_log

            class _RingHandler(logging.Handler):
                def emit(self, record):
                    try:
                        if record.levelno >= logging.WARNING:
                            ts = datetime.now(timezone.utc).strftime('%m-%d %H:%M:%S')
                            buf.append(
                                f"{ts} {record.levelname} "
                                f"{record.name}: {record.getMessage()}"
                            )
                    except Exception:
                        pass

            root = logging.getLogger()
            if not any(getattr(h, '_wp_ring', False) for h in root.handlers):
                h = _RingHandler()
                h._wp_ring = True
                h.setLevel(logging.WARNING)
                root.addHandler(h)
                if root.level == 0 or root.level > logging.WARNING:
                    root.setLevel(logging.WARNING)
        except Exception:
            pass

    def attach_ml(self, ml):
        """Wire the ML decision engine so /mlanalysis can use it for a narrative."""
        self.ml = ml

    # ==============================================================
    # LIFECYCLE (startup ready / start / restart fresh)
    # ==============================================================

    def _main_keyboard(self) -> dict:
        """Inline keyboard shown on startup: Start / Settings / Restart."""
        return {'inline_keyboard': [[
            {'text': '▶️ Start Trading', 'callback_data': 'act:start'},
            {'text': '⚙️ Settings', 'callback_data': 'act:settings'},
            {'text': '♻️ Restart', 'callback_data': 'act:restart'},
        ]]}

    def send_startup_ready(self):
        """Announce a successful deploy/boot WITHOUT auto-trading and show the
        Start / Settings / Restart inline keyboard. Trading begins only when the
        user taps Start Trading (or sends /start, or types 'start')."""
        try:
            from bot import settings_store
            bools, _nums = settings_store.snapshot()
        except Exception:
            bools = {}
        try:
            bal = self.pm.get_balance() if self.pm else 0.0
        except Exception:
            bal = 0.0
        mode = '📋 PAPER' if Config.is_paper() else '🔴 LIVE'
        trading = '🟢 ON' if bools.get('TRADING_ENABLED') else '🔴 OFF (tap Start Trading)'
        msg = (
            f"✅ <b>Bot initialized successfully</b>\n"
            f"{mode} | starting balance ${bal:.2f}\n"
            f"Trading: <b>{trading}</b>\n\n"
            f"▶️ <b>Start Trading</b> — begin placing trades (or send /start)\n"
            f"⚙️ <b>Settings</b> — strategies, gates & starting balance\n"
            f"♻️ <b>Restart</b> — clear ALL positions & start fresh\n"
        )
        self.send(msg, reply_markup=self._main_keyboard())

    def _prompt_restart(self):
        """Ask for confirmation before the destructive restart-fresh action."""
        self._restart_pending = True
        kb = {'inline_keyboard': [[
            {'text': '✅ Yes, clear all & restart', 'callback_data': 'act:restart_confirm'},
            {'text': '✖️ Cancel', 'callback_data': 'act:restart_cancel'},
        ]]}
        self.send(
            "♻️ <b>Restart fresh?</b>\n"
            "This CLOSES/clears ALL positions and resets the paper balance to "
            "the configured starting balance. This cannot be undone.",
            reply_markup=kb,
        )

    def _do_restart(self):
        """Invoke the dashboard restart hook (clear all positions + reset balance)."""
        self._restart_pending = False
        if not self._on_restart:
            self.send("⚠️ Restart hook not wired — cannot restart from here.")
            return
        try:
            self._on_restart()
            try:
                bal = self.pm.get_balance() if self.pm else 0.0
            except Exception:
                bal = 0.0
            self.send(
                f"♻️ <b>Restarted fresh</b> — all positions cleared, "
                f"balance reset to ${bal:.2f}. Tap Start Trading to begin.",
                reply_markup=self._main_keyboard(),
            )
        except Exception as e:
            log.debug(f"restart failed: {e}")
            self.send("⚠️ Restart failed — see logs.")

    def notify_trade(self, side: str, bucket_label: str, price: float,
                     size_usd: float, shares: float, strategy: str,
                     edge: float = 0, city: str = ''):
        """Send trade notification."""
        emoji = '🟢' if side == 'BUY' else '🔴'
        msg = (
            f"{emoji} <b>{side}</b> — {self._esc(strategy.upper())}\n"
            f"📍 {self._esc(city)} | {self._esc(bucket_label)}\n"
            f"💰 ${price:.4f} × {shares:.0f} = ${size_usd:.2f}\n"
        )
        if edge > 0:
            msg += f"📊 Edge: {edge:.1%}\n"
        mode = '📋 PAPER' if Config.is_paper() else '🔴 LIVE'
        msg += f"\n{mode}"
        self.send(msg)

    def notify_cluster(self, box_label: str, city: str, market_title: str,
                       legs: List, total_cost: float, combined_prob: float,
                       roi_pct: float, group_label: str = None):
        """ONE grouped alert for a whole peak-cluster basket.

        Replaces the old behaviour of firing a separate notify_trade per leg
        (6 buckets => 6 messages). Now a single "🧺 PEAK CLUSTER Box N" message
        lists every bucket bought, the combined basket cost, and the any-one-
        wins ROI. `legs` is the list of placed TrackedPositions in the basket.
        """
        try:
            n = len(legs)
            total_cost_usd = sum(getattr(l, 'cost_usd', 0.0) or 0.0 for l in legs)
            title = (group_label or 'PEAK CLUSTER').upper()
            head = (
                f"🧺 <b>{self._esc(title)} {self._esc(box_label)}</b> — {n} bucket{'s' if n != 1 else ''}\n"
                f"📍 {self._esc(city)} | {self._esc((market_title or '')[:60])}\n"
            )
            lines = []
            for l in legs:
                lines.append(
                    f"   • {self._esc(getattr(l, 'bucket_label', ''))} "
                    f"@ ${getattr(l, 'entry_price', 0.0):.3f} × "
                    f"{getattr(l, 'shares', 0.0):.0f} = ${getattr(l, 'cost_usd', 0.0):.2f}\n"
                )
            foot = (
                f"💰 basket cost ${total_cost_usd:.2f} "
                f"(per-share ${total_cost:.3f}) | P(any)~{combined_prob:.0%}\n"
                f"🎯 ROI ~{roi_pct:.0f}% if ANY bucket wins | holds → resolution "
                f"(never stop-lossed)\n"
            )
            mode = '📋 PAPER' if Config.is_paper() else '🔴 LIVE'
            self.send(head + ''.join(lines) + foot + f"\n{mode}")
        except Exception as e:
            log.debug(f"notify_cluster failed: {e}")

    def notify_cluster_resolution(self, box_label: str, legs: List):
        """ONE grouped resolution summary for a peak-cluster basket once EVERY
        leg has settled. Shows which bucket WON and the amount it won, plus the
        losing buckets and their loss, and the net basket PnL. Replaces the
        per-leg won/lost spam for cluster baskets.

        `legs` is the list of resolved TrackedPositions in the basket (fed by
        PositionManager._maybe_notify_cluster_close once none are open/pending).
        """
        try:
            if not legs:
                return
            city = self._esc(getattr(legs[0], 'city', ''))
            market_title = self._esc((getattr(legs[0], 'market_title', '') or '')[:60])
            winners = [l for l in legs if getattr(l, 'status', '') in ('won', 'redeemed')]
            losers = [l for l in legs if getattr(l, 'status', '') == 'lost']
            others = [l for l in legs if l not in winners and l not in losers]
            net = sum(getattr(l, 'pnl', 0.0) or 0.0 for l in legs)
            cost = sum(getattr(l, 'cost_usd', 0.0) or 0.0 for l in legs)
            ret = cost + net
            head_emoji = '✅' if net >= 0 else '🔴'
            head = (
                f"{head_emoji} 🧺 <b>PEAK CLUSTER {self._esc(box_label)} RESOLVED</b>\n"
                f"📍 {city} | {market_title}\n"
            )
            lines = []
            if winners:
                for l in winners:
                    payout = (getattr(l, 'shares', 0.0) or 0.0) * 1.0
                    lines.append(
                        f"   ✅ WON {self._esc(getattr(l, 'bucket_label', ''))} "
                        f"→ ${getattr(l, 'pnl', 0.0):+.2f} "
                        f"(entry ${getattr(l, 'entry_price', 0.0):.3f} × "
                        f"{getattr(l, 'shares', 0.0):.0f}sh → payout ${payout:.2f})\n"
                    )
            else:
                lines.append("   ⚠️ No winning bucket in this basket.\n")
            for l in losers:
                lines.append(
                    f"   ❌ {self._esc(getattr(l, 'bucket_label', ''))} "
                    f"→ ${getattr(l, 'pnl', 0.0):+.2f} "
                    f"(cost ${getattr(l, 'cost_usd', 0.0):.2f} lost)\n"
                )
            for l in others:
                lines.append(
                    f"   • {self._esc(getattr(l, 'bucket_label', ''))} "
                    f"→ ${getattr(l, 'pnl', 0.0):+.2f} ({self._esc(getattr(l, 'status', ''))})\n"
                )
            foot = (
                f"💰 <b>Basket net PnL ${net:+.2f}</b> "
                f"(cost ${cost:.2f} → return ${ret:.2f})\n"
            )
            mode = '📋 PAPER' if Config.is_paper() else '🔴 LIVE'
            self.send(head + ''.join(lines) + foot + f"\n{mode}")
        except Exception as e:
            log.debug(f"notify_cluster_resolution failed: {e}")

    def notify_resolution(self, won: bool, bucket_label: str, pnl: float, city: str = ''):
        """Send simple resolution notification (kept for compatibility)."""
        emoji = '✅' if won else '❌'
        result = 'WON' if won else 'LOST'
        msg = (
            f"{emoji} <b>RESOLVED: {result}</b>\n"
            f"📍 {self._esc(city)} | {self._esc(bucket_label)}\n"
            f"💰 PnL: ${pnl:+.2f}\n"
        )
        self.send(msg)

    def notify_close(self, pos):
        """Send a close/resolution alert for ANY closed position — stop-loss,
        take-profit, trailing-stop, flip/thesis exit, or won/lost resolution.

        Wired via PositionManager._notify_close (risk-trigger & resolution
        closes) and called directly by the dashboard for flip/thesis exits
        (whose reason is relabeled 'manual' after close, so the PM hook skips
        them to avoid a double-notify). Fully defensive — never raises."""
        try:
            reason = getattr(pos, 'exit_reason', '') or ''
            status = getattr(pos, 'status', '') or ''
            pnl = getattr(pos, 'pnl', 0.0) or 0.0
            roi = getattr(pos, 'roi_pct', 0.0) or 0.0
            if status == 'won':
                head = '✅ <b>RESOLVED WON</b>'
            elif status == 'lost':
                head = '❌ <b>RESOLVED LOST</b>'
            else:
                head = {
                    'take_profit': '🎯 <b>TAKE PROFIT</b>',
                    'stop_loss': '🛑 <b>STOP LOSS</b>',
                    'trailing_stop': '📉 <b>TRAILING STOP</b>',
                    'flip_timeout': '⏲️ <b>FLIP book-or-cut</b>',
                    'thesis_invalidated': '🚫 <b>THESIS EXIT</b>',
                    'manual': '🔴 <b>SOLD</b>',
                }.get(reason, '🔴 <b>SOLD</b>')
            entry = getattr(pos, 'entry_price', 0.0) or 0.0
            exit_px = getattr(pos, 'exit_price', None)
            if exit_px is None:
                exit_px = getattr(pos, 'current_price', 0.0) or 0.0
            shares = getattr(pos, 'shares', 0.0) or 0.0
            name = self._esc(getattr(pos, 'bucket_label', '') or getattr(pos, 'market_title', ''))
            mode = '📋 PAPER' if Config.is_paper() else '🔴 LIVE'
            msg = (
                f"{head} — {self._esc(getattr(pos, 'strategy', ''))}\n"
                f"📍 {self._esc(getattr(pos, 'city', ''))} | {name}\n"
                f"💵 entry ${entry:.4f} → exit ${exit_px:.4f} | {shares:.0f}sh\n"
                f"📊 PnL ${pnl:+.2f} ({roi:+.0f}%)\n"
                f"{mode}"
            )
            self.send(msg)
        except Exception as e:
            log.debug(f"notify_close failed: {e}")

    def notify_redeems_recent(self):
        """Find positions that have newly become 'redeemed' since the last call
        and announce them in full detail. Self-discovers from the position
        manager so the dashboard doesn't need to pass anything."""
        if not self.pm:
            return
        fresh = [p for p in self.pm.positions
                 if p.status == 'redeemed' and p.id not in self._announced_redeemed]
        for p in fresh:
            self._announced_redeemed.add(p.id)
        if fresh:
            self.notify_redeems(fresh)

    def notify_redeems(self, positions: List):
        """Detailed redemption notification — one block per redeemed position with
        the full market name, entry/exit price, cost/payout and realized PnL."""
        if not positions:
            return
        payout_total = sum(p.shares * 1.0 for p in positions)
        pnl_total = sum(p.pnl for p in positions)
        n = len(positions)
        header = (
            f"💰 <b>REDEEMED {n} winning position{'s' if n != 1 else ''}</b>\n"
            f"   payout +${payout_total:.2f} | realized PnL ${pnl_total:+.2f}\n"
        )
        blocks = []
        for p in positions:
            exit_px = p.exit_price if p.exit_price is not None else 1.0
            payout = p.shares * 1.0
            name = self._esc(p.bucket_label or p.market_title)
            blocks.append(
                f"\n✅ <b>{self._esc(p.city)}</b>  ({self._esc(p.strategy)})\n"
                f"   {name}\n"
                f"   entry ${p.entry_price:.4f} → exit ${exit_px:.4f} | {p.shares:.0f}sh\n"
                f"   cost ${p.cost_usd:.2f} → payout ${payout:.2f} | "
                f"PnL ${p.pnl:+.2f} ({p.roi_pct:+.0f}%)\n"
            )
        # Respect Telegram's ~4096-char message cap — chunk if necessary.
        msg = header
        for b in blocks:
            if len(msg) + len(b) > 3900:
                self.send(msg)
                msg = ''
            msg += b
        if msg:
            self.send(msg)

    def notify_redeem(self, bucket_label: str, amount: float):
        """Legacy single-redeem notification (kept for compatibility)."""
        msg = (
            f"💰 <b>REDEEMED</b>\n"
            f"📍 {self._esc(bucket_label)}\n"
            f"💵 +${amount:.2f}\n"
        )
        self.send(msg)

    # ==============================================================
    # POSITIONS VIEW (paginated + sortable, peak-cluster legs GROUPED)
    # ==============================================================

    def _open_units(self, sort_key: str) -> List[dict]:
        """Group open positions into display UNITS so a peak-cluster basket shows
        as ONE entry ("Box N" + all its legs) instead of N separate rows.

        Each unit: {kind, box, positions, pnl, roi, recent}. Non-cluster
        positions are single-position units. Units are sorted as a whole.
        """
        open_pos = self.pm.get_open_positions() if self.pm else []
        clusters: Dict[str, list] = {}
        units: List[dict] = []
        basket_strats = ('peak_cluster', 'peaker_cool_basket', 'peaker_warm_basket')
        for p in open_pos:
            box = getattr(p, 'cluster_box', '') or ''
            if box and getattr(p, 'strategy', '') in basket_strats:
                clusters.setdefault(box, []).append(p)
            else:
                units.append({
                    'kind': 'single', 'box': '', 'positions': [p],
                    'pnl': p.unrealized_pnl, 'roi': p.roi_pct,
                    'recent': p.entry_time,
                    'strategy': getattr(p, 'strategy', '') or '',
                })
        for box, legs in clusters.items():
            pnl = sum(l.unrealized_pnl for l in legs)
            cost = sum(l.cost_usd for l in legs)
            roi = (pnl / cost * 100.0) if cost > 0 else 0.0
            recent = max(l.entry_time for l in legs)
            strat = getattr(legs[0], 'strategy', 'peak_cluster') if legs else 'peak_cluster'
            units.append({
                'kind': 'cluster', 'box': box, 'positions': legs,
                'pnl': pnl, 'roi': roi, 'recent': recent, 'strategy': strat,
            })
        if sort_key == 'pnl':
            units.sort(key=lambda u: u['pnl'], reverse=True)
        elif sort_key == 'loss':
            units.sort(key=lambda u: u['pnl'])
        elif sort_key == 'roi':
            units.sort(key=lambda u: u['roi'], reverse=True)
        elif sort_key == 'strategy':
            units.sort(key=lambda u: (u.get('strategy', '') or '', -u['pnl']))
        else:  # 'recent'
            units.sort(key=lambda u: u['recent'], reverse=True)
        return units

    def _fmt_position(self, p, idx: int) -> str:
        pe = '🟢' if p.unrealized_pnl >= 0 else '🔴'
        lock = ' 🔒' if getattr(p, 'preclose_locked', False) else ''
        stale = ' ~stale' if getattr(p, 'current_price_stale', False) else ''
        name = self._esc(p.bucket_label or p.market_title)
        return (
            f"{idx}. {pe} <b>{self._esc(p.city)}</b>  "
            f"${p.unrealized_pnl:+.2f} ({p.roi_pct:+.0f}%){lock}{stale}\n"
            f"   {name}\n"
            f"   entry ${p.entry_price:.4f} → ${p.current_price:.4f} | "
            f"{p.shares:.0f}sh | cost ${p.cost_usd:.2f} | {self._esc(p.strategy)}\n\n"
        )

    def _fmt_cluster_unit(self, unit: dict, idx: int) -> str:
        """Render a whole peak-cluster basket as ONE grouped block: a "Box N"
        header with the aggregate PnL, then each bucket leg indented under it."""
        legs = unit['positions']
        pe = '🟢' if unit['pnl'] >= 0 else '🔴'
        city = self._esc(getattr(legs[0], 'city', '') if legs else '')
        cost = sum(l.cost_usd for l in legs)
        label = {
            'peak_cluster': 'Peak Cluster',
            'peaker_cool_basket': 'Peaker Cool Basket',
            'peaker_warm_basket': 'Peaker Warm Basket',
        }.get(unit.get('strategy', 'peak_cluster'), 'Peak Cluster')
        out = (
            f"{idx}. {pe} 🧺 <b>{self._esc(label)} {self._esc(unit['box'])}</b> — {city}  "
            f"${unit['pnl']:+.2f} ({unit['roi']:+.0f}%)\n"
            f"   {len(legs)} buckets | cost ${cost:.2f} | hold → resolution\n"
        )
        for l in legs:
            name = self._esc(l.bucket_label or l.market_title)
            out += (
                f"      • {name}: ${l.entry_price:.3f}→${l.current_price:.3f} "
                f"{l.shares:.0f}sh (${l.unrealized_pnl:+.2f})\n"
            )
        out += "\n"
        return out

    def _positions_view(self, page: int = 0, sort: str = 'recent',
                        with_summary: bool = False):
        """Build (text, inline_keyboard) for a page of open positions.

        Pagination is by display UNIT (a peak-cluster basket counts as one
        unit), so a 6-leg basket no longer eats 6 of the 10 page slots.
        """
        units = self._open_units(sort)
        total_units = len(units)
        total_pos = sum(len(u['positions']) for u in units)
        pages = max(1, (total_units + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        page = max(0, min(page, pages - 1))
        start = page * self.PAGE_SIZE
        chunk = units[start:start + self.PAGE_SIZE]

        text = ''
        if with_summary and self.pm:
            s = self.pm.get_stats()
            text += (
                f"📊 <b>Weather Sniper Status</b>\n"
                f"Mode: {s['mode']} | Balance: ${s['balance']:.2f}\n"
                f"Positions value: ${s.get('position_value', 0.0):.2f} | "
                f"Portfolio: ${s['portfolio_value']:.2f}\n"
                f"PnL: ${s['total_pnl']:+.2f} ({s['roi_pct']:+.1f}%) | "
                f"WR: {s['win_rate']:.0f}% ({s['wins']}W/{s['losses']}L)\n"
                f"Trades: {s['total_trades']} | Open: {s['open_positions']} | "
                f"Redeemed: ${s['total_redeemed']:.2f}\n"
                f"{'-'*28}\n"
            )
        sort_name = self._SORT_NAMES.get(sort, sort)
        shown_to = start + len(chunk)
        text += (f"<b>Open {start + 1}-{shown_to} of {total_units} "
                 f"({total_pos} positions)</b> · sorted: {sort_name}\n\n")
        if not chunk:
            text += "No open positions.\n"
        else:
            last_strat = None
            for i, u in enumerate(chunk, start=start + 1):
                if sort == 'strategy':
                    su = u.get('strategy', '') or '—'
                    if su != last_strat:
                        text += f"\n📂 <b>{self._esc(su)}</b>\n"
                        last_strat = su
                if u['kind'] == 'cluster':
                    text += self._fmt_cluster_unit(u, i)
                else:
                    text += self._fmt_position(u['positions'][0], i)

        sm = '1' if with_summary else '0'
        nav = []
        if page > 0:
            nav.append({'text': '⬅️ Prev', 'callback_data': f"pos:{page-1}:{sort}:{sm}"})
        nav.append({'text': f"{page+1}/{pages}", 'callback_data': 'noop'})
        if page < pages - 1:
            nav.append({'text': 'Next ➡️', 'callback_data': f"pos:{page+1}:{sort}:{sm}"})
        dot = lambda k: ('• ' if sort == k else '')
        sort_row = [
            {'text': dot('pnl') + '💰 PnL', 'callback_data': f"pos:0:pnl:{sm}"},
            {'text': dot('loss') + '📉 Losses', 'callback_data': f"pos:0:loss:{sm}"},
            {'text': dot('roi') + '📈 ROI', 'callback_data': f"pos:0:roi:{sm}"},
            {'text': dot('recent') + '🕒 Recent', 'callback_data': f"pos:0:recent:{sm}"},
        ]
        strat_row = [
            {'text': dot('strategy') + '🗂 By strategy',
             'callback_data': f"pos:0:strategy:{sm}"},
        ]
        return text, {'inline_keyboard': [nav, sort_row, strat_row]}

    def send_positions(self, page: int = 0, sort: str = 'recent',
                       with_summary: bool = False, edit_message_id: int = None):
        if not self.pm:
            return
        text, kb = self._positions_view(page, sort, with_summary)
        if edit_message_id is not None:
            self._edit(edit_message_id, text, kb)
        else:
            self.send(text, reply_markup=kb)

    def send_status(self):
        """Status = summary + first page of open positions (paginated/sortable)."""
        if not self.pm:
            return
        self.send_positions(page=0, sort='recent', with_summary=True)

    def send_markets_summary(self):
        """Send summary of available markets."""
        if not self.scanner:
            return
        markets = self.scanner.scan_weather_markets(days_ahead=2)
        msg = f"🌤️ <b>Active Weather Markets: {len(markets)}</b>\n\n"
        by_city: Dict[str, int] = {}
        for m in markets:
            by_city[m.city] = by_city.get(m.city, 0) + 1
        for city, count in sorted(by_city.items(), key=lambda x: -x[1]):
            msg += f"  📍 {self._esc(city)}: {count} markets\n"
        self.send(msg)

    def _outcome_breakdown_text(self) -> str:
        """Grouped outcome breakdown (Req-30): settlements/redeems kept SEPARATE
        from the small quick-flip/exit scalps (gains & losses)."""
        if not self.pm or not hasattr(self.pm, 'get_outcome_breakdown'):
            return ''
        try:
            b = self.pm.get_outcome_breakdown()
        except Exception:
            return ''
        g = lambda k: b.get(k, {'count': 0, 'pnl': 0.0})
        sw, rd, sl = g('settle_win'), g('redeemed'), g('settle_loss')
        sg, slo = g('small_gain'), g('small_loss')
        return (
            f"🏦 <b>Settled/Redeemed</b>: ✅ {sw['count']} ${sw['pnl']:+.2f} | "
            f"💰 {rd['count']} ${rd['pnl']:+.2f} | "
            f"❌ {sl['count']} ${sl['pnl']:+.2f}\n"
            f"⚡ <b>Flip/exit scalps</b>: 🟢 {sg['count']} ${sg['pnl']:+.2f} | "
            f"🔴 {slo['count']} ${slo['pnl']:+.2f}\n"
        )

    def send_periodic_summary(self, interval_min: int = 0):
        """Periodic status summary pushed every SUMMARY_INTERVAL_MIN minutes
        (Req-30 summary timer): balance, position value, PnL, WR + the grouped
        settle/redeem vs flip-scalp breakdown."""
        if not self.pm:
            return
        s = self.pm.get_stats()
        hdr = (f"⏲️ <b>Summary</b> (every {interval_min}m)\n"
               if interval_min else "⏲️ <b>Summary</b>\n")
        msg = (
            hdr + f"{'-'*28}\n"
            f"Mode: {s['mode']} | Balance: ${s['balance']:.2f}\n"
            f"Positions value: ${s.get('position_value', 0.0):.2f} "
            f"(open {s['open_positions']})\n"
            f"Portfolio: ${s['portfolio_value']:.2f}\n"
            f"PnL: ${s['total_pnl']:+.2f} ({s['roi_pct']:+.1f}%) | "
            f"WR {s['win_rate']:.0f}% ({s['wins']}W/{s['losses']}L)\n"
        )
        msg += self._outcome_breakdown_text()
        self.send(msg)

    def send_daily_summary(self):
        """Send end-of-day summary."""
        if not self.pm:
            return
        stats = self.pm.get_stats()
        today_positions = [p for p in self.pm.positions
                          if p.entry_time.date() == datetime.now(timezone.utc).date()]
        today_pnl = sum(p.pnl for p in today_positions if p.status != 'open')
        msg = (
            f"📅 <b>Daily Summary</b>\n"
            f"{'-'*30}\n"
            f"New trades today: {len(today_positions)}\n"
            f"Today's PnL: ${today_pnl:+.2f}\n"
            f"Total PnL: ${stats['total_pnl']:+.2f}\n"
            f"Balance: ${stats['balance']:.2f}\n"
            f"Positions value: ${stats.get('position_value', 0.0):.2f}\n"
            f"Portfolio: ${stats['portfolio_value']:.2f}\n"
            f"Win Rate: {stats['win_rate']:.0f}%\n"
        )
        msg += self._outcome_breakdown_text()
        self.send(msg)

    # ==============================================================
    # ANALYSIS (/analysis) — per-strategy performance + downloadable trade log
    # ==============================================================

    def _trade_log_path(self) -> str:
        """Resolve the paper-trade JSONL path (PositionManager's, else Config)."""
        path = getattr(self.pm, '_paper_trades_file', None) if self.pm else None
        return path or getattr(Config, 'PAPER_TRADE_LOG', 'data/paper_trades.jsonl')

    def _read_trade_log(self) -> List[dict]:
        """Read every structured record from data/paper_trades.jsonl (one per
        BUY / SELL / SETTLE / REDEEM / PRECLOSE_LOCK). Returns [] if missing."""
        path = self._trade_log_path()
        recs: List[dict] = []
        try:
            if not os.path.exists(path):
                return recs
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            log.debug(f"trade log read failed: {e}")
        return recs

    def _send_document(self, file_path: str, caption: str = '') -> bool:
        """Upload a file to the chat as a downloadable document (sendDocument).
        Used by /analysis to ship the raw trade log. Fully defensive."""
        if not self.enabled:
            return False
        try:
            if not os.path.exists(file_path):
                self.send(f"⚠️ Log file not found: {self._esc(file_path)}")
                return False
            with open(file_path, 'rb') as fh:
                files = {'document': (os.path.basename(file_path), fh)}
                data = {'chat_id': self.chat_id}
                if caption:
                    data['caption'] = caption[:1000]
                resp = self._session.post(
                    f"{self.base_url}/sendDocument",
                    data=data, files=files, timeout=30,
                )
            return resp.status_code == 200
        except Exception as e:
            log.debug(f"Telegram sendDocument failed: {e}")
            return False

    _CSV_COLUMNS = [
        'ts', 'action', 'city', 'bucket', 'market', 'strategy', 'signal',
        'entry_price', 'exit_price', 'shares', 'cost_usd', 'edge', 'grade',
        'status', 'exit_reason', 'settle_source', 'pnl', 'roi_pct',
        'minutes_to_close', 'balance_after',
    ]

    def _csv_path(self) -> str:
        base = self._trade_log_path()
        if base.endswith('.jsonl'):
            return base[:-6] + '.csv'
        return base + '.csv'

    def _write_trades_csv(self, recs: List[dict]) -> Optional[str]:
        """Flatten the trade-log records into a tidy CSV for download."""
        if not recs:
            return None
        path = self._csv_path()
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self._CSV_COLUMNS,
                                   extrasaction='ignore')
                w.writeheader()
                for r in recs:
                    w.writerow({k: r.get(k, '') for k in self._CSV_COLUMNS})
            return path
        except Exception as e:
            log.debug(f"csv write failed: {e}")
            return None

    def send_analysis(self):
        """/analysis — full strategy performance breakdown + a downloadable log
        of every BUY / SELL / SETTLE / REDEEM / exit.

        Live W/L/PnL come from the PositionManager (thesis exits ARE counted as
        losses — see PositionManager._closed_outcome), and per-strategy BUY
        counts + action/exit tallies come from data/paper_trades.jsonl.
        """
        if not self.pm:
            self.send("⚠️ Analysis unavailable — position manager not wired.")
            return
        stats = self.pm.get_stats()
        by_strat = self.pm.get_per_strategy_stats()
        recs = self._read_trade_log()

        # Tally actions + per-strategy BUY counts + exit reasons from the log.
        action_counts: Dict[str, int] = {}
        buys_by_strat: Dict[str, int] = {}
        exit_counts: Dict[str, int] = {}
        for r in recs:
            a = r.get('action', '') or '?'
            action_counts[a] = action_counts.get(a, 0) + 1
            if a == 'BUY':
                s = r.get('strategy', '?') or '?'
                buys_by_strat[s] = buys_by_strat.get(s, 0) + 1
            if a in ('SELL', 'SETTLE'):
                xr = r.get('exit_reason', '') or '—'
                exit_counts[xr] = exit_counts.get(xr, 0) + 1

        text = (
            f"📈 <b>Strategy Analysis</b> — {stats['mode']}\n"
            f"Balance ${stats['balance']:.2f} | PnL ${stats['total_pnl']:+.2f} "
            f"({stats['roi_pct']:+.1f}%)\n"
            f"WR {stats['win_rate']:.0f}% ({stats['wins']}W/{stats['losses']}L) | "
            f"Trades {stats['total_trades']} | Open {stats['open_positions']} | "
            f"Redeemed ${stats['total_redeemed']:.2f}\n"
            f"Positions value ${stats.get('position_value', 0.0):.2f} | "
            f"Portfolio ${stats['portfolio_value']:.2f}\n"
            f"{'-'*28}\n"
            f"<b>By strategy</b> (buys · W/L · WR · PnL)\n"
        )
        text += self._outcome_breakdown_text()
        if not by_strat:
            text += "  (no trades yet)\n"
        else:
            for strat, s in sorted(by_strat.items(),
                                   key=lambda kv: kv[1]['pnl'], reverse=True):
                closed = s['wins'] + s['losses']
                wr = (s['wins'] / closed * 100.0) if closed else 0.0
                buys = buys_by_strat.get(strat, s['trades'])
                pe = '🟢' if s['pnl'] >= 0 else '🔴'
                text += (
                    f"{pe} <b>{self._esc(strat)}</b>: {buys} buys · "
                    f"{s['wins']}W/{s['losses']}L · {wr:.0f}% · ${s['pnl']:+.2f}\n"
                )

        if action_counts:
            text += f"{'-'*28}\n<b>Log actions</b>: "
            text += " · ".join(f"{self._esc(k)} {v}"
                                for k, v in sorted(action_counts.items()))
            text += "\n"
        if exit_counts:
            text += "<b>Exits</b>: "
            text += " · ".join(f"{self._esc(k)} {v}" for k, v in
                                sorted(exit_counts.items(), key=lambda kv: -kv[1]))
            text += "\n"

        self.send(text)

        # Attach a tidy CSV (buys/sells/exits/profits) as the downloadable; fall
        # back to the raw JSONL if the CSV can't be written.
        if recs:
            csv_path = self._write_trades_csv(recs)
            if csv_path:
                self._send_document(
                    csv_path,
                    caption=(f"📎 Trades CSV — {len(recs)} rows "
                             f"(buys / sells / exits / profits)"),
                )
            else:
                self._send_document(
                    self._trade_log_path(),
                    caption=(f"📎 Full trade log — {len(recs)} records "
                             f"(BUY/SELL/SETTLE/REDEEM/exits)"),
                )
        else:
            self.send("ℹ️ No trade-log records yet — the log is empty.")

    # ==============================================================
    # MANUAL CLOSE (/close) — list open positions with a Sell button
    # ==============================================================

    def send_close_menu(self, edit_message_id: int = None):
        """List open positions, each with a Sell button that closes it at the
        current price (manual exit)."""
        if not self.pm:
            self.send("⚠️ Position manager not wired.")
            return
        open_pos = self.pm.get_open_positions()
        if not open_pos:
            text = "✅ No open positions to close."
            kb = {'inline_keyboard': []}
        else:
            text = ("🧮 <b>Manual close</b> — tap a Sell button to close that "
                    "position at its current price:\n\n")
            rows = []
            for i, p in enumerate(open_pos[:30], start=1):
                pe = '🟢' if p.unrealized_pnl >= 0 else '🔴'
                name = self._esc(p.bucket_label or p.market_title)
                text += (
                    f"{i}. {pe} <b>{self._esc(p.city)}</b> {name} · "
                    f"{self._esc(p.strategy)}\n"
                    f"   ${p.entry_price:.3f}→${p.current_price:.3f} | "
                    f"{p.shares:.0f}sh | ${p.unrealized_pnl:+.2f} "
                    f"({p.roi_pct:+.0f}%)\n"
                )
                rows.append([{
                    'text': f"🔴 Sell #{i} · {p.city} ${p.unrealized_pnl:+.2f}",
                    'callback_data': f"close:{p.id}",
                }])
            kb = {'inline_keyboard': rows}
        if edit_message_id is not None:
            self._edit(edit_message_id, text, kb)
        else:
            self.send(text, reply_markup=kb)

    def _do_manual_close(self, pos_id: str, callback_id: str, message_id):
        """Sell ONE open position at its current price via the PositionManager."""
        pos = (next((p for p in self.pm.positions if p.id == pos_id), None)
               if self.pm else None)
        if not pos or pos.status != 'open':
            self._answer_callback(callback_id, 'Not open')
            self.send("⚠️ That position is no longer open.")
            return
        try:
            px = pos.current_price or pos.entry_price
            self.pm._close_position(pos, px, 'manual')
            try:
                self.pm._save_state()
            except Exception:
                pass
            self._answer_callback(callback_id, 'Sold')
            self.send(
                f"🔴 <b>SOLD (manual)</b> — {self._esc(pos.strategy)}\n"
                f"📍 {self._esc(pos.city)} | "
                f"{self._esc(pos.bucket_label or pos.market_title)}\n"
                f"💵 entry ${pos.entry_price:.4f} → exit ${px:.4f} | "
                f"{pos.shares:.0f}sh\n"
                f"📊 PnL ${pos.pnl:+.2f} ({pos.roi_pct:+.0f}%)"
            )
            self.send_close_menu(edit_message_id=message_id)
        except Exception as e:
            log.debug(f"manual close failed: {e}")
            self._answer_callback(callback_id, 'Failed')
            self.send("⚠️ Manual close failed — see logs.")

    # ==============================================================
    # /done — Closed history + Open positions
    # ==============================================================

    _DONE_PAGE = 8

    def send_done_menu(self, edit_message_id: int = None):
        kb = {'inline_keyboard': [[
            {'text': '📕 Closed', 'callback_data': 'done:closed:0'},
            {'text': '📗 Open', 'callback_data': 'done:open:0'},
        ]]}
        text = (
            "🗂 <b>Positions</b>\n"
            "📕 <b>Closed</b> — all exits / loss / settle / redeem (history)\n"
            "📗 <b>Open</b> — current positions (🟢 profit / 🔴 losing)"
        )
        if edit_message_id is not None:
            self._edit(edit_message_id, text, kb)
        else:
            self.send(text, reply_markup=kb)

    @staticmethod
    def _close_label(p) -> str:
        """Human-readable description of HOW a closed position ended."""
        st = getattr(p, 'status', '')
        reason = getattr(p, 'exit_reason', '') or ''
        if st == 'redeemed':
            return '💰 redeemed'
        if st == 'won':
            return '✅ won (settled)'
        if st == 'lost':
            return '❌ lost (settled)'
        return {
            'take_profit': '🎯 take-profit',
            'stop_loss': '🛑 stop-loss',
            'trailing_stop': '📉 trailing-stop',
            'flip_timeout': '⏲️ flip book/cut',
            'thesis_invalidated': '🚫 thesis-exit',
            'manual': '🔴 manual sell',
        }.get(reason, '🔴 sold')

    def _done_closed_view(self, page: int = 0):
        """Build (text, keyboard) for a page of CLOSED positions — when bought,
        when/how closed, and the per-symbol profit/loss."""
        closed = ([p for p in self.pm.positions if p.status != 'open']
                  if self.pm else [])
        closed.sort(key=lambda p: getattr(p, 'exit_time', None) or p.entry_time,
                    reverse=True)
        total = len(closed)
        pages = max(1, (total + self._DONE_PAGE - 1) // self._DONE_PAGE)
        page = max(0, min(page, pages - 1))
        chunk = closed[page * self._DONE_PAGE:(page + 1) * self._DONE_PAGE]

        wins = sum(1 for p in closed if self.pm._closed_outcome(p) == 'win')
        losses = sum(1 for p in closed if self.pm._closed_outcome(p) == 'loss')
        realized = sum((p.pnl or 0.0) for p in closed)
        text = (f"📕 <b>Closed positions</b> ({total}) — "
                f"{wins}W/{losses}L | realized ${realized:+.2f}\n\n")
        if not chunk:
            text += "No closed positions yet.\n"
        for p in chunk:
            val = p.pnl or 0.0
            pe = '✅' if val > 0 else ('❌' if val < 0 else '➖')
            bought = p.entry_time.strftime('%m-%d %H:%M') if p.entry_time else '?'
            closed_at = (p.exit_time.strftime('%m-%d %H:%M')
                         if getattr(p, 'exit_time', None) else '?')
            exit_px = p.exit_price if p.exit_price is not None else p.current_price
            name = self._esc(p.bucket_label or p.market_title)
            text += (
                f"{pe} <b>{self._esc(p.city)}</b> {name} · {self._esc(p.strategy)}\n"
                f"   {self._close_label(p)} | ${val:+.2f} ({p.roi_pct:+.0f}%)\n"
                f"   bought {bought} @ ${p.entry_price:.3f} → "
                f"closed {closed_at} @ ${exit_px:.3f} | {p.shares:.0f}sh\n"
            )
        nav = []
        if page > 0:
            nav.append({'text': '⬅️ Prev',
                        'callback_data': f"done:closed:{page-1}"})
        nav.append({'text': f"{page+1}/{pages}", 'callback_data': 'noop'})
        if page < pages - 1:
            nav.append({'text': 'Next ➡️',
                        'callback_data': f"done:closed:{page+1}"})
        rows = [nav, [{'text': '📗 Open positions',
                       'callback_data': 'done:open:0'}]]
        return text, {'inline_keyboard': rows}

    # ==============================================================
    # /aisummary — captured runtime warnings/errors
    # ==============================================================

    def send_ai_summary(self):
        """Dump recent WARNING+ runtime log lines captured since startup so you
        can copy them to share. Healthy = nothing captured."""
        lines = list(self._error_log)
        if not lines:
            self.send("✅ <b>AI summary</b> — no warnings or errors captured "
                      "since startup. Bot looks healthy. 🟢")
            return
        errs = sum(1 for l in lines if ' ERROR' in l or ' CRITICAL' in l)
        warns = sum(1 for l in lines if ' WARNING' in l)
        tail = lines[-40:]
        head = (f"🩺 <b>AI summary — runtime issues</b>\n"
                f"Captured {errs} error(s), {warns} warning(s); showing last "
                f"{len(tail)}.\n{'-'*28}\n")
        body = "\n".join(self._esc(l) for l in tail)
        msg = head + f"<code>{body}</code>"
        while len(msg) > 3900 and len(tail) > 5:
            tail = tail[len(tail) // 2:]
            body = "\n".join(self._esc(l) for l in tail)
            msg = head + f"<code>{body}</code>"
        self.send(msg)

    # ==============================================================
    # /mlanalysis — ML (or heuristic) report on all trades
    # ==============================================================

    def send_ml_analysis(self):
        """A report of how trading is going, what's failing, what's observed and
        what to improve. Uses the ML engine for the narrative when it's enabled
        (ML_API_KEY set); otherwise falls back to a heuristic summary."""
        if not self.pm:
            self.send("⚠️ ML analysis unavailable — position manager not wired.")
            return
        stats = self.pm.get_stats()
        by_strat = self.pm.get_per_strategy_stats()
        by_city = (self.pm.get_per_city_stats()
                   if hasattr(self.pm, 'get_per_city_stats') else {})
        ranked = sorted(by_strat.items(), key=lambda kv: kv[1]['pnl'],
                        reverse=True)
        winners = [(k, v) for k, v in ranked if v['pnl'] > 0]
        losers = [(k, v) for k, v in ranked if v['pnl'] < 0]

        text = (f"🧠 <b>ML Analysis</b> — {stats['mode']}\n"
                f"WR {stats['win_rate']:.0f}% "
                f"({stats['wins']}W/{stats['losses']}L) | "
                f"PnL ${stats['total_pnl']:+.2f} | "
                f"Trades {stats['total_trades']}\n{'-'*28}\n")
        narrative = self._ml_narrative(stats, by_strat, by_city)
        if narrative:
            text += narrative + f"\n{'-'*28}\n"
        text += "<b>What's working</b>\n"
        if winners:
            for k, v in winners[:4]:
                c = v['wins'] + v['losses']
                wr = (v['wins'] / c * 100) if c else 0
                text += f"  🟢 {self._esc(k)}: ${v['pnl']:+.2f} ({wr:.0f}% WR)\n"
        else:
            text += "  (no net-positive strategy yet)\n"
        text += "<b>What's failing</b>\n"
        if losers:
            for k, v in losers[:4]:
                c = v['wins'] + v['losses']
                wr = (v['wins'] / c * 100) if c else 0
                text += f"  🔴 {self._esc(k)}: ${v['pnl']:+.2f} ({wr:.0f}% WR)\n"
        else:
            text += "  (no net-losing strategy)\n"
        tips = self._ml_heuristic_tips(stats, ranked)
        if tips:
            text += "<b>Suggested improvements</b>\n"
            for t in tips:
                text += f"  • {self._esc(t)}\n"
        self.send(text)

    def _ml_narrative(self, stats, by_strat, by_city) -> str:
        """Ask the ML engine for a short narrative if it's wired + enabled."""
        ml = getattr(self, 'ml', None)
        if not ml or not getattr(ml, 'enabled', False):
            return ("<i>ML narrative inactive — set ML_API_KEY to let the model "
                    "write the report. Heuristic analysis below.</i>")
        if not getattr(Config, 'ML_ANALYSIS_ENABLED', True):
            return ("<i>ML Analysis is turned OFF (toggle it on in ⚙️ Settings › ML). "
                    "Heuristic analysis below.</i>")
        try:
            if hasattr(ml, 'write_trade_report'):
                return self._esc(ml.write_trade_report(stats, by_strat, by_city))
        except Exception as e:
            log.debug(f"ml narrative failed: {e}")
        return ""

    @staticmethod
    def _ml_heuristic_tips(stats, ranked) -> List[str]:
        tips = []
        closed = stats['wins'] + stats['losses']
        if closed < 20:
            tips.append("Sample still small (<20 closed) — let it run to judge "
                        "edge.")
        if stats['win_rate'] < 50 and closed >= 10:
            tips.append("Win-rate <50% — tighten entry gates on the losing "
                        "strategies.")
        worst = ranked[-1] if ranked else None
        if worst and worst[1]['pnl'] < 0:
            tips.append(f"Consider disabling or tuning '{worst[0]}' — biggest "
                        f"PnL drag.")
        if not tips:
            tips.append("No red flags from the heuristic pass.")
        return tips

    # ==============================================================
    # COMMAND HANDLER (polls for incoming commands)
    # ==============================================================

    def start_polling(self):
        """Start polling for commands in a background thread."""
        if not self.enabled:
            return
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        log.info("Telegram command polling started")

    def stop_polling(self):
        """Stop polling."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)

    def _poll_loop(self):
        """Background loop to check for incoming commands."""
        while self._running:
            try:
                self._check_updates()
            except Exception as e:
                log.debug(f"Telegram poll error: {e}")
            try:
                self.notify_redeems_recent()
            except Exception as e:
                log.debug(f"redeem announce error: {e}")
            time.sleep(3)

    def _check_updates(self):
        """Check for new Telegram messages/commands."""
        try:
            resp = self._session.get(
                f"{self.base_url}/getUpdates",
                params={'offset': self._last_update_id + 1, 'timeout': 2},
                timeout=10,
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            for update in data.get('result', []):
                self._last_update_id = update['update_id']

                cb = update.get('callback_query')
                if cb:
                    cb_chat = str(cb.get('message', {}).get('chat', {}).get('id', ''))
                    if cb_chat == self.chat_id:
                        self._handle_callback(
                            cb.get('data', ''), cb.get('id', ''),
                            cb.get('message', {}).get('message_id'),
                        )
                    continue

                msg = update.get('message', {})
                text = msg.get('text', '').strip()
                chat_id = str(msg.get('chat', {}).get('id', ''))

                if chat_id != self.chat_id:
                    continue

                self._handle_command(text)
        except Exception:
            pass

    # ==============================================================
    # SETTINGS PANEL (live tunables + tick-box toggles)
    # ==============================================================

    _SETTINGS_DEFAULT_GROUP = 'main'

    # Short button labels for the on/off toggles (fallback = the key name).
    _LABELS = {
        'TRADING_ENABLED': 'Trading',
        'LATE_OBSERVED_ENABLED': 'Late-Obs',
        'LATE_OBSERVED_NO_SIDE': 'LateObs NO',
        'QUICK_FLIP_ENABLED': 'Quick-Flip',
        'PEAK_CLUSTER_ENABLED': 'Cluster',
        'PEAKER_ENABLED': 'Peaker',
        'CONFIDENT_ENABLED': 'Confident',
        'SNIPER_ENABLED': 'Sniper',
        'SPREAD_ENABLED': 'Spread',
        'STABILITY_ENABLED': 'Stability',
        'ML_ENABLED': 'Use ML',
        'ML_DECISION_ENABLED': 'ML-Decide',
        'ML_ANALYSIS_ENABLED': 'ML Analysis',
        'ML_REVIEW_POSITIONS': 'ML Review-Pos',
        'ML_SELECT_MARKETS': 'ML Market-Pick',
        'AUTO_REDEEM_ENABLED': 'Auto-Redeem',
        'PORTFOLIO_GUARD_ENABLED': 'Port-Guard',
        'DRAWDOWN_GATE_ENABLED': 'Drawdown-Gate',
        'QUICK_FLIP_PROFIT_ONLY_EXIT': 'Flip profit-only',
        'QUICK_FLIP_USE_ML_EXIT': 'Flip ML-exit',
        'QUICK_FLIP_BOOK_OR_CUT': 'Flip book-or-cut',
        'PEAKER_PREFER_COOL': 'Prefer cool',
        'PEAKER_TRADE_DECIDED': 'Peaker decided',
        'PEAK_CLUSTER_TRADE_DECIDED': 'Cluster decided',
        'THESIS_EXIT_ENABLED': 'Thesis-exit',
        'LIQUIDITY_GUARD_ENABLED': 'LiqGuard',
        'LIQUIDITY_STRICT_BLOCK': 'LiqStrict',
        'GRADE_SIZING_ENABLED': 'GradeSize',
        'SKIP_DECIDED_MARKETS': 'SkipDecided',
    }

    @staticmethod
    def _fmt_num(v):
        """Compact number formatting for buttons/labels (ints w/o decimals)."""
        if isinstance(v, bool) or v is None:
            return str(v)
        if isinstance(v, int):
            return str(v)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if f == int(f):
            return str(int(f))
        return f"{f:g}"

    def _settings_view(self, group: str = None):
        """Build (text, inline_keyboard) for ONE settings tab/group, so the
        panel stays browsable instead of one giant +/- wall."""
        from bot import settings_store
        bools, nums = settings_store.snapshot()
        groups = settings_store.GROUPS
        gid = group or self._SETTINGS_DEFAULT_GROUP
        g = next((x for x in groups if x['id'] == gid), groups[0])
        gid = g['id']
        bkeys, nkeys = settings_store.group_keys(gid)
        skeys = settings_store.group_str_keys(gid)
        strs = settings_store.str_snapshot()

        mode = '📋 PAPER' if Config.is_paper() else '🔴 LIVE'
        master = '🟢 ON' if bools.get('TRADING_ENABLED') else '🔴 OFF'
        text = (
            f"⚙️ <b>Bot Settings</b> · {mode}\n"
            f"Master trading: <b>{master}</b>\n"
            f"{'-'*30}\n"
            f"📂 <b>{self._esc(g['title'])}</b>\n"
        )
        if bkeys:
            text += "\n<b>Toggles</b>\n"
            for k in bkeys:
                text += f"  {'✅' if bools.get(k) else '❌'} {self._esc(self._LABELS.get(k, k))}\n"
        if nkeys:
            text += "\n<b>Gates</b>\n"
            for k in nkeys:
                text += f"  • {self._esc(k)} = <b>{self._fmt_num(nums.get(k))}</b>\n"
        if skeys:
            text += "\n<b>Models / Choices</b>\n"
            for k in skeys:
                text += f"  • {self._esc(k)} = <b>{self._esc(str(strs.get(k)))}</b>\n"
        text += "\n<i>Or type /set KEY VALUE · /toggle KEY</i>"

        rows = []
        # Tab row(s): 3 per row, the active tab marked with a dot.
        tab_row = []
        for x in groups:
            label = ('• ' if x['id'] == gid else '') + x['tab']
            tab_row.append({'text': label, 'callback_data': f"st:{x['id']}"})
            if len(tab_row) == 3:
                rows.append(tab_row)
                tab_row = []
        if tab_row:
            rows.append(tab_row)
        # Toggle buttons: 2 per row.
        for i in range(0, len(bkeys), 2):
            row = []
            for k in bkeys[i:i + 2]:
                on = bools.get(k)
                row.append({'text': f"{'✅' if on else '❌'} {self._LABELS.get(k, k)}",
                            'callback_data': f"tg:{k}:{gid}"})
            rows.append(row)
        # Numeric gates: one row each [➖ step][KEY = val][➕ step].
        for k in nkeys:
            step = settings_store.NUM_KEYS[k][2]
            v = nums.get(k)
            rows.append([
                {'text': f"➖{self._fmt_num(step)}", 'callback_data': f"dn:{k}:{gid}"},
                {'text': f"{k} = {self._fmt_num(v)}", 'callback_data': 'noop'},
                {'text': f"➕{self._fmt_num(step)}", 'callback_data': f"up:{k}:{gid}"},
            ])
        # String/choice settings (e.g. ML model): tap to cycle to the next value.
        for k in skeys:
            rows.append([
                {'text': f"🔁 {self._LABELS.get(k, k)}: {strs.get(k)}", 'callback_data': f"cy:{k}:{gid}"},
            ])
        # Req-29: type-to-change starting balance + an OK/Apply button that
        # summarises changes and offers Start. Shown on every tab.
        bal_now = self.pm.get_balance() if self.pm else 0.0
        rows.append([
            {'text': f"💰 Set Starting Balance (now ${self._fmt_num(bal_now)})",
             'callback_data': 'act:setbal'},
        ])
        rows.append([
            {'text': '✅ OK / Apply changes', 'callback_data': 'act:settings_ok'},
        ])
        return text, {'inline_keyboard': rows}

    def send_settings(self, group: str = None, edit_message_id: int = None):
        text, kb = self._settings_view(group)
        if edit_message_id is not None:
            self._edit(edit_message_id, text, kb)
        else:
            self.send(text, reply_markup=kb)

    def _handle_callback(self, data: str, callback_id: str, message_id):
        from bot import settings_store
        if not data or data == 'noop':
            self._answer_callback(callback_id)
            return

        # Positions pager/sorter: "pos:<page>:<sort>:<with_summary>"
        if data.startswith('pos:'):
            try:
                _, page_s, sort_key, sm = data.split(':')
                page = int(page_s)
            except (ValueError, IndexError):
                self._answer_callback(callback_id)
                return
            self._answer_callback(callback_id)
            self.send_positions(page=page, sort=sort_key,
                                with_summary=(sm == '1'),
                                edit_message_id=message_id)
            return

        # Manual close: "close:<position_id>"
        if data.startswith('close:'):
            self._do_manual_close(data.split(':', 1)[1], callback_id, message_id)
            return

        # /done sub-views: "done:closed:<page>" | "done:open:<page>"
        if data.startswith('done:'):
            try:
                _, which, pg_s = data.split(':')
                pg = int(pg_s)
            except (ValueError, IndexError):
                self._answer_callback(callback_id)
                return
            self._answer_callback(callback_id)
            if which == 'closed':
                d_text, d_kb = self._done_closed_view(pg)
                self._edit(message_id, d_text, d_kb)
            else:
                self.send_positions(page=pg, sort='pnl', with_summary=True,
                                    edit_message_id=message_id)
            return

        # Settings tab switch: "st:<group_id>"
        if data.startswith('st:'):
            group = data.split(':', 1)[1]
            self._answer_callback(callback_id)
            self.send_settings(group=group, edit_message_id=message_id)
            return

        # Lifecycle action buttons: "act:start|settings|restart"
        if data.startswith('act:'):
            action = data.split(':', 1)[1]
            if action == 'start':
                from bot import settings_store
                settings_store.set_value('TRADING_ENABLED', True)
                self._restart_pending = False
                self._answer_callback(callback_id, 'Trading enabled')
                self.send(self._start_message())
            elif action == 'settings':
                self._restart_pending = False
                self._answer_callback(callback_id)
                self.send_settings(edit_message_id=message_id)
            elif action == 'setbal':
                self._awaiting = 'balance'
                self._answer_callback(callback_id, 'Type the new balance')
                bal_now = self.pm.get_balance() if self.pm else 0.0
                self.send(
                    f"💰 <b>Set starting balance</b>\n"
                    f"Current: <b>${bal_now:.2f}</b>\n\n"
                    f"Type the new amount as a number (e.g. <code>500</code>)."
                )
            elif action == 'settings_ok':
                self._answer_callback(callback_id, 'Applying')
                self._finish_settings()
            elif action == 'restart':
                self._answer_callback(callback_id)
                self._prompt_restart()
            elif action == 'restart_confirm':
                self._answer_callback(callback_id, 'Restarting fresh')
                self._do_restart()
            elif action == 'restart_cancel':
                self._restart_pending = False
                self._answer_callback(callback_id, 'Cancelled')
                self.send("✖️ Restart cancelled — positions untouched.")
            else:
                self._answer_callback(callback_id)
            return

        # Toggle / bump: "<action>:<KEY>[:<group>]"
        parts = data.split(':')
        action = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        group = parts[2] if len(parts) > 2 else None
        if not key:
            self._answer_callback(callback_id)
            return
        ok, msg = False, 'no change'
        if action == 'tg':
            ok, msg = settings_store.toggle(key)
        elif action == 'cy':
            ok, msg = settings_store.cycle(key)
        elif action == 'up':
            ok, msg = settings_store.bump(key, +1)
        elif action == 'dn':
            ok, msg = settings_store.bump(key, -1)
        if ok:
            self._note_change(msg or key)
        self._answer_callback(callback_id, msg)
        if ok and message_id is not None:
            self.send_settings(group=group, edit_message_id=message_id)

    # ----- Req-29 settings / balance UX helpers -----------------------------
    def _note_change(self, msg: str):
        """Record a human-readable settings change for the OK summary."""
        try:
            if msg and msg not in self._session_changes:
                self._session_changes.append(msg)
                self._session_changes = self._session_changes[-40:]
        except Exception:
            pass

    def _consume_awaited_input(self, text: str):
        """Handle a typed value we were waiting for (currently: balance)."""
        from bot import settings_store
        what = self._awaiting
        self._awaiting = None
        if what == 'balance':
            raw = text.strip().lstrip('$').replace(',', '')
            try:
                val = float(raw)
            except ValueError:
                self._awaiting = 'balance'
                self.send("⚠️ That doesn't look like a number. Type e.g. <code>500</code>.")
                return
            ok, msg = settings_store.set_value('STARTING_BALANCE', val)
            if ok:
                self._note_change(msg or f"STARTING_BALANCE = {val:g}")
            self.send(
                ("✅ " if ok else "⚠️ ") + msg + "\n\n"
                "Tap <b>OK / Apply changes</b> when you're done, or change more first.",
                reply_markup={'inline_keyboard': [[
                    {'text': '✅ OK / Apply changes', 'callback_data': 'act:settings_ok'},
                    {'text': '⚙️ Settings', 'callback_data': 'act:settings'},
                ]]},
            )

    def _start_message(self) -> str:
        """Enable-trading confirmation. Applies the configured starting balance
        to the live paper ledger when the book is empty (fixes 'set 300 -> only
        traded 100')."""
        note = ""
        if self.pm is not None:
            try:
                res = self.pm.apply_starting_balance()
                if res.get('applied'):
                    note = f"\nStarting balance: <b>${res['balance']:.2f}</b>"
                elif res.get('reason') == 'positions_open':
                    note = (f"\n⚠️ Balance NOT changed — {res.get('open', 0)} position(s) "
                            f"still open. Tap ♻️ Restart to apply ${res['target']:.2f}.")
                elif res.get('reason') == 'has_history':
                    note = (f"\n⚠️ Balance kept at ${res['balance']:.2f} (closed-trade "
                            f"history present). Tap ♻️ Restart to start fresh at "
                            f"${res['target']:.2f}.")
            except Exception:
                pass
        return "🟢 <b>Trading ENABLED</b> — the bot will place new trades." + note

    def _finish_settings(self):
        """OK button: summarise changes, apply the balance if flat, offer Start."""
        changes = list(self._session_changes)
        self._session_changes = []
        if changes:
            body = "\n".join(f"  • {self._esc(c)}" for c in changes)
            summary = f"✅ <b>Settings changed</b>\n{body}"
        else:
            summary = "✅ <b>Settings saved</b> — no changes this session."
        bal_note = ""
        if self.pm is not None:
            try:
                res = self.pm.apply_starting_balance()
                if res.get('applied'):
                    bal_note = f"\n💰 Starting balance is now <b>${res['balance']:.2f}</b>."
                elif res.get('reason') == 'positions_open':
                    bal_note = (f"\n⚠️ {res.get('open', 0)} position(s) open — new balance "
                                f"(${res['target']:.2f}) applies after ♻️ Restart.")
                elif res.get('reason') == 'has_history':
                    bal_note = (f"\n💰 Balance ${res['balance']:.2f}. Tap ♻️ Restart to "
                                f"start fresh at ${res['target']:.2f}.")
            except Exception:
                pass
        kb = {'inline_keyboard': [[
            {'text': '▶️ Start bot now', 'callback_data': 'act:start'},
            {'text': '♻️ Restart fresh', 'callback_data': 'act:restart'},
        ]]}
        self.send(summary + bal_note + "\n\nReady — <b>settings changed, start bot now.</b>",
                  reply_markup=kb)

    def _handle_command(self, text: str):
        """Handle incoming bot commands."""
        # Req-29: capture a typed value when we're awaiting one (e.g. a new
        # starting balance). A slash-command cancels the awaiting state.
        if self._awaiting and text and not text.startswith('/'):
            self._consume_awaited_input(text)
            return
        if self._awaiting and text.startswith('/'):
            self._awaiting = None
        cmd = text.lower().split()[0] if text else ''
        parts = text.split()

        if cmd in ('/start', '/resume', 'start'):
            from bot import settings_store
            settings_store.set_value('TRADING_ENABLED', True)
            self.send(self._start_message())
        elif cmd in ('/restart', 'restart'):
            self._prompt_restart()
        elif cmd == '/stop' or cmd == '/pause':
            from bot import settings_store
            settings_store.set_value('TRADING_ENABLED', False)
            self.send("🔴 <b>Trading DISABLED</b> — monitoring & resolving only, no new buys.")
        elif cmd == '/settings' or cmd == '/config':
            grp = parts[1].lower() if len(parts) >= 2 else None
            self.send_settings(group=grp)
        elif cmd == '/set':
            from bot import settings_store
            if len(parts) >= 3:
                ok, msg = settings_store.set_value(parts[1], parts[2])
                if ok:
                    self._note_change(msg or f"{parts[1]} = {parts[2]}")
                self.send(("✅ " if ok else "⚠️ ") + msg)
            else:
                self.send("Usage: <code>/set KEY VALUE</code>  e.g. <code>/set BASKET_MAX_COST 0.80</code>")
        elif cmd == '/toggle':
            from bot import settings_store
            if len(parts) >= 2:
                ok, msg = settings_store.toggle(parts[1])
                self.send(("✅ " if ok else "⚠️ ") + msg)
            else:
                self.send("Usage: <code>/toggle KEY</code>  e.g. <code>/toggle SNIPER_ENABLED</code>")
        elif cmd == '/status' or cmd == '/stats':
            self.send_status()
        elif cmd == '/balance' or cmd == '/bal':
            bal = self.pm.get_balance() if self.pm else 0
            self.send(f"💰 Balance: ${bal:.2f}")
        elif cmd == '/pnl':
            pnl = self.pm.get_total_pnl() if self.pm else 0
            self.send(f"📊 Total PnL: ${pnl:+.2f}")
        elif cmd == '/positions' or cmd == '/pos':
            self.send_positions(page=0, sort='recent', with_summary=False)
        elif cmd == '/markets':
            self.send_markets_summary()
        elif cmd == '/analysis' or cmd == '/analyze' or cmd == '/report':
            self.send_analysis()
        elif cmd in ('/close', '/sell'):
            self.send_close_menu()
        elif cmd in ('/done', '/history'):
            self.send_done_menu()
        elif cmd in ('/aisummary', '/errors', '/ai'):
            self.send_ai_summary()
        elif cmd in ('/mlanalysis', '/ml', '/mlreport'):
            self.send_ml_analysis()
        elif cmd == '/redeem':
            if self.pm:
                count = self.pm.redeem_all_winning()
                # redeem_all_winning may return a count (int) or a list.
                n = len(count) if isinstance(count, list) else count
                self.notify_redeems_recent()
                self.send(f"💰 Redeemed {n} positions")
        elif cmd in ('/reserve', '/res'):
            from bot import settings_store
            if len(parts) >= 2:
                ok, msg = settings_store.set_value('TAKEOUT_RESERVE_USD', parts[1])
                self.send(("✅ " if ok else "⚠️ ") + msg)
            else:
                try:
                    from overlay import reserve_takeout as _rt
                    self.send(_rt.status(self.pm))
                except Exception as e:
                    self.send(f"⚠️ reserve unavailable: {e}")
        elif cmd in ('/takeout', '/take'):
            arg = parts[1].lower() if len(parts) >= 2 else ''
            try:
                from overlay import reserve_takeout as _rt
            except Exception as e:
                self.send(f"⚠️ takeout unavailable: {e}")
                return
            if arg in ('withdraw', 'out', 'cash', 'w'):
                try:
                    ok, msg = _rt.withdraw(self.pm)
                    self.send(("✅ " if ok else "⚠️ ") + msg)
                except Exception as e:
                    self.send(f"⚠️ withdraw failed: {e}")
            elif arg:
                from bot import settings_store
                ok, msg = settings_store.set_value('TAKEOUT_TARGET_USD', parts[1])
                self.send(("✅ " if ok else "⚠️ ") + msg)
            else:
                self.send(_rt.status(self.pm))
        elif cmd == '/info':
            from bot import settings_store
            info = getattr(settings_store, 'INFO', {}) or {}
            if len(parts) >= 2:
                key = parts[1].upper()
                txt = info.get(key)
                self.send(f"ℹ️ <b>{key}</b>\n{txt}" if txt else f"No info for <code>{key}</code>. Try /info with no key to list.")
            else:
                keys = ', '.join(sorted(info.keys()))
                self.send(f"ℹ️ <b>{len(info)} documented settings</b> — use <code>/info KEY</code>:\n{keys}")
        elif cmd == '/help':
            self.send(
                "🌤️ <b>Weather Sniper Commands</b>\n"
                "<b>/start</b> — enable trading (or just type 'start')\n"
                "<b>/restart</b> — clear ALL positions & start fresh (or type 'restart')\n"
                "<b>/stop</b> — disable trading (monitor only)\n"
                "<b>/settings</b> — tabbed panel: toggle strategies & tune every gate\n"
                "   (e.g. <code>/settings peaker</code> opens that tab)\n"
                "/set KEY VALUE — set a gate, e.g. /set BASKET_MAX_COST 0.80\n"
                "/toggle KEY — flip a toggle, e.g. /toggle SNIPER_ENABLED\n"
                "/status — summary + positions (paged, sortable)\n"
                "/balance — current balance\n"
                "/pnl — total profit/loss\n"
                "/positions — open positions (10/page; sort by PnL/Losses/ROI/Recent)\n"
                "/markets — active weather markets\n"
                "/analysis — per-strategy performance + downloadable CSV\n"
                "/close — manually sell an open position (tap Sell)\n"
                "/done — closed history + open positions (🟢/🔴)\n"
                "/aisummary — recent runtime warnings/errors to share\n"
                "/mlanalysis — ML report: how it's going, what's failing\n"
                "/redeem — redeem winning positions\n"
                "/reserve [USD] — view/set untouchable cash reserve\n"
                "/takeout [USD|withdraw] — set win-skim target / withdraw the pool\n"
                "/info KEY — explain any setting (effect + range)\n"
                "/help — this message"
            )
        elif cmd.startswith('/'):
            self.send(f"❓ Unknown command. Try /help")
