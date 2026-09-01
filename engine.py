"""
================================================================================
 ⚡ PRO Quant Options Signal Engine — CORE ENGINE (Production Grade)
================================================================================
engine.py contains ALL trading logic and NO Streamlit / UI code:
    - Structured (JSON) logging
    - SQLite trade ledger
    - Thread-safe shared state (EngineState)
    - Risk management layer (kill switch, max trades, cooldown, sizing)
    - Expiry safety filter
    - Entry-zone / limit-style execution with slippage simulation
    - Choppy-market / overtrading protection
    - Broker execution abstraction (paper + live hook)
    - Fyers REST + WebSocket integration with TTL caching

app.py (Streamlit) imports this module and never touches global state
directly — it only calls functions/methods exposed here. This means the
engine can also be driven headlessly (e.g. from a cron job, a CLI script,
or a future live-trading daemon) without any Streamlit dependency at all.
================================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from scipy.stats import norm

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

# ==============================================================================
# 1. STRUCTURED LOGGING  (requirement #7)
# ==============================================================================
IST_TZ = pytz.timezone("Asia/Kolkata")
LOG_JSON_PATH = "engine_events.jsonl"


class JsonFormatter(logging.Formatter):
    """Emits one JSON object per line: timestamp, level, event, reason, message + extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "level": record.levelname,
            "event": getattr(record, "event", "generic"),
            "message": record.getMessage(),
        }
        extra_data = getattr(record, "extra_data", None)
        if extra_data:
            for k, v in extra_data.items():
                # keep payload JSON-serializable
                try:
                    json.dumps(v)
                    payload[k] = v
                except TypeError:
                    payload[k] = str(v)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


logger = logging.getLogger("quant_scalper_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_console)

    _json_file = logging.FileHandler(LOG_JSON_PATH)
    _json_file.setFormatter(JsonFormatter())
    logger.addHandler(_json_file)


def log_event(event: str, message: str, level: int = logging.INFO, **extra) -> None:
    logger.log(level, message, extra={"event": event, "extra_data": extra})


def log_signal(index_symbol: str, signal: dict) -> None:
    log_event(
        "signal",
        f"[{index_symbol}] Signal={signal.get('signal')} Conf={signal.get('conf')}% :: {signal.get('reason')}",
        index_symbol=index_symbol,
        **{k: v for k, v in signal.items() if k != "reason"},
        reason=signal.get("reason"),
    )


def log_trade(action: str, trade: dict, reason: str = "") -> None:
    log_event(
        "trade",
        f"{action} :: {trade.get('index_symbol')} {trade.get('strike')}{trade.get('type')} "
        f"status={trade.get('status')} pnl={trade.get('pnl')}",
        action=action,
        reason=reason,
        db_id=trade.get("db_id"),
        symbol=trade.get("symbol"),
        status=trade.get("status"),
        pnl=trade.get("pnl"),
    )


def log_error(context: str, exc: Exception) -> None:
    log_event("error", f"{context}: {exc}", level=logging.ERROR, context=context, error=str(exc))


# ==============================================================================
# 2. STATIC CONFIGURATION
# ==============================================================================
DB_PATH = "quant_paper_trades_live.db"
JSON_LOG_PATH = "closed_paper_trades_live_log.json"


@dataclass(frozen=True)
class IndexConfig:
    label: str
    symbol: str
    exchange_segment: str
    strike_step: int
    lot_size: int


INDEX_CONFIG: Dict[str, IndexConfig] = {
    "NIFTY50": IndexConfig("NIFTY 50", "NSE:NIFTY50-INDEX", "NSE_FNO", 50, 65),
    "BANKNIFTY": IndexConfig("NIFTY BANK", "NSE:NIFTYBANK-INDEX", "NSE_FNO", 100, 30),
    "SENSEX": IndexConfig("BSE SENSEX", "BSE:SENSEX-INDEX", "BSE_FNO", 100, 20),
}

STRIKES_AROUND_ATM = 4


@dataclass
class RiskConfig:
    """All risk / execution knobs in one place. Passed explicitly into every
    engine call — no hidden globals — so the same engine can safely run
    multiple configs (e.g. in tests) without cross-talk."""

    capital: float = 100_000.0
    risk_pct: float = 2.0                     # % of capital risked per trade
    max_capital_per_trade_pct: float = 25.0    # hard cap on capital deployed per trade
    daily_loss_limit_pct: float = 5.0          # kill switch threshold
    max_trades_per_day: int = 10
    cooldown_seconds: int = 120                # 2 min default cooldown between trades
    min_minutes_to_expiry: float = 30.0        # expiry safety filter (requirement #3)
    slippage_pct: float = 0.15                 # realistic fill slippage, 0.10%-0.30% band
    entry_zone_pct: float = 0.15               # +/- zone around reference entry price
    entry_timeout_seconds: int = 90            # auto-cancel stale pending "limit" orders
    choppy_filter_enabled: bool = True


# ==============================================================================
# 3. BLACK-SCHOLES IV & GREEKS ENGINE (kept, + memoized for performance)
# ==============================================================================
def get_dte_years(expiry_str: str) -> float:
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        exp_datetime = IST_TZ.localize(exp_date.replace(hour=15, minute=30, second=0, microsecond=0))
        now = datetime.now(IST_TZ)
        time_diff = exp_datetime - now

        if time_diff.total_seconds() <= 0:
            return 0.0

        T = time_diff.total_seconds() / (365.0 * 24.0 * 3600.0)
        min_years = 1.0 / (365.0 * 24.0 * 60.0)  # 1-minute floor for 0 DTE
        return max(T, min_years)
    except Exception:
        return 0.02


def check_expiry_safety(expiry_str: str, min_minutes: float = 30.0) -> Tuple[bool, str]:
    """Requirement #3 — Expiry safety filter. Blocks entries too close to expiry."""
    t_years = get_dte_years(expiry_str)
    minutes_left = t_years * 365.0 * 24.0 * 60.0
    if minutes_left < min_minutes:
        return False, (
            f"NO TRADE — Expiry safety filter: only {minutes_left:.1f} min left to "
            f"expiry {expiry_str} (< {min_minutes:.0f} min threshold)."
        )
    return True, f"OK — {minutes_left:.1f} min to expiry."


def compute_live_greeks(S: float, K: float, T: float, market_price: float, opt_type: str, r: float = 0.07) -> dict:
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return {"IV": 0.0, "Delta": 0.0, "Gamma": 0.0, "Theta": 0.0}

    sigma = 0.20
    for _ in range(20):
        sigma = max(0.01, min(sigma, 5.0))
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if opt_type == "CE":
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            diff = price - market_price
            if abs(diff) < 0.05:
                break

            vega_val = S * norm.pdf(d1) * np.sqrt(T)
            if vega_val == 0:
                break
            sigma -= diff / vega_val
        except Exception:
            break

    sigma = max(0.01, min(sigma, 5.0))

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        n_d1 = norm.pdf(d1)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)

        gamma = n_d1 / (S * sigma * np.sqrt(T))

        if opt_type == "CE":
            delta = N_d1
            theta = (-(S * sigma * n_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365.0
        else:
            delta = N_d1 - 1.0
            theta = (-(S * sigma * n_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    except Exception:
        delta, gamma, theta = 0.0, 0.0, 0.0

    return {
        "IV": round(sigma * 100, 2),
        "Delta": round(delta, 3),
        "Gamma": round(gamma, 5),
        "Theta": round(theta, 2),
    }


@lru_cache(maxsize=4096)
def _compute_greeks_cached(S_r: float, K: float, T_r: float, price_r: float, opt_type: str) -> Tuple[float, float, float, float]:
    g = compute_live_greeks(S_r, K, T_r, price_r, opt_type)
    return g["IV"], g["Delta"], g["Gamma"], g["Theta"]


def compute_live_greeks_fast(S: float, K: float, T: float, market_price: float, opt_type: str) -> dict:
    """Requirement #5 — avoid unnecessary recalculation of the IV solver
    (20 Newton-Raphson iterations each call) by memoizing on rounded inputs.
    Spot/price/time rarely change meaningfully tick-to-tick, so this collapses
    thousands of redundant solves per minute into a handful of cache hits."""
    IV, Delta, Gamma, Theta = _compute_greeks_cached(round(S, 1), K, round(T, 6), round(market_price, 2), opt_type)
    return {"IV": IV, "Delta": Delta, "Gamma": Gamma, "Theta": Theta}


# ==============================================================================
# 4. LIGHTWEIGHT TTL CACHE  (requirement #5 — replaces st.cache_data so the
#    engine has zero Streamlit dependency and caching also works headlessly)
# ==============================================================================
class TTLCache:
    def __init__(self, ttl_seconds: float = 5.0):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[object, float]] = {}
        self._lock = threading.Lock()

    def get_or_set(self, key: str, factory: Callable[[], object]) -> object:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is not None and (now - entry[1]) < self.ttl:
                return entry[0]
        # compute outside the lock so slow network calls don't block other keys
        value = factory()
        with self._lock:
            self._store[key] = (value, time.time())
        return value


_option_chain_cache = TTLCache(ttl_seconds=5.0)
_candles_cache = TTLCache(ttl_seconds=5.0)


# ==============================================================================
# 5. DATABASE LAYER (thread-safe via module lock; schema auto-migrates)
# ==============================================================================
DB_LOCK = threading.Lock()


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def init_db() -> None:
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, index_symbol TEXT, signal TEXT, strike FLOAT,
                opt_type TEXT, symbol TEXT, entry FLOAT, exit_price FLOAT, sl FLOAT, target_1 FLOAT,
                target_2 FLOAT, lots INTEGER, status TEXT, pnl FLOAT, is_live INTEGER
            )
            """
        )
        # Non-destructive migration for new production-grade fields.
        for col, col_type in [
            ("entry_zone_low", "FLOAT"), ("entry_zone_high", "FLOAT"),
            ("entry_fill", "FLOAT"), ("slippage_pct", "FLOAT"),
            ("exit_reason", "TEXT"),
        ]:
            try:
                _ensure_column(conn, "trades", col, col_type)
            except Exception as e:
                log_error("db_migration", e)
        conn.commit()
        conn.close()


def db_save_trade(trade: dict) -> int:
    init_db()
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (timestamp, index_symbol, signal, strike, opt_type, symbol, entry, exit_price,
                                 sl, target_1, target_2, lots, status, pnl, is_live,
                                 entry_zone_low, entry_zone_high, entry_fill, slippage_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.get("timestamp", datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S")),
                trade["index_symbol"], trade["signal"], trade["strike"], trade["type"],
                trade["symbol"], trade["entry"], trade.get("exit_price", None), trade["sl"],
                trade["target_1"], trade["target_2"], trade["lots"], trade["status"], trade["pnl"], 0,
                trade.get("entry_zone_low"), trade.get("entry_zone_high"),
                trade.get("entry_fill"), trade.get("slippage_pct"),
            ),
        )
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id


def db_update_trade(trade_id: Optional[int], status: str, pnl: float, exit_price: Optional[float] = None,
                     sl: Optional[float] = None, entry_fill: Optional[float] = None,
                     exit_reason: Optional[str] = None) -> None:
    if not trade_id:
        return
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        sets, params = ["status = ?", "pnl = ?", "exit_price = ?"], [status, pnl, exit_price]
        if sl is not None:
            sets.append("sl = ?"); params.append(sl)
        if entry_fill is not None:
            sets.append("entry_fill = ?"); params.append(entry_fill)
        if exit_reason is not None:
            sets.append("exit_reason = ?"); params.append(exit_reason)
        params.append(trade_id)
        cursor.execute(f"UPDATE trades SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        conn.close()


def log_trade_to_json(trade: dict, exit_price: float) -> None:
    if trade.get("json_logged"):
        return
    log_entry = {
        "timestamp": datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": trade["index_symbol"], "strike": trade["strike"], "side": trade["type"],
        "entry": round(trade.get("entry_fill") or trade["entry"], 2),
        "exit": round(exit_price, 2) if exit_price else 0.0,
        "pnl": round(trade.get("pnl", 0.0), 2), "status": trade.get("status", "CLOSED"),
    }
    try:
        data = []
        if os.path.exists(JSON_LOG_PATH):
            with open(JSON_LOG_PATH, "r") as f:
                data = json.load(f)
        data.append(log_entry)
        with open(JSON_LOG_PATH, "w") as f:
            json.dump(data, f, indent=4)
        trade["json_logged"] = True
    except Exception as e:
        log_error("json_trade_log", e)


def db_load_today_trades() -> Tuple[pd.DataFrame, float]:
    init_db()
    today_str = datetime.now(IST_TZ).strftime("%Y-%m-%d")
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        query = """
        SELECT timestamp as Time, index_symbol as "Index", signal as Signal,
               strike || ' ' || opt_type as Strike, entry as Entry,
               COALESCE(entry_fill, 0.0) as "Fill Price",
               COALESCE(exit_price, 0.0) as "Exit Price", status as Status, pnl as PnL
        FROM trades WHERE timestamp LIKE ?
        ORDER BY id DESC
        """
        df = pd.read_sql_query(query, conn, params=(f"{today_str}%",))
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(pnl) FROM trades WHERE timestamp LIKE ?", (f"{today_str}%",))
        row = cursor.fetchone()
        daily_pnl = row[0] if row and row[0] is not None else 0.0
        conn.close()
    return df, daily_pnl


# ==============================================================================
# 6. THREAD-SAFE ENGINE STATE  (requirement #4)
# ==============================================================================
class EngineState:
    """
    Centralized, thread-safe container for ALL mutable engine state.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.tick_cache: Dict[str, dict] = {}
        self.micro_candles: Dict[str, deque] = {}
        self.active_trades: Dict[str, dict] = {}
        self.daily_pnl: float = 0.0
        self.daily_trades_count: int = 0
        self.last_trade_time: float = 0.0
        self.ws_connected: bool = False
        self.last_closed_message: Optional[str] = None
        self.ws_worker = None
        self.ws_queue: "Queue" = Queue()

    # ---------------------------------------------------------------- ticks
    def update_tick(self, symbol: str, ltp: float, bid_qty: float = 0, ask_qty: float = 0) -> None:
        with self._lock:
            self.tick_cache[symbol] = {
                "ltp": round(ltp, 2), "ts": time.time(), "bid_qty": bid_qty, "ask_qty": ask_qty,
            }
            candles = self.micro_candles.setdefault(symbol, deque(maxlen=600))
            candles.append((time.time(), ltp))

    def get_tick(self, symbol: str) -> Optional[dict]:
        with self._lock:
            data = self.tick_cache.get(symbol)
            return dict(data) if data else None

    def get_micro_candles(self, symbol: str) -> List[float]:
        with self._lock:
            return list(self.micro_candles.get(symbol, []))

    # ------------------------------------------------------------- bootstrap
    def hydrate_daily(self, pnl: float, trades_count: int) -> None:
        with self._lock:
            self.daily_pnl = pnl
            self.daily_trades_count = trades_count

    # ------------------------------------------------------------ risk gates
    def _evaluate_risk_gates_locked(self, cfg: RiskConfig) -> Tuple[bool, str]:
        max_loss_amount = cfg.capital * (cfg.daily_loss_limit_pct / 100.0)
        if self.daily_pnl <= -max_loss_amount:
            return False, (
                f"🛑 KILL SWITCH — daily loss limit ₹{max_loss_amount:,.0f} breached "
                f"(current PnL ₹{self.daily_pnl:,.0f}). No further trades today."
            )
        if self.daily_trades_count >= cfg.max_trades_per_day:
            return False, f"🚫 Max daily trades ({cfg.max_trades_per_day}) reached."
        elapsed = time.time() - self.last_trade_time
        if elapsed < cfg.cooldown_seconds:
            return False, f"⏳ Cooldown active — {cfg.cooldown_seconds - elapsed:.0f}s remaining before next trade."
        return True, "OK"

    def can_trade(self, cfg: RiskConfig) -> Tuple[bool, str]:
        with self._lock:
            return self._evaluate_risk_gates_locked(cfg)

    # ------------------------------------------------------- trade lifecycle
    def register_trade(self, index_symbol: str, trade: dict, cfg: RiskConfig) -> Tuple[bool, str]:
        with self._lock:
            if index_symbol in self.active_trades:
                return False, "🚫 Duplicate entry blocked — a trade is already active/pending for this index."
            allowed, reason = self._evaluate_risk_gates_locked(cfg)
            if not allowed:
                return False, reason
            self.active_trades[index_symbol] = trade
            self.daily_trades_count += 1
            self.last_trade_time = time.time()
            return True, "Registered"

    def get_active_trade(self, index_symbol: str) -> Optional[dict]:
        with self._lock:
            t = self.active_trades.get(index_symbol)
            return dict(t) if t else None

    def remove_trade(self, index_symbol: str) -> Optional[dict]:
        with self._lock:
            return self.active_trades.pop(index_symbol, None)

    def manual_close(self, index_symbol: str, pnl_delta: float) -> Optional[dict]:
        with self._lock:
            t = self.active_trades.pop(index_symbol, None)
            if t is not None:
                self.daily_pnl += pnl_delta
                self.last_trade_time = time.time()
            return t

    def process_tick_for_trade(self, index_symbol: str, live_ltp: float, lot_size: int,
                                cfg: RiskConfig) -> Optional[dict]:
        with self._lock:
            trade = self.active_trades.get(index_symbol)
            if trade is None:
                return None

            if trade["status"] == "WAITING":
                fill_price = try_fill_entry(trade, live_ltp, cfg.slippage_pct)
                if fill_price is not None:
                    trade["status"] = "ENTERED"
                    trade["entry_fill"] = fill_price
                    trade["pnl"] = 0.0
                    return {"type": "ENTRY_FILLED", "trade": dict(trade)}

                if time.time() - trade.get("placed_ts", time.time()) > trade.get(
                    "entry_timeout_seconds", cfg.entry_timeout_seconds
                ):
                    self.active_trades.pop(index_symbol, None)
                    trade["status"] = "CANCELLED_TIMEOUT"
                    return {"type": "ENTRY_TIMEOUT", "trade": dict(trade)}
                return None

            if trade["status"] in ("ENTERED", "PARTIAL_EXIT"):
                entry_ref = trade.get("entry_fill") or trade["entry"]
                trade["pnl"] = round((live_ltp - entry_ref) * trade["lots"] * lot_size, 2)

                if live_ltp <= trade["sl"]:
                    etype = "STOP_LOSS" if trade["status"] == "ENTERED" else "TRAILED_SL"
                    trade["status"] = "CLOSED"
                    trade["exit_price"] = live_ltp
                    self.active_trades.pop(index_symbol, None)
                    self.daily_pnl += trade["pnl"]
                    self.last_trade_time = time.time()
                    return {"type": etype, "trade": dict(trade)}

                if trade["status"] == "ENTERED" and live_ltp >= trade["target_1"]:
                    trade["status"] = "PARTIAL_EXIT"
                    trade["sl"] = entry_ref  # trail SL to breakeven
                    return {"type": "PARTIAL_EXIT", "trade": dict(trade)}

                if trade["status"] == "PARTIAL_EXIT" and live_ltp >= trade["target_2"]:
                    trade["status"] = "CLOSED"
                    trade["exit_price"] = live_ltp
                    self.active_trades.pop(index_symbol, None)
                    self.daily_pnl += trade["pnl"]
                    self.last_trade_time = time.time()
                    return {"type": "TARGET_2", "trade": dict(trade)}
            return None

    # -------------------------------------------------------------- misc UI
    def set_closed_message(self, msg: str) -> None:
        with self._lock:
            self.last_closed_message = msg

    def pop_closed_message(self) -> Optional[str]:
        with self._lock:
            msg = self.last_closed_message
            self.last_closed_message = None
            return msg


def handle_trade_event(state: EngineState, event: dict) -> None:
    trade, etype = event["trade"], event["type"]
    db_id = trade.get("db_id")
    try:
        if etype == "ENTRY_FILLED":
            db_update_trade(db_id, "ENTERED", 0.0, entry_fill=trade["entry_fill"])
            log_trade("ENTRY_FILLED", trade,
                      reason=f"Filled @ {trade['entry_fill']} within zone "
                             f"[{trade.get('entry_zone_low')}, {trade.get('entry_zone_high')}]")
        elif etype == "ENTRY_TIMEOUT":
            db_update_trade(db_id, "CANCELLED_TIMEOUT", 0.0, exit_reason="Entry zone never reached")
            log_trade("ENTRY_TIMEOUT", trade, reason="Entry zone not reached within timeout window")
            state.set_closed_message(f"⌛ Pending order for {trade['symbol']} auto-cancelled (entry zone unreached).")
        elif etype == "PARTIAL_EXIT":
            db_update_trade(db_id, "PARTIAL_EXIT", trade["pnl"], sl=trade["sl"])
            log_trade("PARTIAL_EXIT", trade, reason="Target 1 hit — SL trailed to breakeven")
        elif etype in ("STOP_LOSS", "TRAILED_SL", "TARGET_2"):
            db_update_trade(db_id, etype, trade["pnl"], exit_price=trade["exit_price"], exit_reason=etype)
            log_trade_to_json(trade, trade["exit_price"])
            log_trade(etype, trade, reason=f"Exit @ {trade['exit_price']}")
            icon = {"STOP_LOSS": "🛑", "TRAILED_SL": "🛡️", "TARGET_2": "🎯"}[etype]
            state.set_closed_message(
                f"{icon} {etype.replace('_', ' ').title()} on {trade['index_symbol']}! "
                f"Exit: ₹{trade['exit_price']:.2f} | PnL: ₹{trade['pnl']:.2f}"
            )
    except Exception as e:
        log_error("handle_trade_event", e)


def process_tick_queue(state: EngineState, cfg_provider: Callable[[], RiskConfig],
                        lot_size_lookup: Callable[[str], int]) -> None:
    while True:
        message = state.ws_queue.get()
        try:
            if not message:
                continue
            symbol = message.get("symbol")
            ltp = float(message.get("ltp", 0) or 0)
            if not symbol or ltp <= 0:
                continue

            state.update_tick(symbol, ltp, message.get("total_buy_qty", 0), message.get("total_sell_qty", 0))

            cfg = cfg_provider()
            for index_symbol, trade in list(state.active_trades.items()):
                if trade.get("symbol") != symbol:
                    continue
                lot_size = lot_size_lookup(trade["index_symbol"])
                event = state.process_tick_for_trade(index_symbol, ltp, lot_size, cfg)
                if event:
                    handle_trade_event(state, event)
        except Exception as e:
            log_error("tick_queue_processing", e)
        finally:
            state.ws_queue.task_done()


def start_tick_worker(state: EngineState, cfg_provider: Callable[[], RiskConfig],
                       lot_size_lookup: Callable[[str], int]) -> threading.Thread:
    t = threading.Thread(target=process_tick_queue, args=(state, cfg_provider, lot_size_lookup), daemon=True)
    t.start()
    return t


# ==============================================================================
# 7. ENTRY EXECUTION LOGIC
# ==============================================================================
def compute_entry_zone(reference_price: float, zone_pct: float) -> Tuple[float, float]:
    half = reference_price * (zone_pct / 100.0)
    return round(reference_price - half, 2), round(reference_price + half, 2)


def try_fill_entry(trade: dict, live_ltp: float, slippage_pct: float) -> Optional[float]:
    zone_low = trade.get("entry_zone_low")
    zone_high = trade.get("entry_zone_high")
    if zone_low is None or zone_high is None:
        zone_low, zone_high = compute_entry_zone(trade["entry"], trade.get("entry_zone_pct", 0.15))

    if zone_low <= live_ltp <= zone_high:
        fill_price = round(live_ltp * (1 + slippage_pct / 100.0), 2)
        return fill_price
    return None


# ==============================================================================
# 8. RISK MANAGEMENT LAYER
# ==============================================================================
def detect_choppy_market(prices: List[float], window: int = 60) -> bool:
    if len(prices) < window:
        return False
    arr = np.array(prices[-window:], dtype=float)
    diffs = np.diff(arr)
    if len(diffs) < 2:
        return False
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    net_move = abs(arr[-1] - arr[0])
    avg_abs_move = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0
    return sign_changes > (window * 0.4) and net_move < (avg_abs_move * 3.0)


def calculate_position_size(state: EngineState, cfg: RiskConfig, entry: float, sl: float, lot_size: int) -> dict:
    allowed, reason = state.can_trade(cfg)
    if not allowed:
        return {"allowed": False, "reason": reason, "lots": 0}

    sl_distance = abs(entry - sl)
    if sl_distance <= 0:
        return {"allowed": False, "reason": "Invalid SL distance (entry == SL).", "lots": 0}

    max_risk_amount = cfg.capital * (cfg.risk_pct / 100.0)
    lots = max(1, math.floor(max_risk_amount / (sl_distance * lot_size)))

    max_capital_alloc = cfg.capital * (cfg.max_capital_per_trade_pct / 100.0)
    if (lots * lot_size * entry) > max_capital_alloc:
        lots = math.floor(max_capital_alloc / (lot_size * entry))

    if lots <= 0:
        return {"allowed": False, "reason": "Insufficient capital for 1 lot within the safety cap.", "lots": 0}

    max_loss = round(sl_distance * lots * lot_size, 2)
    if max_loss > max_risk_amount * 1.5:
        return {"allowed": False, "reason": "Computed risk exceeds safety tolerance for configured risk%.", "lots": 0}

    return {"allowed": True, "reason": "Risk Approved", "lots": lots, "max_loss": max_loss}


def evaluate_all_entry_filters(state: EngineState, cfg: RiskConfig, expiry_str: str,
                                micro_candles: List[float]) -> Tuple[bool, str]:
    ok, reason = state.can_trade(cfg)
    if not ok:
        return False, reason

    ok, reason = check_expiry_safety(expiry_str, cfg.min_minutes_to_expiry)
    if not ok:
        return False, reason

    prices_only = [x[1] for x in micro_candles] if micro_candles else []
    if cfg.choppy_filter_enabled and detect_choppy_market(prices_only):
        return False, "🌊 NO TRADE — Choppy/range-bound market detected. Standing aside to avoid overtrading."

    return True, "All entry filters passed."


# ==============================================================================
# 9. BROKER EXECUTION ABSTRACTION
# ==============================================================================
class BrokerExecutor(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: int, order_type: str = "MARKET",
                     limit_price: Optional[float] = None) -> dict:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        ...


class PaperBrokerExecutor(BrokerExecutor):
    def __init__(self, state: EngineState, slippage_pct: float = 0.15):
        self.state = state
        self.slippage_pct = slippage_pct
        self._seq = 0
        self._seq_lock = threading.Lock()

    def place_order(self, symbol: str, side: str, qty: int, order_type: str = "MARKET",
                     limit_price: Optional[float] = None) -> dict:
        with self._seq_lock:
            self._seq += 1
            order_id = f"PAPER-{int(time.time())}-{self._seq}"
        tick = self.state.get_tick(symbol)
        ref_price = tick["ltp"] if tick else (limit_price or 0.0)
        slip_mult = (1 + self.slippage_pct / 100.0) if side.upper() == "BUY" else (1 - self.slippage_pct / 100.0)
        fill_price = round(ref_price * slip_mult, 2)
        log_event("paper_order", f"Paper {side} {symbol} x{qty} filled @ {fill_price}", order_id=order_id)
        return {"order_id": order_id, "status": "FILLED", "fill_price": fill_price, "qty": qty}

    def cancel_order(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "CANCELLED"}

    def get_order_status(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "FILLED"}


class FyersLiveBrokerExecutor(BrokerExecutor):
    def __init__(self, client: "FyersClientWrapper"):
        self.client = client

    def place_order(self, symbol: str, side: str, qty: int, order_type: str = "MARKET",
                     limit_price: Optional[float] = None) -> dict:
        payload = {
            "symbol": symbol,
            "qty": qty,
            "type": 2 if order_type == "MARKET" else 1,
            "side": 1 if side.upper() == "BUY" else -1,
            "productType": "INTRADAY",
            "limitPrice": limit_price or 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }
        try:
            res = self.client.fyers.place_order(data=payload)
            log_event("live_order", f"LIVE order placed: {symbol} {side} x{qty}", response=res)
            return res
        except Exception as e:
            log_error("live_order_placement", e)
            return {"status": "ERROR", "reason": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self.client.fyers.cancel_order(data={"id": order_id})
        except Exception as e:
            log_error("live_order_cancel", e)
            return {"status": "ERROR", "reason": str(e)}

    def get_order_status(self, order_id: str) -> dict:
        try:
            return self.client.fyers.orderbook(data={"id": order_id})
        except Exception as e:
            log_error("live_order_status", e)
            return {"status": "ERROR", "reason": str(e)}


# ==============================================================================
# 10. FYERS WEBSOCKET WORKER
# ==============================================================================
class FyersWebSocketWorker(threading.Thread):
    def __init__(self, app_id: str, access_token: str, initial_symbols: List[str], state: EngineState):
        super().__init__(daemon=True)
        self.auth_token = f"{app_id}:{access_token}"
        self.initial_symbols = initial_symbols
        self.state = state
        self.running = True
        self.ws = None

    def on_message(self, message, *args, **kwargs):
        if isinstance(message, dict) and "symbol" in message:
            self.state.ws_queue.put(message)

    def on_open(self, *args, **kwargs):
        log_event("ws", f"Fyers WebSocket connected, subscribing {len(self.initial_symbols)} indices")
        self.state.ws_connected = True
        if self.ws and self.initial_symbols:
            self.ws.subscribe(symbols=self.initial_symbols, data_type="SymbolUpdate")

    def on_close(self, *args, **kwargs):
        log_event("ws", f"Fyers WebSocket closed: {args}", level=logging.WARNING)
        self.state.ws_connected = False

    def on_error(self, *args, **kwargs):
        err = args[0] if args else kwargs
        if isinstance(err, dict) and err.get("code") == -300:
            log_error("ws_auth", Exception(err.get("message")))
            if "token" in str(err.get("message")).lower():
                self.state.ws_connected = False
                self.running = False
        else:
            log_event("ws", f"Fyers WebSocket error: {args} {kwargs}", level=logging.WARNING)

    def subscribe_symbol(self, symbol: str) -> None:
        if self.ws and self.state.ws_connected:
            self.ws.subscribe(symbols=[symbol], data_type="DepthUpdate")

    def subscribe_multiple(self, symbols: List[str]) -> None:
        if self.ws and self.state.ws_connected and symbols:
            valid = list({s for s in symbols if s})
            if valid:
                self.ws.subscribe(symbols=valid, data_type="DepthUpdate")

    def run(self):
        backoff = 1
        while self.running:
            try:
                self.ws = data_ws.FyersDataSocket(
                    access_token=self.auth_token, log_path="", litemode=False,
                    write_to_file=False, reconnect=False, on_connect=self.on_open,
                    on_close=self.on_close, on_error=self.on_error, on_message=self.on_message,
                )
                self.ws.connect()
                time.sleep(2)
                while self.running and self.state.ws_connected:
                    time.sleep(1)
            except Exception as e:
                log_error("ws_exception", e)

            if not self.running:
                break
            self.state.ws_connected = False
            time.sleep(backoff)
            backoff = min(30, backoff * 2)


# ==============================================================================
# 11. FYERS REST API LOGIC (TTL-cached)
# ==============================================================================
class FyersClientWrapper:
    def __init__(self, client_id: str, access_token: str):
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")


def _fetch_candles(client: FyersClientWrapper, symbol: str, resolution: str) -> pd.DataFrame:
    today_str = datetime.now(IST_TZ).strftime("%Y-%m-%d")
    payload = {
        "symbol": symbol, "resolution": resolution, "date_format": "1",
        "range_from": today_str, "range_to": today_str, "cont_flag": "1",
    }
    try:
        res = client.fyers.history(data=payload)
    except Exception as e:
        log_error("fetch_candles", e)
        return pd.DataFrame()
    if not res or res.get("s") != "ok" or not res.get("candles"):
        return pd.DataFrame()
    df = pd.DataFrame(res["candles"], columns=["epoch", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert(IST_TZ)
    return df


def get_cached_candles(client: FyersClientWrapper, symbol: str, resolution: str = "1") -> pd.DataFrame:
    return _candles_cache.get_or_set(f"candles:{symbol}:{resolution}", lambda: _fetch_candles(client, symbol, resolution))


def _fetch_option_chain(client: FyersClientWrapper, symbol: str) -> dict:
    payload = {"symbol": symbol, "strikecount": 12}
    try:
        res = client.fyers.optionchain(data=payload)
    except Exception as e:
        log_error("fetch_option_chain", e)
        return {}
    if not res or res.get("s") != "ok":
        return {}

    data_block = res.get("data", {})
    options = data_block.get("optionsChain", [])
    if not options:
        return {}

    expiry_list = data_block.get("expiryData", [])
    target_expiry_str = "Unknown"
    if expiry_list and isinstance(expiry_list, list) and len(expiry_list) > 0:
        raw_date = expiry_list[0].get("date", "")
        if raw_date:
            try:
                parsed_date = datetime.strptime(raw_date, "%d-%m-%Y").date()
                target_expiry_str = parsed_date.strftime("%Y-%m-%d")
            except Exception:
                target_expiry_str = raw_date

    return {"options": options, "expiry": target_expiry_str}


def get_cached_option_chain(client: FyersClientWrapper, symbol: str) -> dict:
    return _option_chain_cache.get_or_set(f"chain:{symbol}", lambda: _fetch_option_chain(client, symbol))


# ==============================================================================
# 12. LIVE DUAL-MODE ANALYSIS & SIGNAL GENERATION
# ==============================================================================
def calculate_live_metrics(state: EngineState, spot_symbol: str, mode: str) -> dict:
    micros = state.get_micro_candles(spot_symbol)
    
    obi_score = 0.0 
    velocity = "FLAT"
    
    now = time.time()
    lookback_secs = 60 if mode == "Scalping" else 240
    
    valid_ticks = [x for x in micros if (now - x[0]) <= lookback_secs]
    prices_only = [x[1] for x in micros] 
    
    if len(valid_ticks) >= 10:
        first_price = valid_ticks[0][1]
        last_price = valid_ticks[-1][1]
        pct_change = ((last_price - first_price) / first_price) * 100
        
        vel_threshold = 0.03 if mode == "Scalping" else 0.06
        
        if pct_change > vel_threshold:
            velocity = "BULLISH_SURGE"
        elif pct_change < -vel_threshold:
            velocity = "BEARISH_SURGE"

    return {"obi_score": obi_score, "velocity": velocity, "is_choppy": detect_choppy_market(prices_only)}


def process_option_chain_live(chain_raw: dict, cfg: IndexConfig, spot: float) -> dict:
    options = chain_raw.get("options", [])
    if not options:
        return {"oi_bias": "NEUTRAL", "pcr": 1.0, "best_ce_sym": None, "best_pe_sym": None,
                 "best_ce_ltp": 0.0, "best_pe_ltp": 0.0, "mini_chain": [], "atm_strike": 0, "expiry": "Unknown"}

    atm = round(spot / cfg.strike_step) * cfg.strike_step
    df = pd.DataFrame(options)

    strike_col = "strike_price" if "strike_price" in df.columns else "strikePrice"
    type_col = "option_type" if "option_type" in df.columns else "optionType"

    near_df = df[(df[strike_col] >= atm - (STRIKES_AROUND_ATM * cfg.strike_step)) &
                 (df[strike_col] <= atm + (STRIKES_AROUND_ATM * cfg.strike_step))]

    ce_df = near_df[near_df[type_col] == "CE"]
    pe_df = near_df[near_df[type_col] == "PE"]

    oi_col = "oi"
    pcr = pe_df[oi_col].sum() / max(1, ce_df[oi_col].sum())

    bullish, bearish = 0.0, 0.0
    for _, row in near_df.iterrows():
        oi_chg = row.get("oic", row.get("change_in_oi", 0)) 
        
        if row[type_col] == "CE" and oi_chg < 0:
            bullish += 2.0
        elif row[type_col] == "CE" and oi_chg > 0:
            bearish += 1.0
        elif row[type_col] == "PE" and oi_chg < 0:
            bearish += 2.0
        elif row[type_col] == "PE" and oi_chg > 0:
            bullish += 1.0

    bias_score = bullish - bearish
    oi_bias = "BULLISH" if bias_score > 2 else "BEARISH" if bias_score < -2 else "NEUTRAL"

    atm_ce = ce_df[ce_df[strike_col] == atm]
    atm_pe = pe_df[pe_df[strike_col] == atm]

    expiry_str = chain_raw.get("expiry", "2026-12-31")
    t_years = get_dte_years(expiry_str)

    mini_chain = []
    for strike in [atm + (i * cfg.strike_step) for i in range(-2, 3)]:
        ce_row = df[(df[strike_col] == strike) & (df[type_col] == "CE")]
        pe_row = df[(df[strike_col] == strike) & (df[type_col] == "PE")]

        ce_ltp = float(ce_row.iloc[0].get("ltp", 0.0)) if not ce_row.empty else 0.0
        pe_ltp = float(pe_row.iloc[0].get("ltp", 0.0)) if not pe_row.empty else 0.0

        ce_greeks = compute_live_greeks_fast(spot, strike, t_years, ce_ltp, "CE")
        pe_greeks = compute_live_greeks_fast(spot, strike, t_years, pe_ltp, "PE")

        mini_chain.append({
            "CE_Sym": ce_row.iloc[0]["symbol"] if not ce_row.empty else None,
            "CE Price": ce_ltp, "CE Delta": ce_greeks["Delta"], "Strike": strike,
            "PE Price": pe_ltp, "PE Delta": pe_greeks["Delta"],
            "PE_Sym": pe_row.iloc[0]["symbol"] if not pe_row.empty else None,
        })

    best_ce_ltp = float(atm_ce.iloc[0].get("ltp", 0.0)) if not atm_ce.empty else 0.0
    best_pe_ltp = float(atm_pe.iloc[0].get("ltp", 0.0)) if not atm_pe.empty else 0.0

    best_ce_delta = compute_live_greeks_fast(spot, atm, t_years, best_ce_ltp, "CE")["Delta"] if not atm_ce.empty else 0.5
    best_pe_delta = compute_live_greeks_fast(spot, atm, t_years, best_pe_ltp, "PE")["Delta"] if not atm_pe.empty else -0.5

    return {
        "oi_bias": oi_bias, "pcr": pcr, "expiry": expiry_str,
        "best_ce_sym": atm_ce.iloc[0]["symbol"] if not atm_ce.empty else None,
        "best_pe_sym": atm_pe.iloc[0]["symbol"] if not atm_pe.empty else None,
        "best_ce_ltp": best_ce_ltp, "best_pe_ltp": best_pe_ltp,
        "best_ce_delta": best_ce_delta, "best_pe_delta": best_pe_delta,
        "mini_chain": mini_chain, "atm_strike": atm,
    }


def generate_live_signal(spot_price: float, live_metrics: dict, oi_data: dict, mode: str) -> dict:
    score, conf = 0.0, 30
    is_scalping = (mode == "Scalping")

    vel_weight = 2.5 if is_scalping else 1.5
    if live_metrics["velocity"] == "BULLISH_SURGE":
        score += vel_weight; conf += 20
    elif live_metrics["velocity"] == "BEARISH_SURGE":
        score -= vel_weight; conf += 20

    score += live_metrics["obi_score"] * 0.5

    oi_weight = 2.0 if is_scalping else 3.0
    if oi_data.get("oi_bias") == "BULLISH":
        score += oi_weight; conf += 25
    elif oi_data.get("oi_bias") == "BEARISH":
        score -= oi_weight; conf += 25

    pcr = oi_data.get("pcr", 1.0)
    if pcr > 1.2:
        score += 1.5; conf += 15
    elif pcr < 0.8:
        score -= 1.5; conf += 15

    if score > 0 and abs(oi_data.get("best_ce_delta", 0.5)) < 0.30:
        return {"signal": "NO TRADE", "score": score, "conf": conf, "reason": "CE Delta Filter Rejected (< 0.30)"}
    elif score < 0 and abs(oi_data.get("best_pe_delta", -0.5)) < 0.30:
        return {"signal": "NO TRADE", "score": score, "conf": conf, "reason": "PE Delta Filter Rejected (< 0.30)"}

    threshold = 2.0 if is_scalping else 3.0 
    signal = "BUY CALL" if score >= threshold else "BUY PUT" if score <= -threshold else "NO TRADE"

    return {
        "signal": signal, "score": round(score, 2), "conf": min(100, max(0, int(conf))),
        "reason": f"[{mode}] Score: {score:.1f} (Req: ±{threshold})",
    }


def calculate_dynamic_trade_levels(client: FyersClientWrapper, state: EngineState, symbol: str,
                                    fallback_ltp: float, mode: str, cfg: RiskConfig) -> dict:
    current_ltp = 0.0
    tick = state.get_tick(symbol)
    if tick and (time.time() - tick["ts"] <= 10.0):
        current_ltp = tick["ltp"]
    if current_ltp == 0.0:
        current_ltp = fallback_ltp
    if current_ltp <= 0.0:
        return {}

    res = "1" if mode == "Scalping" else "5"
    sl_mult = 0.8 if mode == "Scalping" else 1.5
    t1_mult = 1.0 if mode == "Scalping" else 2.0
    t2_mult = 1.5 if mode == "Scalping" else 3.5
    buffer_pct = 0.0025 if mode == "Scalping" else 0.0040

    opt_df = get_cached_candles(client, symbol, resolution=res)
    if not opt_df.empty and len(opt_df) >= 14:
        opt_df["tr0"] = (opt_df["high"] - opt_df["low"]).abs()
        opt_df["tr1"] = (opt_df["high"] - opt_df["close"].shift()).abs()
        opt_df["tr2"] = (opt_df["low"] - opt_df["close"].shift()).abs()
        atr = float(opt_df[["tr0", "tr1", "tr2"]].max(axis=1).rolling(14).mean().iloc[-1])
    else:
        atr = current_ltp * (0.05 if mode == "Scalping" else 0.09)

    entry = current_ltp * (1 + buffer_pct)
    sl = max(0.05, entry - (atr * sl_mult))
    zone_low, zone_high = compute_entry_zone(entry, cfg.entry_zone_pct)

    return {
        "current_ltp": round(current_ltp, 2),
        "entry": round(entry, 2),
        "entry_zone_low": zone_low,
        "entry_zone_high": zone_high,
        "sl": round(sl, 2),
        "target_1": round(entry + (atr * t1_mult), 2),
        "target_2": round(entry + (atr * t2_mult), 2),
        "atr": round(atr, 2),
    }


# ==============================================================================
# 13. ORDER PLACEMENT ENTRY POINT (used by app.py)
# ==============================================================================
def place_paper_trade(state: EngineState, cfg: RiskConfig, index_symbol: str, opt_symbol: str,
                       strike: float, opt_type: str, signal_name: str, levels: dict, lots: int) -> Tuple[bool, str, Optional[dict]]:
    trade_obj = {
        "index_symbol": index_symbol, "signal": signal_name, "symbol": opt_symbol,
        "strike": strike, "type": opt_type, "lots": lots,
        "entry": levels["entry"], "entry_zone_low": levels["entry_zone_low"],
        "entry_zone_high": levels["entry_zone_high"], "entry_zone_pct": cfg.entry_zone_pct,
        "entry_fill": None, "exit_price": None,
        "sl": levels["sl"], "original_sl": levels["sl"],
        "target_1": levels["target_1"], "target_2": levels["target_2"],
        "status": "WAITING", "is_live": 0, "pnl": 0.0,
        "placed_ts": time.time(), "entry_timeout_seconds": cfg.entry_timeout_seconds,
        "slippage_pct": cfg.slippage_pct,
        "timestamp": datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }

    ok, reason = state.register_trade(index_symbol, trade_obj, cfg)
    if not ok:
        log_event("order_rejected", f"Order rejected for {index_symbol}: {reason}", level=logging.WARNING)
        return False, reason, None

    db_id = db_save_trade(trade_obj)
    trade_obj["db_id"] = db_id
    log_trade("ORDER_PLACED", trade_obj,
              reason=f"Watching entry zone [{levels['entry_zone_low']}, {levels['entry_zone_high']}]")
    return True, "Order placed — watching entry zone.", trade_obj
