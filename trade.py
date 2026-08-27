"""
================================================================================
 ⚡ PRO Quant Options Signal Engine | Scalper Paper Trading Edition (v6.4)
================================================================================
FEATURES:
1. Low-Latency Event-Driven Execution: SL/Targets managed inside WS thread.
2. Live Market Data via WebSocket (DhanHQ).
3. Real-time dynamic ATR signal generation & MTF Confirmation.
4. 100% PAPER TRADING: Orders are tracked virtually against live LTP.
================================================================================
"""

import logging
import math
import time
import os
import struct
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import requests
import pytz
import streamlit as st
import websocket

# Optional: Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ==============================================================================
# 1. CONFIGURATION & LOGGING
# ==============================================================================
logger = logging.getLogger("quant_scalper_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(_fmt)
    logger.addHandler(_stream_handler)

DHAN_BASE_URL = "https://api.dhan.co/v2"
DHAN_WS_URL = "wss://api-feed.dhan.co"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
LOCAL_SCRIP_PATH = "scrip_master_fallback.csv"
DB_PATH = "quant_paper_trades.db"
JSON_LOG_PATH = "closed_paper_trades_log.json"

@dataclass(frozen=True)
class IndexConfig:
    label: str; symbol: str; exchange: str; security_id: str
    exchange_segment: str; opt_exchange_segment: str; strike_step: int; lot_size: int

# FIXED: 2026 Lot Sizes integrated from v6.3
INDEX_CONFIG: Dict[str, IndexConfig] = {
    "NIFTY50": IndexConfig("NIFTY 50", "NIFTY", "NSE", "13", "IDX_I", "NSE_FNO", 50, 65),
    "BANKNIFTY": IndexConfig("NIFTY BANK", "BANKNIFTY", "NSE", "25", "IDX_I", "NSE_FNO", 100, 30),
    "SENSEX": IndexConfig("BSE SENSEX", "SENSEX", "BSE", "51", "IDX_I", "BSE_FNO", 100, 20)
}

STRIKES_AROUND_ATM = 4
ATR_TARGET_MULT = 2.5
ATR_SL_MULT = 1.2
IST_TZ = pytz.timezone('Asia/Kolkata')

TICK_LOCK = threading.Lock()
DB_LOCK = threading.Lock()

# ⚡ HIGH-SPEED GLOBAL STATE: Bridges WS Thread and Streamlit UI
GLOBAL_STATE = {
    "tick_cache": {"INDEX": {}, "OPTION": {}},
    "active_trade": None, 
    "daily_pnl": 0.0,
    "last_trade_time": 0.0,
    "daily_trades_count": 0,
    "ws_connected": False,
    "system_paused": False,
    "last_closed_message": None # To pass toast alerts to Streamlit
}

# ==============================================================================
# 2. SQLITE & JSON DATABASE LAYER
# ==============================================================================
def init_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute('PRAGMA journal_mode=WAL;')
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, index_symbol TEXT, signal TEXT, strike FLOAT,
                opt_type TEXT, sec_id TEXT, entry FLOAT, sl FLOAT, target_1 FLOAT,
                target_2 FLOAT, lots INTEGER, status TEXT, pnl FLOAT, is_live INTEGER
            )
        """)
        conn.commit()
        conn.close()

def db_save_trade(trade: dict) -> int:
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (timestamp, index_symbol, signal, strike, opt_type, sec_id, entry, sl, target_1, target_2, lots, status, pnl, is_live)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get('timestamp', datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S")),
            trade['index_symbol'], trade['signal'], trade['strike'], trade['type'],
            trade['sec_id'], trade['entry'], trade['sl'], trade['target_1'], trade['target_2'],
            trade['lots'], trade['status'], trade['pnl'], 0
        ))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id

def db_update_trade(trade_id: int, status: str, pnl: float, sl: float = None):
    if not trade_id: return
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        if sl is not None:
            cursor.execute("UPDATE trades SET status = ?, pnl = ?, sl = ? WHERE id = ?", (status, pnl, sl, trade_id))
        else:
            cursor.execute("UPDATE trades SET status = ?, pnl = ? WHERE id = ?", (status, pnl, trade_id))
        conn.commit()
        conn.close()

def log_trade_to_json(trade: dict, exit_price: float):
    if trade.get('json_logged'): return
    log_entry = {
        "timestamp": datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": trade['index_symbol'], "strike": trade['strike'], "side": trade['type'],
        "entry": round(trade['entry'], 2), "exit": round(exit_price, 2), "pnl": round(trade['pnl'], 2)
    }
    try:
        data = []
        if os.path.exists(JSON_LOG_PATH):
            with open(JSON_LOG_PATH, 'r') as f:
                data = json.load(f)
        data.append(log_entry)
        with open(JSON_LOG_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        trade['json_logged'] = True
    except Exception as e:
        logger.error(f"JSON Log Error: {e}")

def db_load_today_trades() -> Tuple[pd.DataFrame, float]:
    init_db()
    today_str = datetime.now(IST_TZ).strftime("%Y-%m-%d")
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        query = f"""
        SELECT timestamp as Time, index_symbol as "Index", signal as Signal, 
               strike || ' ' || opt_type as Strike, entry as Entry, status as Status, pnl as PnL
        FROM trades WHERE timestamp LIKE '{today_str}%'
        """
        df = pd.read_sql_query(query, conn)
        cursor = conn.cursor()
        cursor.execute(f"SELECT SUM(pnl) FROM trades WHERE timestamp LIKE '{today_str}%'")
        row = cursor.fetchone()
        daily_pnl = row[0] if row and row[0] is not None else 0.0
        conn.close()
    return df, daily_pnl

# ==============================================================================
# 3. MILLISECOND LATENCY TRADE MANAGER (Runs in Background WS Thread)
# ==============================================================================
def process_live_tick_for_trade(sec_id: str, live_ltp: float):
    """⚡ Executes Stop Loss / Targets instantly without waiting for UI refresh."""
    trade = GLOBAL_STATE["active_trade"]
    if not trade or trade['sec_id'] != sec_id: return

    # Dynamically find lot size
    idx_key = trade['index_symbol'].replace(" ", "").replace("BANK", "BANKNIFTY")
    lot_size = INDEX_CONFIG.get(idx_key, INDEX_CONFIG["NIFTY50"]).lot_size
    
    calc_pnl = lambda ltp: round((ltp - trade['entry']) * trade['lots'] * lot_size, 2)

    if trade['status'] == 'WAITING':
        if live_ltp >= trade['entry']:
            trade['status'] = 'ENTERED'
            trade['pnl'] = 0.0
            threading.Thread(target=db_update_trade, args=(trade.get('db_id'), "ENTERED", 0.0)).start()
            
    elif trade['status'] == 'ENTERED':
        trade['pnl'] = calc_pnl(live_ltp)
        
        if live_ltp <= trade['sl']:
            trade['status'] = 'CLOSED'
            GLOBAL_STATE['daily_pnl'] += trade['pnl']
            threading.Thread(target=db_update_trade, args=(trade.get('db_id'), "STOP_LOSS", trade['pnl'])).start()
            log_trade_to_json(trade, live_ltp)
            GLOBAL_STATE["last_closed_message"] = f"🛑 Stop Loss Hit! PnL: ₹{trade['pnl']:.2f}"
            GLOBAL_STATE["last_trade_time"] = time.time()
            GLOBAL_STATE["active_trade"] = None
            
        elif live_ltp >= trade['target_1']:
            trade['status'] = 'PARTIAL_EXIT'
            trade['sl'] = trade['entry']
            threading.Thread(target=db_update_trade, args=(trade.get('db_id'), "PARTIAL_EXIT", trade['pnl'], trade['sl'])).start()

    elif trade['status'] == 'PARTIAL_EXIT':
        trade['pnl'] = calc_pnl(live_ltp)
        
        if live_ltp <= trade['sl']:
            trade['status'] = 'CLOSED'
            GLOBAL_STATE['daily_pnl'] += trade['pnl']
            threading.Thread(target=db_update_trade, args=(trade.get('db_id'), "TRAILED_SL", trade['pnl'])).start()
            log_trade_to_json(trade, live_ltp)
            GLOBAL_STATE["last_closed_message"] = f"🛡️ Trailed SL Hit. Final PnL: ₹{trade['pnl']:.2f}"
            GLOBAL_STATE["last_trade_time"] = time.time()
            GLOBAL_STATE["active_trade"] = None
            
        elif live_ltp >= trade['target_2']:
            trade['status'] = 'CLOSED'
            GLOBAL_STATE['daily_pnl'] += trade['pnl']
            threading.Thread(target=db_update_trade, args=(trade.get('db_id'), "TARGET_2", trade['pnl'])).start()
            log_trade_to_json(trade, live_ltp)
            GLOBAL_STATE["last_closed_message"] = f"🎯 Full Target Achieved! Final PnL: ₹{trade['pnl']:.2f}"
            GLOBAL_STATE["last_trade_time"] = time.time()
            GLOBAL_STATE["active_trade"] = None

# ==============================================================================
# 4. WEBSOCKET WORKER (Optimized for Scalping)
# ==============================================================================
class DhanWebSocketWorker(threading.Thread):
    def __init__(self, client_id, access_token):
        super().__init__(daemon=True)
        self.url = f"{DHAN_WS_URL}?version=2&token={access_token}&clientId={client_id}&authType=2"
        self.running = True

    def run(self):
        backoff = 1
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_message=self.on_message,
                    on_close=self.on_close,
                    on_open=self.on_open
                )
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"WS Exception: {e}")
            if not self.running: break
            GLOBAL_STATE["ws_connected"] = False
            time.sleep(backoff)
            backoff = min(60, backoff * 2)

    def on_open(self, ws):
        GLOBAL_STATE["ws_connected"] = True

    def on_message(self, ws, message):
        if not isinstance(message, bytes) or len(message) < 9: return
        try:
            sec_id_int = struct.unpack("<I", message[1:5])[0]
            ltp = struct.unpack("<f", message[5:9])[0]
            if ltp <= 0 or math.isnan(ltp): return
            
            sec_id = str(sec_id_int)
            now = time.time()
            
            # ⚡ OPTIMIZATION: Instant Trade Check
            process_live_tick_for_trade(sec_id, round(ltp, 2))

            # Store in cache for Streamlit UI
            with TICK_LOCK:
                if sec_id in [idx.security_id for idx in INDEX_CONFIG.values()]:
                    GLOBAL_STATE["tick_cache"]["INDEX"][sec_id] = {"ltp": round(ltp, 2), "ts": now}
                else:
                    GLOBAL_STATE["tick_cache"]["OPTION"][sec_id] = {"ltp": round(ltp, 2), "ts": now}
        except Exception:
            pass

    def on_close(self, ws, *args):
        GLOBAL_STATE["ws_connected"] = False

# ==============================================================================
# 5. REST API CLIENT (Data Fetching Only)
# ==============================================================================
class DhanClient:
    def __init__(self, client_id: str, access_token: str):
        self.headers = {
            "Content-Type": "application/json", "Accept": "application/json",
            "access-token": access_token.strip(), "client-id": client_id.strip(),
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self.session.post(f"{DHAN_BASE_URL}{path}", json=payload, timeout=8)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            return {}

    def get_intraday_minute(self, security_id: str, segment: str, instrument: str = "INDEX") -> pd.DataFrame:
        today_str = datetime.now(IST_TZ).strftime("%Y-%m-%d")
        payload = {"securityId": str(security_id), "exchangeSegment": segment, "instrument": instrument, "interval": "1", "fromDate": today_str, "toDate": today_str}
        data = self._post("/charts/intraday", payload)
        if not data or not data.get("close"): return pd.DataFrame()
        return pd.DataFrame({
            "open": pd.to_numeric(data.get("open", [])), "high": pd.to_numeric(data.get("high", [])),
            "low": pd.to_numeric(data.get("low", [])), "close": pd.to_numeric(data.get("close", [])),
            "volume": pd.to_numeric(data.get("volume", [0]*len(data.get("close", [])))),
        })

    def get_option_chain(self, security_id: str, segment: str, expiry: str) -> dict:
        return self._post("/optionchain", {"UnderlyingScrip": int(security_id), "UnderlyingSeg": segment, "Expiry": expiry}).get("data", {})
    
    def get_expiry_list(self, security_id: str, segment: str) -> list:
        return self._post("/optionchain/expirylist", {"UnderlyingScrip": int(security_id), "UnderlyingSeg": segment}).get("data", [])

# ==============================================================================
# 6. TECHNICAL ANALYSIS & SIGNAL ENGINE
# ==============================================================================
@st.cache_data(ttl=60, show_spinner=False)
def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if len(df) < period * 2: return pd.Series([0]*len(df), index=df.index)
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    tr = pd.DataFrame({'h-l': df['high']-df['low'], 'h-pc': abs(df['high']-df['close'].shift(1)), 'l-pc': abs(df['low']-df['close'].shift(1))}).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (df['plus_dm'].ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (df['minus_dm'].ewm(span=period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    return dx.ewm(span=period, adjust=False).mean()

@st.cache_data(ttl=60, show_spinner=False)
def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if len(df) < period + 1: return pd.Series([50]*len(df), index=df.index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def process_spot_and_mtf(df_1m: pd.DataFrame) -> dict:
    res = {"spot": 0.0, "vwap": 0.0, "adx": 0.0, "rsi": 50.0, "trend_1m": "SIDEWAYS", "trend_5m": "SIDEWAYS", "trend_15m": "SIDEWAYS", "vwap_distance_pct": 0.0, "daily_range_pct": 0.0}
    if df_1m.empty or len(df_1m) < 20: return res
    
    df_1m['typical_price'] = (df_1m['high'] + df_1m['low'] + df_1m['close']) / 3
    df_1m['vwap'] = (df_1m['typical_price'] * df_1m['volume']).cumsum() / max(1, df_1m['volume'].cumsum().iloc[-1])
    df_1m['adx'] = calc_adx(df_1m)
    df_1m['rsi'] = calc_rsi(df_1m)
    
    res["spot"], res["vwap"], res["adx"], res["rsi"] = float(df_1m['close'].iloc[-1]), float(df_1m['vwap'].iloc[-1]), float(df_1m['adx'].iloc[-1]), float(df_1m['rsi'].iloc[-1])
    res["vwap_distance_pct"] = abs(res["spot"] - res["vwap"]) / res["vwap"] * 100
    day_high, day_low = df_1m['high'].max(), df_1m['low'].min()
    res["daily_range_pct"] = (day_high - day_low) / day_low * 100
    
    df_5m = df_1m.groupby(np.arange(len(df_1m)) // 5).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_15m = df_1m.groupby(np.arange(len(df_1m)) // 15).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    
    def get_trend(d):
        if len(d) < 20: return "SIDEWAYS"
        e20, e50, c = d['close'].ewm(span=20).mean().iloc[-1], d['close'].ewm(span=50).mean().iloc[-1], d['close'].iloc[-1]
        return "BULLISH" if (e20 > e50 and c > e20) else "BEARISH" if (e20 < e50 and c < e20) else "SIDEWAYS"

    res["trend_1m"], res["trend_5m"], res["trend_15m"] = get_trend(df_1m), get_trend(df_5m), get_trend(df_15m)
    return res

@st.cache_data(ttl=10, show_spinner=False)
def process_option_chain(_client: DhanClient, cfg: IndexConfig, expiry: str, spot: float) -> dict:
    chain_raw = _client.get_option_chain(cfg.security_id, cfg.exchange_segment, expiry)
    rows = []
    for strike_str, sides in chain_raw.get("oc", {}).items():
        try: strike = float(strike_str)
        except ValueError: continue
        ce, pe = sides.get("ce", {}), sides.get("pe", {})
        rows.append({
            "strike": strike, "ce_oi": ce.get("oi", 0), "ce_prev_oi": ce.get("previous_oi", 0), "ce_iv": ce.get("implied_volatility", 0),
            "pe_oi": pe.get("oi", 0), "pe_prev_oi": pe.get("previous_oi", 0), "pe_iv": pe.get("implied_volatility", 0)
        })
    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    if df.empty: return {"oi_bias": "NEUTRAL", "sc_prob": 0.0, "pcr": 1.0, "avg_iv": 0.0, "df": pd.DataFrame()}

    atm = round(spot / cfg.strike_step) * cfg.strike_step
    near_df = df[(df["strike"] >= atm - (STRIKES_AROUND_ATM * cfg.strike_step)) & (df["strike"] <= atm + (STRIKES_AROUND_ATM * cfg.strike_step))]
    
    pcr = near_df["pe_oi"].sum() / max(1, near_df["ce_oi"].sum())
    bullish, bearish, sc_score, sc_max = 0, 0, 0, 0
    for _, row in near_df.iterrows():
        ce_oi_chg, pe_oi_chg = row["ce_oi"] - row["ce_prev_oi"], row["pe_oi"] - row["pe_prev_oi"]
        if ce_oi_chg < 0: bullish += 1.5; sc_score += abs(ce_oi_chg); sc_max += abs(ce_oi_chg)
        elif ce_oi_chg > 0: bearish += 1.0; sc_max += abs(ce_oi_chg)
        if pe_oi_chg < 0: bearish += 1.5; sc_score += abs(pe_oi_chg); sc_max += abs(pe_oi_chg)
        elif pe_oi_chg > 0: bullish += 1.0; sc_max += abs(pe_oi_chg)

    sc_prob = sc_score / max(1, sc_max)
    bias_score = bullish - bearish
    oi_bias = "BULLISH" if bias_score > 2 else "BEARISH" if bias_score < -2 else "NEUTRAL"
    return {"oi_bias": oi_bias, "sc_prob": sc_prob, "pcr": pcr, "avg_iv": (near_df["ce_iv"].mean() + near_df["pe_iv"].mean()) / 2.0, "df": df}

def generate_signal(spot_data: dict, oi_data: dict, mode: str, expiry: str) -> dict:
    score, conf = 0.0, 30 
    t_5m, t_15m = spot_data["trend_5m"], spot_data["trend_15m"]
    is_aggressive = (mode == "Aggressive")
    
    if spot_data["vwap_distance_pct"] > 0.75 and not is_aggressive: return {"signal": "NO TRADE", "conf": 0, "reason": "Blocked: Price Overextended from VWAP"}

    vwap_bullish, vwap_bearish = spot_data["spot"] > spot_data["vwap"], spot_data["spot"] < spot_data["vwap"]
    
    if not is_aggressive:
        if "SIDEWAYS" in [t_5m, t_15m]: return {"signal": "NO TRADE", "conf": 0, "reason": "MTF Sideways"}
        if t_5m != t_15m: return {"signal": "NO TRADE", "conf": 0, "reason": "MTF Conflict"}
        if spot_data["adx"] < 20: return {"signal": "NO TRADE", "conf": 0, "reason": "Low ADX (<20)"}
    
    if t_5m == "BULLISH": score += (1.5 if is_aggressive else 2.0); conf += 15
    elif t_5m == "BEARISH": score -= (1.5 if is_aggressive else 2.0); conf += 15
    if vwap_bullish: score += 1.0; conf += 10
    elif vwap_bearish: score -= 1.0; conf += 10
    
    rsi = spot_data["rsi"]
    if rsi > 60: score += 1.0; conf += 10
    elif rsi < 40: score -= 1.0; conf += 10
    if rsi > 80 and not is_aggressive: return {"signal": "NO TRADE", "conf": 0, "reason": "Blocked: RSI Overbought (>80)"}
    if rsi < 20 and not is_aggressive: return {"signal": "NO TRADE", "conf": 0, "reason": "Blocked: RSI Oversold (<20)"}

    if oi_data["oi_bias"] == "BULLISH": score += 1.5; conf += 20
    elif oi_data["oi_bias"] == "BEARISH": score -= 1.5; conf += 20
    if oi_data["pcr"] > 1.1: score += 1.0; conf += 10
    elif oi_data["pcr"] < 0.8: score -= 1.0; conf += 10
    
    threshold = 4.5 if is_aggressive else 6.0
    signal = "BUY CALL" if score >= threshold else "BUY PUT" if score <= -threshold else "NO TRADE"
    return {"signal": signal, "score": score, "threshold": threshold, "conf": min(100, max(0, int(conf))), "reason": f"Score: {score:.1f}"}

def resolve_option_sec_id(scrip_df: pd.DataFrame, cfg: IndexConfig, expiry: str, strike: float, opt_type: str) -> str:
    try:
        exp_dt = pd.to_datetime(expiry).tz_localize(None)
        match = scrip_df[(scrip_df["SEM_EXM_EXCH_ID"] == cfg.exchange) & (scrip_df["SEM_CUSTOM_SYMBOL"].str.contains(cfg.symbol, na=False)) & (scrip_df["SEM_STRIKE_PRICE"] == strike) & (scrip_df["SEM_OPTION_TYPE"] == opt_type)]
        match = match[match["SEM_EXPIRY_DATE"].dt.date == exp_dt.date()]
        return str(match.iloc[0]["SEM_SMST_SECURITY_ID"]) if not match.empty else None
    except Exception: return None

def calculate_dynamic_trade_levels(client: DhanClient, cfg: IndexConfig, sec_id: str) -> dict:
    with TICK_LOCK:
        ws_data = GLOBAL_STATE["tick_cache"]["OPTION"].get(sec_id)
    if not ws_data or (time.time() - ws_data["ts"] > 2.0): return {}
    
    current_ltp = ws_data["ltp"]
    opt_df = client.get_intraday_minute(sec_id, cfg.opt_exchange_segment, "OPTIDX")
    if not opt_df.empty and len(opt_df) >= 14:
        opt_df["tr0"], opt_df["tr1"], opt_df["tr2"] = (opt_df["high"]-opt_df["low"]).abs(), (opt_df["high"]-opt_df["close"].shift()).abs(), (opt_df["low"]-opt_df["close"].shift()).abs()
        atr = float(opt_df[["tr0", "tr1", "tr2"]].max(axis=1).rolling(14).mean().iloc[-1])
    else:
        atr = current_ltp * 0.05

    entry = current_ltp + 0.5 
    sl = max(0.05, entry - (atr * ATR_SL_MULT)) - 0.5 
    return {"current_ltp": round(current_ltp, 2), "entry": round(entry, 2), "sl": round(sl, 2), "target_1": round(entry + (atr * 1.0), 2), "target_2": round(entry + (atr * ATR_TARGET_MULT), 2), "atr": round(atr, 2)}

def calculate_position_size(capital: float, risk_pct: float, entry: float, sl: float, lot_size: int, daily_loss_limit_pct: float) -> dict:
    if GLOBAL_STATE["daily_pnl"] <= -(capital * (daily_loss_limit_pct / 100)): return {"allowed": False, "reason": "Max Daily Drawdown Reached.", "lots": 0}
    sl_distance = abs(entry - sl)
    if sl_distance <= 0: return {"allowed": False, "reason": "Invalid SL Distance", "lots": 0}
    
    lots = max(1, math.floor((capital * (risk_pct / 100.0)) / sl_distance / lot_size))
    if (lots * lot_size * entry) > capital: lots = math.floor(capital / (lot_size * entry))
    if lots == 0: return {"allowed": False, "reason": "Insufficient Capital for 1 Lot.", "lots": 0}
    return {"allowed": True, "reason": "Risk Approved", "lots": lots, "max_loss": round(sl_distance * lots * lot_size, 2)}

# ==============================================================================
# 7. STREAMLIT UI
# ==============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_scrip_master() -> pd.DataFrame:
    try:
        df = pd.read_csv(SCRIP_MASTER_URL, low_memory=False)
        df.to_csv(LOCAL_SCRIP_PATH, index=False)
        return df
    except Exception:
        return pd.read_csv(LOCAL_SCRIP_PATH, low_memory=False) if os.path.exists(LOCAL_SCRIP_PATH) else pd.DataFrame()

def main():
    st.set_page_config(page_title="PRO Scalping Engine v6.4", layout="wide")
    
    # Initialize DB and Global State once
    if 'initialized' not in st.session_state:
        df, pnl = db_load_today_trades()
        GLOBAL_STATE["daily_pnl"] = pnl
        GLOBAL_STATE["daily_trades_count"] = len(df)
        st.session_state.initialized = True
        st.session_state.ws_worker = None
        
    now_ist = datetime.now(IST_TZ)
    market_open, market_close = now_ist.replace(hour=9, minute=15, second=0), now_ist.replace(hour=15, minute=30, second=0)
    market_status = "MARKET CLOSED" if (now_ist.weekday() >= 5 or now_ist < market_open or now_ist >= market_close) else "LIVE MARKET"

    # Display WS Notifications pushed from background thread
    if GLOBAL_STATE["last_closed_message"]:
        st.toast(GLOBAL_STATE["last_closed_message"])
        GLOBAL_STATE["last_closed_message"] = None

    st.markdown(f"**🕒 Market Status:** `{market_status}` | **WS Connected:** `{GLOBAL_STATE['ws_connected']}`")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        client_id = st.text_input("Dhan Client ID", type="password")
        access_token = st.text_input("Dhan Token", type="password")
        index_key = st.selectbox("Index", options=list(INDEX_CONFIG.keys()))
        engine_mode = st.radio("Signal Mode", ["Conservative", "Aggressive"], index=0)
        
        st.subheader("Capital & Risk")
        user_capital = st.number_input("Paper Capital (₹)", value=100000, step=10000)
        risk_pct = st.slider("Risk Per Trade (%)", 1.0, 5.0, 2.0)
        daily_loss_limit = st.slider("Max Daily Drawdown (%)", 2.0, 10.0, 5.0)
        max_trades = st.number_input("Max Trades Per Day", min_value=1, max_value=10, value=5)
        cooldown_mins = st.number_input("Trade Cooldown (Mins)", min_value=1, max_value=60, value=10)
        
        st.toggle("Paper Trading Mode", value=True, disabled=True, help="Locked to Virtual Execution")

        if st.button("Start WebSocket Feed"):
            if st.session_state.ws_worker is None or not st.session_state.ws_worker.is_alive():
                worker = DhanWebSocketWorker(client_id, access_token)
                st.session_state.ws_worker = worker
                worker.start()
                st.toast("Feed started! Scalping Engine Online.", icon="⚡")
                
        if st.toggle("Auto-Refresh UI", value=True) and market_status == "LIVE MARKET":
            st_autorefresh(interval=1500, key="data_refresh") # Fast 1.5s UI refresh

    if not client_id or not access_token:
        st.warning("Please configure your DhanHQ credentials in the sidebar.")
        st.stop()

    client = DhanClient(client_id, access_token)
    cfg = INDEX_CONFIG[index_key]
    scrip_df = get_scrip_master()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Available Paper Capital", f"₹{user_capital:,.2f}")
    col_m2.metric("Today's Paper PnL", f"₹{GLOBAL_STATE['daily_pnl']:,.2f}")
    col_m3.metric("Max Allowed Daily Loss", f"₹{(user_capital * (daily_loss_limit/100)):,.2f}")

    df_1m = client.get_intraday_minute(cfg.security_id, cfg.exchange_segment)
    spot_data = process_spot_and_mtf(df_1m)
    spot_price = spot_data["spot"]
    
    with TICK_LOCK:
        if cfg.security_id in GLOBAL_STATE["tick_cache"]["INDEX"]:
            spot_price = GLOBAL_STATE["tick_cache"]["INDEX"][cfg.security_id]["ltp"]

    st.write(f"### Live {cfg.label} Spot: ₹{spot_price:,.2f}")
    
    active_trade = GLOBAL_STATE["active_trade"]
    
    if active_trade:
        live_opt_ltp = active_trade['entry']
        with TICK_LOCK:
            ws_opt_data = GLOBAL_STATE["tick_cache"]["OPTION"].get(active_trade['sec_id'])
            if ws_opt_data and (time.time() - ws_opt_data["ts"] < 2.0): live_opt_ltp = ws_opt_data["ltp"]
            
        st.success(f"### ⚡ ACTIVE POSITION: {active_trade['strike']} {active_trade['type']}")
        st.info(f"**Status:** `{active_trade['status']}` | **LTP:** ₹{live_opt_ltp:.2f} | **Live PnL:** ₹{active_trade.get('pnl', 0.0):.2f}")
        st.progress(max(0.0, min(1.0, (live_opt_ltp - active_trade['sl']) / max(1.0, (active_trade['target_2'] - active_trade['sl'])))))
        
    else:
        expiries = client.get_expiry_list(cfg.security_id, cfg.exchange_segment)
        cooldown_passed = (time.time() - GLOBAL_STATE["last_trade_time"]) > (cooldown_mins * 60)
        daily_limit_ok = GLOBAL_STATE["daily_trades_count"] < max_trades
        
        if not cooldown_passed: st.warning("Cooldown active. Waiting for system to stabilize.")
        elif not daily_limit_ok: st.warning(f"Maximum daily trades ({max_trades}) reached.")
        elif expiries and spot_price > 0:
            active_exp = expiries[0]
            oi_data = process_option_chain(client, cfg, active_exp, spot_price)
            signal = generate_signal(spot_data, oi_data, engine_mode, active_exp)
            
            st.write(f"**Signal Status:** `{signal['signal']}` (Confidence: {signal['conf']}%)")
            st.caption(f"Details: {signal['reason']}")
            
            if signal["signal"] != "NO TRADE":
                opt_type = "CE" if signal["signal"] == "BUY CALL" else "PE"
                rec_strike = round(spot_price / cfg.strike_step) * cfg.strike_step
                sec_id = resolve_option_sec_id(scrip_df, cfg, active_exp, rec_strike, opt_type)
                
                if sec_id:
                    levels = calculate_dynamic_trade_levels(client, cfg, sec_id)
                    if levels and market_status == "LIVE MARKET" and GLOBAL_STATE["ws_connected"]:
                        risk_check = calculate_position_size(user_capital, risk_pct, levels['entry'], levels['sl'], cfg.lot_size, daily_loss_limit)
                        
                        st.write("---")
                        st.subheader("🔥 Scalp Execution Parameters")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Dynamic Entry", f"₹{levels['entry']}")
                        c2.metric("Target 1 (1x ATR)", f"₹{levels['target_1']}")
                        c3.metric("Target 2 (2.5x ATR)", f"₹{levels['target_2']}")
                        c4.metric("Stop Loss", f"₹{levels['sl']}")
                        
                        if risk_check['allowed']:
                            st.success(f"Position Approved: **{risk_check['lots']} Lots** | Max Risk: **₹{risk_check['max_loss']}**")
                            if st.button(f"Place {rec_strike} {opt_type} Paper Trade"):
                                trade_obj = {
                                    "index_symbol": cfg.symbol, "signal": signal["signal"], "sec_id": sec_id, "strike": rec_strike, "type": opt_type,
                                    "lots": risk_check['lots'], "entry": levels['entry'], "sl": levels['sl'], "target_1": levels['target_1'], 
                                    "target_2": levels['target_2'], "status": "WAITING", "is_live": 0, "pnl": 0.0, 
                                    "timestamp": datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S")
                                }
                                db_id = db_save_trade(trade_obj)
                                trade_obj['db_id'] = db_id
                                
                                # Assign to GLOBAL_STATE to let WS thread instantly take over
                                GLOBAL_STATE["active_trade"] = trade_obj
                                GLOBAL_STATE["daily_trades_count"] += 1
                                st.rerun()
                        else:
                            st.error(f"Trade Blocked: {risk_check['reason']}")

    st.write("---")
    st.subheader("📜 Today's Ledger")
    df_history, _ = db_load_today_trades()
    st.dataframe(df_history, use_container_width=True)

if __name__ == "__main__":
    main()