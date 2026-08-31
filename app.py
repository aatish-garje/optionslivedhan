"""
================================================================================
 ⚡ PRO Quant Options Signal Engine | Streamlit UI (Production Grade)
================================================================================
app.py contains ONLY presentation logic. All trading, risk, execution and
persistence logic lives in engine.py and can run independently of Streamlit
(requirement #6). This file:
    - Reads sidebar config into an engine.RiskConfig
    - Boots a single shared engine.EngineState via st.cache_resource
    - Starts the WebSocket + tick-processing workers once
    - Calls engine functions to get signals / levels / risk checks
    - Renders the UI and dispatches order placement / manual exit through
      engine.place_paper_trade() and engine.PaperBrokerExecutor
================================================================================
"""

import time
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

import engine
from engine import (
    INDEX_CONFIG, IST_TZ, RiskConfig, EngineState, FyersClientWrapper, FyersWebSocketWorker,
    PaperBrokerExecutor, get_cached_option_chain, process_option_chain_live, calculate_live_metrics,
    generate_live_signal, calculate_dynamic_trade_levels, calculate_position_size,
    evaluate_all_entry_filters, place_paper_trade, db_load_today_trades, log_error,
)


# ==============================================================================
# ENGINE BOOTSTRAP (single shared instance across reruns / sessions)
# ==============================================================================
@st.cache_resource
def get_engine_state() -> EngineState:
    return EngineState()


@st.cache_resource
def get_tick_worker_started(_state: EngineState, _cfg_box: dict) -> bool:
    """Starts the background tick-processing thread exactly once per process.
    `_cfg_box` is a mutable dict the sidebar keeps updated so the worker
    always reads the LATEST risk config without needing a restart."""

    def cfg_provider() -> RiskConfig:
        return _cfg_box["cfg"]

    def lot_size_lookup(index_key: str) -> int:
        idx_key = index_key.replace(" ", "").replace("BANK", "BANKNIFTY")
        return INDEX_CONFIG.get(idx_key, INDEX_CONFIG["NIFTY50"]).lot_size

    engine.start_tick_worker(_state, cfg_provider, lot_size_lookup)
    return True


# A tiny mutable box so the tick-worker thread (started once) can still see
# sidebar changes made on later reruns.
if "cfg_box" not in st.session_state:
    st.session_state.cfg_box = {"cfg": RiskConfig()}


def build_risk_config(sidebar_values: dict) -> RiskConfig:
    return RiskConfig(
        capital=sidebar_values["capital"],
        risk_pct=sidebar_values["risk_pct"],
        max_capital_per_trade_pct=sidebar_values["max_capital_pct"],
        daily_loss_limit_pct=sidebar_values["daily_loss_limit"],
        max_trades_per_day=sidebar_values["max_trades"],
        cooldown_seconds=sidebar_values["cooldown_mins"] * 60,
        min_minutes_to_expiry=sidebar_values["min_minutes_to_expiry"],
        slippage_pct=sidebar_values["slippage_pct"],
        entry_zone_pct=sidebar_values["entry_zone_pct"],
        entry_timeout_seconds=sidebar_values["entry_timeout_secs"],
        choppy_filter_enabled=sidebar_values["choppy_filter"],
    )


def main():
    st.set_page_config(page_title="Options Paper Trading Engine", layout="wide")
    state = get_engine_state()

    if "initialized" not in st.session_state:
        df, pnl = db_load_today_trades()
        state.hydrate_daily(pnl, len(df))
        st.session_state.initialized = True

    now_ist = datetime.now(IST_TZ)
    market_open = now_ist.replace(hour=9, minute=15, second=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0)
    market_status = "MARKET CLOSED" if (now_ist.weekday() >= 5 or now_ist < market_open or now_ist >= market_close) else "LIVE MARKET"

    closed_msg = state.pop_closed_message()
    if closed_msg:
        st.toast(closed_msg)

    with st.sidebar:
        st.header("⚙️ Configuration")
        client_id = st.text_input("Fyers App ID (e.g., XCXXXXX-200)")
        access_token = st.text_input("Fyers Access Token", type="password")
        engine_mode = st.radio("Trading Mode", ["Scalping", "Intraday"], index=0)

        broker_mode = st.radio(
            "Execution Mode", ["Paper (Simulated)", "Live (Coming Soon)"], index=0,
            help="Live trading routes through engine.FyersLiveBrokerExecutor once wired up — "
                 "kept disabled here so no real orders can be sent from this UI yet.",
        )
        if broker_mode.startswith("Live"):
            st.warning("🚧 Live execution hook exists in `engine.FyersLiveBrokerExecutor` but is not "
                       "enabled from this UI. Wire it in only after independent testing.")

        st.subheader("Risk & Capital Controls")
        capital = st.number_input("Paper Capital (₹)", value=100000, step=10000)
        risk_pct = st.slider("Risk Per Trade (%)", 1.0, 5.0, 2.0)
        max_capital_pct = st.slider("Max Capital Deployed / Trade (%)", 5.0, 50.0, 25.0,
                                     help="Hard safety cap independent of Risk %.")
        daily_loss_limit = st.slider("Max Daily Drawdown (%) — Kill Switch", 2.0, 10.0, 5.0)
        max_trades = st.number_input("Max Daily Trades", min_value=1, max_value=20, value=10)
        cooldown_mins = st.number_input("Trade Cooldown (Mins)", min_value=0, max_value=60, value=2)

        st.subheader("Execution Realism")
        slippage_pct = st.slider("Slippage (%)", 0.10, 0.30, 0.15, step=0.01,
                                  help="Applied against the trader on fill, simulating real market impact.")
        entry_zone_pct = st.slider("Entry Zone Width (±%)", 0.05, 0.50, 0.15, step=0.01,
                                    help="Order fills only when price trades inside this zone (limit-style).")
        entry_timeout_secs = st.number_input("Entry Order Timeout (secs)", min_value=15, max_value=600, value=90,
                                              help="Pending orders auto-cancel if the zone is never reached.")

        st.subheader("Safety Filters")
        min_minutes_to_expiry = st.number_input("Min Minutes to Expiry", min_value=1, max_value=180, value=30)
        choppy_filter = st.toggle("Choppy-Market / Overtrading Guard", value=True)

        sidebar_values = dict(
            capital=capital, risk_pct=risk_pct, max_capital_pct=max_capital_pct,
            daily_loss_limit=daily_loss_limit, max_trades=max_trades, cooldown_mins=cooldown_mins,
            slippage_pct=slippage_pct, entry_zone_pct=entry_zone_pct, entry_timeout_secs=entry_timeout_secs,
            min_minutes_to_expiry=min_minutes_to_expiry, choppy_filter=choppy_filter,
        )
        cfg = build_risk_config(sidebar_values)
        st.session_state.cfg_box["cfg"] = cfg  # keep the running tick worker in sync

        if client_id and access_token:
            if len(access_token) > 50:
                worker = state.ws_worker
                if worker is None or not worker.is_alive():
                    all_spot_symbols = [c.symbol for c in INDEX_CONFIG.values()]
                    new_worker = FyersWebSocketWorker(client_id, access_token, all_spot_symbols, state)
                    state.ws_worker = new_worker
                    new_worker.start()
                    get_tick_worker_started(state, st.session_state.cfg_box)
                    time.sleep(1.5)
                    st.rerun()
            else:
                st.warning("⚠️ Access token incomplete.")

        if st.toggle("Auto-Refresh UI", value=True) and market_status == "LIVE MARKET" and AUTOREFRESH_AVAILABLE:
            st_autorefresh(interval=3000, key="data_refresh")

        st.markdown("---")
        st.markdown(f"**🕒 Market Status:** `{market_status}`\n\n**WS Connected:** `{state.ws_connected}`")
        st.metric("Today's Paper PnL", f"₹{state.daily_pnl:,.2f}")

    if not client_id or not access_token:
        st.warning("Please configure your Fyers credentials in the sidebar.")
        st.stop()

    client = FyersClientWrapper(client_id, access_token)
    executor = PaperBrokerExecutor(state, slippage_pct=cfg.slippage_pct)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Available Capital", f"₹{cfg.capital:,.2f}")
    col_m2.metric("Today's Paper PnL", f"₹{state.daily_pnl:,.2f}")
    col_m3.metric("Max Daily Loss Limit", f"₹{(cfg.capital * (cfg.daily_loss_limit_pct / 100.0)):,.2f}")

    tab_keys = list(INDEX_CONFIG.keys())
    tabs = st.tabs([INDEX_CONFIG[k].label for k in tab_keys])

    for i, tab in enumerate(tabs):
        with tab:
            idx_key = tab_keys[i]
            idx_cfg = INDEX_CONFIG[idx_key]

            tick = state.get_tick(idx_cfg.symbol)
            spot_price = tick["ltp"] if tick else 0.0

            if spot_price == 0.0:
                st.info(f"Awaiting live tick data for {idx_cfg.label} from WebSocket...")
                continue

            st.write(f"### Live Spot: ₹{spot_price:,.2f}")

            active_trade = state.get_active_trade(idx_cfg.symbol)

            # =========================
            # ACTIVE POSITION VIEW
            # =========================
            if active_trade:
                live_opt_ltp = active_trade.get("entry_fill") or active_trade["entry"]
                opt_tick = state.get_tick(active_trade["symbol"])
                if opt_tick:
                    live_opt_ltp = opt_tick["ltp"]

                col_pos, col_act = st.columns([3, 1])
                with col_pos:
                    st.success(f"### ⚡ ACTIVE POSITION: {active_trade['strike']} {active_trade['type']}")
                    if active_trade["status"] == "WAITING":
                        st.info(
                            f"**Status:** `WAITING` | **Watching Zone:** "
                            f"₹{active_trade['entry_zone_low']:.2f} – ₹{active_trade['entry_zone_high']:.2f} | "
                            f"**LTP:** ₹{live_opt_ltp:.2f}"
                        )
                    else:
                        st.info(
                            f"**Status:** `{active_trade['status']}` | "
                            f"**LTP:** ₹{live_opt_ltp:.2f} | "
                            f"**Fill Price:** ₹{(active_trade.get('entry_fill') or 0):.2f} | "
                            f"**SL:** ₹{active_trade['sl']:.2f} | "
                            f"**Target 1:** ₹{active_trade['target_1']:.2f} | "
                            f"**Target 2:** ₹{active_trade['target_2']:.2f} | "
                            f"**Live PnL:** ₹{active_trade.get('pnl', 0.0):.2f}"
                        )

                with col_act:
                    st.write("### Actions")
                    is_waiting = (active_trade["status"] == "WAITING")
                    btn_label = "🚫 Cancel Pending Order" if is_waiting else "🛑 Square Off / Exit Trade"

                    if st.button(btn_label, key=f"btn_{idx_cfg.symbol}", type="primary"):
                        if is_waiting:
                            state.remove_trade(idx_cfg.symbol)
                            engine.db_update_trade(active_trade.get("db_id"), "CANCELLED", 0.0)
                            state.set_closed_message("🚫 Pending order cancelled by user.")
                        else:
                            # Route the manual exit through the broker abstraction so it
                            # gets a realistic (slippage-adjusted) fill, same as an SL/target exit.
                            fill = executor.place_order(active_trade["symbol"], "SELL", active_trade["lots"])
                            exit_price = fill["fill_price"]
                            lot_size = idx_cfg.lot_size
                            entry_ref = active_trade.get("entry_fill") or active_trade["entry"]
                            final_pnl = round((exit_price - entry_ref) * active_trade["lots"] * lot_size, 2)

                            state.manual_close(idx_cfg.symbol, final_pnl)
                            engine.db_update_trade(active_trade.get("db_id"), "MANUAL_EXIT", final_pnl,
                                                    exit_price=exit_price, exit_reason="MANUAL_EXIT")
                            engine.log_trade_to_json({**active_trade, "pnl": final_pnl}, exit_price)
                            state.set_closed_message(
                                f"⚠️ Trade MANUAL_EXIT! Exit Price: ₹{exit_price:.2f} | PnL: ₹{final_pnl:.2f}"
                            )
                        st.rerun()

            # =========================
            # TRADE SIGNAL & ENTRY VIEW
            # =========================
            else:
                chain_raw = get_cached_option_chain(client, idx_cfg.symbol)
                oi_data = process_option_chain_live(chain_raw, idx_cfg, spot_price)
                live_metrics = calculate_live_metrics(state, idx_cfg.symbol, engine_mode)
                signal = generate_live_signal(spot_price, live_metrics, oi_data, engine_mode)
                engine.log_signal(idx_cfg.symbol, signal)

                col_info, col_chain = st.columns([1.1, 1.1])

                with col_info:
                    st.write(f"**Target Expiry:** `{oi_data.get('expiry', 'Unknown')}`")
                    obi_label = "Buyers" if live_metrics["obi_score"] > 0 else "Sellers" if live_metrics["obi_score"] < 0 else "Neutral"
                    st.write(f"**Live Velocity:** `{live_metrics['velocity']}` | **Order Book:** `{obi_label}` | "
                             f"**Chop Guard:** `{'CHOPPY' if live_metrics['is_choppy'] else 'clear'}`")
                    st.write(f"**Signal Status:** `{signal['signal']}` (Conf: {signal['conf']}%) - {signal['reason']}")

                    micro_candles = state.get_micro_candles(idx_cfg.symbol)
                    gates_ok, gates_reason = evaluate_all_entry_filters(
                        state, cfg, oi_data.get("expiry", "Unknown"), micro_candles
                    )

                    if not gates_ok:
                        st.warning(gates_reason)
                    elif signal["signal"] != "NO TRADE":
                        opt_type = "CE" if signal["signal"] == "BUY CALL" else "PE"
                        rec_strike = round(spot_price / idx_cfg.strike_step) * idx_cfg.strike_step
                        opt_symbol = oi_data.get("best_ce_sym") if opt_type == "CE" else oi_data.get("best_pe_sym")
                        fallback_ltp = oi_data.get("best_ce_ltp") if opt_type == "CE" else oi_data.get("best_pe_ltp")

                        if opt_symbol:
                            worker = state.ws_worker
                            if worker and worker.is_alive():
                                worker.subscribe_symbol(opt_symbol)

                            levels = calculate_dynamic_trade_levels(client, state, opt_symbol, fallback_ltp, engine_mode, cfg)
                            if levels and market_status == "LIVE MARKET":
                                risk_check = calculate_position_size(state, cfg, levels["entry"], levels["sl"], idx_cfg.lot_size)

                                st.write("---")
                                st.subheader(f"🔥 Suggested Trade: {rec_strike} {opt_type}")
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Entry Zone", f"₹{levels['entry_zone_low']}–{levels['entry_zone_high']}")
                                c2.metric("Target 1", f"₹{levels['target_1']}")
                                c3.metric("Target 2", f"₹{levels['target_2']}")
                                c4.metric("Stop Loss", f"₹{levels['sl']}")
                                st.caption(f"Slippage on fill: {cfg.slippage_pct}% | "
                                           f"Order auto-cancels after {cfg.entry_timeout_seconds}s if zone unreached.")

                                if risk_check["allowed"]:
                                    st.success(f"✅ Risk Approved: **{risk_check['lots']} Lots** | Max Risk: **₹{risk_check['max_loss']}**")
                                    if st.button(f"⚡ Place Order — {rec_strike} {opt_type} ({risk_check['lots']} Lots)",
                                                 key=f"trade_{idx_cfg.symbol}", type="primary"):
                                        ok, msg, trade_obj = place_paper_trade(
                                            state, cfg, idx_cfg.symbol, opt_symbol, rec_strike, opt_type,
                                            signal["signal"], levels, risk_check["lots"],
                                        )
                                        if ok:
                                            st.success(msg)
                                        else:
                                            st.error(f"🚫 Order Rejected: {msg}")
                                        st.rerun()
                                else:
                                    st.error(f"🚫 Trade Blocked: {risk_check['reason']}")

                with col_chain:
                    mini_chain = oi_data.get("mini_chain", [])
                    if mini_chain:
                        symbols_to_sub = []
                        for row in mini_chain:
                            if row["CE_Sym"]:
                                symbols_to_sub.append(row["CE_Sym"])
                            if row["PE_Sym"]:
                                symbols_to_sub.append(row["PE_Sym"])

                            ce_tick = state.get_tick(row["CE_Sym"]) if row["CE_Sym"] else None
                            pe_tick = state.get_tick(row["PE_Sym"]) if row["PE_Sym"] else None
                            if ce_tick:
                                row["CE Price"] = ce_tick["ltp"]
                            if pe_tick:
                                row["PE Price"] = pe_tick["ltp"]

                        worker = state.ws_worker
                        if worker and worker.is_alive():
                            worker.subscribe_multiple(symbols_to_sub)

                        st.markdown("##### 📊 Live Chain & Greeks (ATM ± 2)")
                        df_display = pd.DataFrame(mini_chain)[["CE Delta", "CE Price", "Strike", "PE Price", "PE Delta"]]

                        def highlight_atm(row):
                            if row["Strike"] == oi_data["atm_strike"]:
                                return ["background-color: rgba(255, 255, 0, 0.15)"] * len(row)
                            return [""] * len(row)

                        st.dataframe(
                            df_display.style.apply(highlight_atm, axis=1).format({
                                "CE Price": "{:.2f}", "PE Price": "{:.2f}",
                                "CE Delta": "{:.3f}", "PE Delta": "{:.3f}",
                            }),
                            hide_index=True,
                        )

    st.write("---")
    st.subheader("📜 Today's Global Ledger")
    df_history, _ = db_load_today_trades()
    st.dataframe(df_history, width="stretch")


if __name__ == "__main__":
    main()
