#!/usr/bin/env python3
"""
JII70 PRO Terminal — tampilan layaknya aplikasi sekuritas
Jalankan: streamlit run jii70_pro.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="JII70 PRO Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

JII70 = [
    "AADI","ACES","ADMR","ADRO","AKRA","ANTM","ARCI","AVIA","BKSL","BRIS",
    "BRMS","BSDE","BTPS","BUMI","CMRY","CPIN","CTRA","DEWA","DKFT","DSNG",
    "ELSA","ENRG","ERAA","ESSA","EXCL","HEAL","HRTA","HRUM","ICBP","IMPC",
    "INDF","INDY","INKP","INTP","ISAT","ITMG","JPFA","JSMR","KIJA","KLBF",
    "KPIG","LSIP","MAPA","MAPI","MARK","MBMA","MDKA","MEDC","MIKA","MTEL",
    "MYOR","PGAS","PTBA","RAJA","RATU","SIDO","SMGR","SMRA","SRTG","SSIA",
    "TAPG","TCPI","TINS","TKIM","TLKM","TOBA","TPIA","UNTR","UNVR","WIFI",
]

# ═══════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════

def calc_rsi(close: pd.Series, n: int = 14):
    if len(close) < n + 1:
        return None
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return round(float(v), 1) if pd.notna(v) else None

def calc_sma(close: pd.Series, n: int):
    if len(close) < n:
        return None
    return round(float(close.tail(n).mean()), 2)

def calc_ema(close: pd.Series, n: int):
    if len(close) < n:
        return None
    return round(float(close.ewm(span=n, adjust=False).mean().iloc[-1]), 2)

def calc_macd(close: pd.Series):
    if len(close) < 35:
        return None, None, None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return (
        round(float(macd.iloc[-1]), 4),
        round(float(signal.iloc[-1]), 4),
        round(float(hist.iloc[-1]), 4),
    )

def calc_atr(high, low, close, n=14):
    if len(close) < n + 1:
        return None
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    v = atr.iloc[-1]
    return round(float(v), 2) if pd.notna(v) else None

def calc_bb(close, n=20):
    if len(close) < n:
        return None, None, None, None
    sma = close.tail(n).mean()
    std = close.tail(n).std()
    if std == 0 or pd.isna(std):
        return None, None, None, None
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (close.iloc[-1] - lower) / (upper - lower)
    return (
        round(float(sma), 2),
        round(float(upper), 2),
        round(float(lower), 2),
        round(float(pct_b), 2),
    )

def calc_stoch(high, low, close, n=14):
    if len(close) < n:
        return None
    ll = low.tail(n).min()
    hh = high.tail(n).max()
    if hh == ll:
        return None
    k = 100 * (close.iloc[-1] - ll) / (hh - ll)
    return round(float(k), 1)

def calc_mom(close, n=10):
    if len(close) < n + 1:
        return None
    return round(float((close.iloc[-1] / close.iloc[-n-1] - 1) * 100), 2)

# ═══════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════

def generate_signal(row: dict) -> dict:
    """Generate BUY / HOLD / SELL with reasons (multi-factor confluence)."""
    buy_pts = 0
    sell_pts = 0
    buy_why = []
    sell_why = []

    price = row.get("Harga") or 0
    rsi = row.get("RSI")
    macd_h = row.get("MACD_H")
    sma20 = row.get("SMA20")
    ema9 = row.get("EMA9")
    atr = row.get("ATR")
    pe = row.get("PE")
    pb = row.get("PBV")
    stoch = row.get("Stoch")
    pct_b = row.get("%B")
    mom = row.get("Mom10")
    chg = row.get("%Hari") or 0
    vol = row.get("Volume") or 0

    # RSI
    if rsi is not None:
        if rsi < 30:
            buy_pts += 3
            buy_why.append(f"RSI oversold ({rsi}) — potensi bounce")
        elif rsi < 40:
            buy_pts += 2
            buy_why.append(f"RSI rendah ({rsi})")
        elif rsi > 75:
            sell_pts += 3
            sell_why.append(f"RSI overbought ({rsi})")
        elif rsi > 65:
            sell_pts += 1
            sell_why.append(f"RSI tinggi ({rsi})")

    # MACD
    if macd_h is not None:
        if macd_h > 0:
            buy_pts += 2
            buy_why.append("MACD histogram positif (momentum naik)")
        else:
            sell_pts += 2
            sell_why.append("MACD histogram negatif (momentum turun)")

    # Trend SMA20 / EMA9
    if sma20 and price:
        if price > sma20:
            buy_pts += 2
            buy_why.append("Harga di atas SMA20 (uptrend pendek)")
        else:
            sell_pts += 2
            sell_why.append("Harga di bawah SMA20 (downtrend pendek)")
    if ema9 and price:
        if price > ema9:
            buy_pts += 1
        else:
            sell_pts += 1

    # Stochastic
    if stoch is not None:
        if stoch < 20:
            buy_pts += 2
            buy_why.append(f"Stochastic oversold ({stoch})")
        elif stoch > 80:
            sell_pts += 2
            sell_why.append(f"Stochastic overbought ({stoch})")

    # Bollinger %B
    if pct_b is not None:
        if pct_b < 0.2:
            buy_pts += 2
            buy_why.append("Di zona bawah Bollinger (mean reversion)")
        elif pct_b > 0.95:
            sell_pts += 2
            sell_why.append("Di zona atas Bollinger (extended)")

    # Value
    if pe is not None and 0 < pe < 10:
        buy_pts += 2
        buy_why.append(f"PE menarik ({pe})")
    elif pe is not None and pe > 30:
        sell_pts += 1
        sell_why.append(f"PE mahal ({pe})")
    if pb is not None and 0 < pb < 1:
        buy_pts += 2
        buy_why.append(f"PBV diskon ({pb})")

    # Momentum
    if mom is not None:
        if 1 < mom < 10:
            buy_pts += 1
            buy_why.append(f"Momentum 10h positif (+{mom}%)")
        elif mom < -10:
            sell_pts += 1
            sell_why.append(f"Momentum 10h lemah ({mom}%)")

    # Volume interest
    if vol > 50_000_000:
        buy_pts += 1
        buy_why.append("Volume tinggi (minat pasar)")

    # Daily crash caution
    if chg < -5:
        sell_pts += 1
        sell_why.append(f"Koreksi tajam hari ini ({chg}%)")

    # Decision
    net = buy_pts - sell_pts
    if buy_pts >= 6 and net >= 3:
        signal = "BELI"
        conf = min(95, 55 + buy_pts * 4)
        color = "#16a34a"
    elif sell_pts >= 6 and net <= -3:
        signal = "JUAL"
        conf = min(95, 55 + sell_pts * 4)
        color = "#dc2626"
    elif buy_pts >= 4 and net >= 1:
        signal = "AKUMULASI"
        conf = min(80, 45 + buy_pts * 3)
        color = "#65a30d"
    elif sell_pts >= 4 and net <= -1:
        signal = "REDUKSI"
        conf = min(80, 45 + sell_pts * 3)
        color = "#ea580c"
    else:
        signal = "HOLD"
        conf = 50
        color = "#64748b"

    # Suggested levels
    stop = None
    target = None
    if price and atr:
        if signal in ("BELI", "AKUMULASI"):
            stop = round(price - 1.5 * atr, 0)
            target = round(price + 2.5 * atr, 0)
        elif signal in ("JUAL", "REDUKSI"):
            stop = round(price + 1.5 * atr, 0)
            target = round(price - 2.5 * atr, 0)

    return {
        "Sinyal": signal,
        "Confidence": conf,
        "Color": color,
        "BuyPts": buy_pts,
        "SellPts": sell_pts,
        "AlasanBeli": buy_why[:4],
        "AlasanJual": sell_why[:4],
        "Stop": stop,
        "Target": target,
        "ATR": atr,
    }

# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=90, show_spinner=False)
def fetch_ihsg():
    try:
        t = yf.Ticker("^JKSE")
        info = t.info or {}
        hist = t.history(period="10d")
        price = info.get("regularMarketPrice") or info.get("previousClose")
        prev = info.get("previousClose")
        chg = info.get("regularMarketChange")
        chg_pct = info.get("regularMarketChangePercent")
        if price is None and len(hist):
            price = float(hist["Close"].iloc[-1])
        if prev is None and len(hist) >= 2:
            prev = float(hist["Close"].iloc[-2])
        if chg is None and price and prev:
            chg = price - prev
        if chg_pct is None and chg and prev:
            chg_pct = chg / prev * 100
        if not price:
            return {}
        last = hist.iloc[-1] if len(hist) else {}
        return {
            "price": round(float(price), 2),
            "change": round(float(chg), 2) if chg is not None else None,
            "changePct": round(float(chg_pct), 2) if chg_pct is not None else None,
            "open": round(float(info.get("regularMarketOpen") or last.get("Open") or 0), 2) or None,
            "high": round(float(info.get("dayHigh") or info.get("regularMarketDayHigh") or last.get("High") or 0), 2) or None,
            "low": round(float(info.get("dayLow") or info.get("regularMarketDayLow") or last.get("Low") or 0), 2) or None,
            "prev": round(float(prev), 2) if prev else None,
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_one(code: str) -> dict | None:
    try:
        t = yf.Ticker(code + ".JK")
        info = t.info or {}
        hist = t.history(period="4mo")
        if hist.empty:
            return None
        close, high, low = hist["Close"], hist["High"], hist["Low"]
        price = info.get("regularMarketPrice") or info.get("currentPrice") or float(close.iloc[-1])
        prev = info.get("previousClose")
        if prev is None and len(close) >= 2:
            prev = float(close.iloc[-2])
        chg = (price - prev) if price and prev else None
        chg_pct = (chg / prev * 100) if chg and prev else None
        name = (info.get("shortName") or info.get("longName") or code)
        name = name.replace(" Tbk", "").replace(" (Persero)", "").replace("Tbk.", "").strip()

        macd, sig, hist_m = calc_macd(close)
        mid, up, lo, pct_b = calc_bb(close)

        return {
            "Kode": code,
            "Nama": name,
            "Harga": round(float(price), 0) if price else None,
            "%Hari": round(float(chg_pct), 2) if chg_pct is not None else None,
            "Volume": info.get("regularMarketVolume") or info.get("volume"),
            "PE": round(float(info["trailingPE"]), 1) if info.get("trailingPE") else None,
            "PBV": round(float(info["priceToBook"]), 2) if info.get("priceToBook") else None,
            "RSI": calc_rsi(close),
            "SMA20": calc_sma(close, 20),
            "EMA9": calc_ema(close, 9),
            "MACD_H": hist_m,
            "ATR": calc_atr(high, low, close),
            "Stoch": calc_stoch(high, low, close),
            "%B": pct_b,
            "Mom10": calc_mom(close, 10),
            "Sektor": info.get("sector") or "Lainnya",
            "MarketCap": info.get("marketCap"),
        }
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner="Memuat data pasar & sinyal...")
def fetch_universe() -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=14) as ex:
        futs = {ex.submit(fetch_one, c): c for c in JII70}
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    signals = [generate_signal(r) for r in rows]
    for k in ["Sinyal", "Confidence", "Color", "AlasanBeli", "AlasanJual", "Stop", "Target"]:
        df[k] = [s[k] for s in signals]
    # Sort: BELI first, then by confidence
    order = {"BELI": 0, "AKUMULASI": 1, "HOLD": 2, "REDUKSI": 3, "JUAL": 4}
    df["_ord"] = df["Sinyal"].map(order)
    df = df.sort_values(["_ord", "Confidence"], ascending=[True, False]).drop(columns="_ord")
    return df.reset_index(drop=True)


def build_excel(df: pd.DataFrame) -> bytes:
    """Export watchlist + signals to Excel (.xlsx)."""
    out = io.BytesIO()
    export = df.copy()
    # Flatten list columns
    if "AlasanBeli" in export.columns:
        export["AlasanBeli"] = export["AlasanBeli"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else str(x or "")
        )
    if "AlasanJual" in export.columns:
        export["AlasanJual"] = export["AlasanJual"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else str(x or "")
        )
    cols = [c for c in [
        "Kode", "Nama", "Harga", "%Hari", "Volume", "PE", "PBV", "RSI",
        "SMA20", "EMA9", "MACD_H", "ATR", "Stoch", "%B", "Mom10",
        "Sinyal", "Confidence", "Stop", "Target", "AlasanBeli", "AlasanJual", "Sektor"
    ] if c in export.columns]
    export = export[cols]
    # Sheet BELI only
    beli = export[export["Sinyal"] == "BELI"] if "Sinyal" in export.columns else export.head(0)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="Semua JII70", index=False)
        beli.to_excel(writer, sheet_name="Sinyal BELI", index=False)
        # Summary
        if "Sinyal" in export.columns:
            summary = export["Sinyal"].value_counts().rename_axis("Sinyal").reset_index(name="Jumlah")
            summary.to_excel(writer, sheet_name="Ringkasan", index=False)
    return out.getvalue()


# ═══════════════════════════════════════════════════════════
# UI STYLE
# ═══════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0b1220; }
section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
h1,h2,h3,h4 { color: #e2e8f0 !important; }
.block-container { padding-top: 1.2rem; }

.topbar {
  background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
  border: 1px solid #334155; border-radius: 12px;
  padding: 14px 18px; margin-bottom: 14px;
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
}
.brand { font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; }
.brand span { color: #38bdf8; }
.ihsg-box { text-align: right; }
.ihsg-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
.ihsg-price { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #f1f5f9; }
.up { color: #4ade80 !important; }
.down { color: #f87171 !important; }

.card {
  background: #111827; border: 1px solid #1f2937; border-radius: 12px;
  padding: 14px 16px; height: 100%;
}
.card-title { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.card-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem; font-weight: 600; color: #f8fafc; }
.card-sub { font-size: 0.8rem; color: #64748b; margin-top: 4px; }

.signal-pill {
  display: inline-block; padding: 3px 10px; border-radius: 6px;
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.03em;
}
.sig-BELI { background: #14532d; color: #4ade80; }
.sig-AKUMULASI { background: #365314; color: #a3e635; }
.sig-HOLD { background: #334155; color: #94a3b8; }
.sig-REDUKSI { background: #7c2d12; color: #fb923c; }
.sig-JUAL { background: #7f1d1d; color: #f87171; }

.alert-box {
  background: linear-gradient(135deg, #14532d22, #052e16);
  border: 1px solid #166534; border-radius: 12px;
  padding: 12px 16px; margin-bottom: 8px;
}
.alert-code { font-weight: 700; color: #4ade80; font-size: 1.05rem; }
.alert-meta { color: #94a3b8; font-size: 0.8rem; }

div[data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 10px; }
.stMetric { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 8px 12px; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
</style>
"""

def fmt_vol(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    n = float(n)
    if n >= 1e9: return f"{n/1e9:.1f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.0f}K"
    return f"{n:.0f}"

def fmt_rp(n):
    if n is None: return "—"
    return f"Rp {n:,.0f}".replace(",", ".")

# ═══════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════

def page_terminal(df: pd.DataFrame, ihsg: dict):
    st.markdown(CSS, unsafe_allow_html=True)

    # Top bar — native components (hindari HTML mentah di browser)
    left, right = st.columns([2, 2])
    with left:
        st.markdown("## JII70 **PRO** Terminal")
        st.caption("Saham syariah JII70 · Sinyal multifaktor · Yahoo Finance delayed")
    with right:
        if ihsg.get("price"):
            chg_pct = ihsg.get("changePct")
            chg = ihsg.get("change")
            delta = None
            if chg_pct is not None:
                delta = f"{chg_pct:+.2f}%"
                if chg is not None:
                    delta += f" ({chg:+,.2f})"
            st.metric(
                "IHSG (Yahoo delayed)",
                f"{ihsg['price']:,.2f}",
                delta=delta,
            )
            o = ihsg.get("open") or "—"
            h = ihsg.get("high") or "—"
            l = ihsg.get("low") or "—"
            st.caption(f"Open {o} · High {h} · Low {l}")
        else:
            st.warning("IHSG tidak tersedia — klik Refresh")

    # KPI row
    n_beli = int((df["Sinyal"] == "BELI").sum()) if len(df) else 0
    n_akum = int((df["Sinyal"] == "AKUMULASI").sum()) if len(df) else 0
    n_jual = int((df["Sinyal"].isin(["JUAL", "REDUKSI"])).sum()) if len(df) else 0
    n_hold = int((df["Sinyal"] == "HOLD").sum()) if len(df) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe", f"{len(df)} saham")
    c2.metric("Sinyal BELI", n_beli)
    c3.metric("Akumulasi", n_akum)
    c4.metric("Hold", n_hold)
    c5.metric("Jual / Reduksi", n_jual)

    # Notifikasi + Export bar
    alerts_all = df[df["Sinyal"] == "BELI"]
    n_alert = len(alerts_all)

    exp_c1, exp_c2, exp_c3 = st.columns([2, 1, 1])
    with exp_c1:
        if n_alert > 0:
            st.success(f"🔔 **{n_alert} saham** mendapat sinyal **BELI** saat ini — cek daftar di bawah.")
            # Toast sekali per session key
            toast_key = f"toast_beli_{n_alert}_{datetime.now().strftime('%Y%m%d%H')}"
            if toast_key not in st.session_state:
                names = ", ".join(alerts_all["Kode"].head(5).tolist())
                st.toast(f"Sinyal BELI: {names}", icon="🔔")
                st.session_state[toast_key] = True
        else:
            st.info("Tidak ada sinyal BELI kuat saat ini. Pantau AKUMULASI di screener.")
    with exp_c2:
        xlsx = build_excel(df)
        st.download_button(
            "📥 Export Excel",
            data=xlsx,
            file_name=f"JII70_sinyal_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with exp_c3:
        # CSV cepat
        csv_buf = df.drop(columns=[c for c in ["Color", "AlasanBeli", "AlasanJual"] if c in df.columns], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📄 Export CSV",
            data=csv_buf,
            file_name=f"JII70_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ALERTS — saham layak beli
    st.markdown("### 🔔 Alert — Saham Layak Dicermati (BELI)")
    alerts = alerts_all.head(8)
    if alerts.empty:
        st.info("Belum ada saham dengan konfluensi BELI yang kuat saat ini. Cek tab Screener untuk AKUMULASI.")
    else:
        cols = st.columns(min(4, len(alerts)))
        for i, (_, r) in enumerate(alerts.iterrows()):
            with cols[i % len(cols)]:
                reasons = r["AlasanBeli"] if isinstance(r["AlasanBeli"], list) else []
                reason_txt = " · ".join(reasons[:2]) if reasons else "—"
                pct = r["%Hari"]
                delta = f"{pct:+.1f}%" if pct is not None else None
                with st.container(border=True):
                    st.markdown(f"**{r['Kode']}** · `BELI`")
                    st.metric("Harga", f"{r['Harga']:,.0f}" if r["Harga"] else "—", delta=delta)
                    st.caption(f"Conf {r['Confidence']}% · {reason_txt}")
                    st.link_button(
                        f"Stockbit {r['Kode']} ↗",
                        f"https://stockbit.com/symbol/{r['Kode']}",
                        use_container_width=True,
                    )
                    if r.get("Stop") and r.get("Target"):
                        st.caption(f"Stop ~{r['Stop']:,.0f} · Target ~{r['Target']:,.0f}")

    st.divider()

    # Main table
    st.markdown("### 📋 Watchlist JII70")
    st.caption("Klik header kolom untuk sort · Klik kode di kolom link untuk buka Stockbit")

    show = df.copy()
    show["Vol"] = show["Volume"].apply(fmt_vol)
    show["Link"] = show["Kode"].apply(lambda x: f"https://stockbit.com/symbol/{x}")
    show["Alasan"] = show.apply(
        lambda r: " | ".join((r["AlasanBeli"] or [])[:2]) if r["Sinyal"] in ("BELI", "AKUMULASI")
        else (" | ".join((r["AlasanJual"] or [])[:2]) if r["Sinyal"] in ("JUAL", "REDUKSI") else "—"),
        axis=1,
    )

    display = show[[
        "Kode", "Nama", "Harga", "%Hari", "Vol", "PE", "PBV", "RSI",
        "SMA20", "Sinyal", "Confidence", "Stop", "Target", "Alasan", "Link"
    ]].copy()

    def style_signal(val):
        colors = {
            "BELI": "background-color:#14532d;color:#4ade80;font-weight:700",
            "AKUMULASI": "background-color:#365314;color:#a3e635;font-weight:600",
            "HOLD": "background-color:#334155;color:#94a3b8",
            "REDUKSI": "background-color:#7c2d12;color:#fb923c;font-weight:600",
            "JUAL": "background-color:#7f1d1d;color:#f87171;font-weight:700",
        }
        return colors.get(val, "")

    def style_pct(val):
        if pd.isna(val): return ""
        if val > 0: return "color:#4ade80;font-weight:600"
        if val < 0: return "color:#f87171;font-weight:600"
        return ""

    styled = (
        display.style
        .map(style_signal, subset=["Sinyal"])
        .map(style_pct, subset=["%Hari"])
        .format({
            "Harga": "{:,.0f}",
            "%Hari": "{:+.2f}",
            "PE": "{:.1f}",
            "PBV": "{:.2f}",
            "RSI": "{:.1f}",
            "SMA20": "{:,.0f}",
            "Confidence": "{:.0f}%",
            "Stop": "{:,.0f}",
            "Target": "{:,.0f}",
        }, na_rep="—")
    )

    st.dataframe(
        styled,
        use_container_width=True,
        height=480,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Stockbit", display_text="Buka ↗"),
            "Confidence": st.column_config.ProgressColumn("Conf", min_value=0, max_value=100, format="%d%%"),
            "Alasan": st.column_config.TextColumn("Alasan utama", width="large"),
            "Kode": st.column_config.TextColumn("Kode", width="small"),
        },
    )

    st.caption(f"Update {datetime.now().strftime('%d %b %Y %H:%M WIB')} · Data Yahoo Finance (delayed) · Bukan saran investasi")


def page_screener(df: pd.DataFrame):
    st.markdown(CSS, unsafe_allow_html=True)
    st.subheader("🔍 Screener & Filter")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sigs = st.multiselect("Sinyal", ["BELI", "AKUMULASI", "HOLD", "REDUKSI", "JUAL"],
                              default=["BELI", "AKUMULASI", "HOLD", "REDUKSI", "JUAL"])
    with c2:
        q = st.text_input("Cari kode/nama")
    with c3:
        rsi_rng = st.slider("RSI", 0, 100, (0, 100))
    with c4:
        pe_max = st.number_input("PE max (0=off)", 0.0, 100.0, 0.0)

    view = df[df["Sinyal"].isin(sigs)].copy()
    if q:
        ql = q.lower()
        view = view[view["Kode"].str.lower().str.contains(ql) | view["Nama"].str.lower().str.contains(ql)]
    view = view[(view["RSI"].fillna(50) >= rsi_rng[0]) & (view["RSI"].fillna(50) <= rsi_rng[1])]
    if pe_max > 0:
        view = view[view["PE"].isna() | (view["PE"] <= pe_max)]

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        st.write(f"**{len(view)}** saham cocok")
    with sc2:
        if len(view):
            st.download_button(
                "📥 Excel hasil filter",
                data=build_excel(view),
                file_name=f"JII70_filter_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


    for _, r in view.iterrows():
        with st.expander(f"{r['Kode']}  ·  {r['Harga']:,.0f}  ·  {r['Sinyal']} ({r['Confidence']}%)", expanded=False):
            a, b, c = st.columns([2, 2, 1])
            with a:
                st.markdown("**Alasan BELI**")
                for x in (r["AlasanBeli"] or ["—"]):
                    st.markdown(f"- {x}")
            with b:
                st.markdown("**Alasan JUAL / waspada**")
                for x in (r["AlasanJual"] or ["—"]):
                    st.markdown(f"- {x}")
            with c:
                st.metric("RSI", r["RSI"])
                st.metric("PE", r["PE"] if r["PE"] else "—")
                if r.get("Stop"):
                    st.caption(f"Stop ~{r['Stop']:,.0f}")
                if r.get("Target"):
                    st.caption(f"Target ~{r['Target']:,.0f}")
            st.link_button("Buka di Stockbit ↗", f"https://stockbit.com/symbol/{r['Kode']}")


def page_calculator(df: pd.DataFrame):
    st.markdown(CSS, unsafe_allow_html=True)
    st.subheader("🧮 Kalkulator Saham")

    tab1, tab2, tab3 = st.tabs(["Position Size", "Profit / Loss", "Break-even + Lot"])

    with tab1:
        st.markdown("**Hitung jumlah lot sesuai risiko** (aturan trader pro: risk per trade 0.5%–2%)")
        c1, c2 = st.columns(2)
        with c1:
            modal = st.number_input("Modal (Rp)", min_value=1_000_000, value=100_000_000, step=1_000_000, format="%d")
            risk_pct = st.slider("% Risiko per trade", 0.25, 5.0, 1.0, 0.25)
            entry = st.number_input("Harga entry", min_value=1.0, value=1000.0, step=1.0)
        with c2:
            stop = st.number_input("Harga stop loss", min_value=1.0, value=950.0, step=1.0)
            target = st.number_input("Harga target (opsional)", min_value=0.0, value=1100.0, step=1.0)

        risk_rp = modal * (risk_pct / 100)
        jarak = abs(entry - stop)
        if jarak > 0:
            lembar = int(risk_rp / jarak)
            lot = lembar // 100
            nilai = lot * 100 * entry
            st.success(f"Lot maksimal: **{lot:,} lot** ({lot*100:,} lembar) · Nilai belanja ≈ {fmt_rp(nilai)}")
            st.write(f"Risiko uang: {fmt_rp(risk_rp)} · Jarak stop: Rp {jarak:,.0f}/lembar")
            if target > 0 and entry != stop:
                rr = abs(target - entry) / jarak
                st.write(f"Risk–Reward: **1 : {rr:.2f}**")
                if rr < 1.5:
                    st.warning("RR di bawah 1.5 — banyak trader pro melewati setup ini.")
        else:
            st.error("Entry dan stop tidak boleh sama.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            buy = st.number_input("Harga beli", min_value=1.0, value=1000.0, key="pl_buy")
            lots = st.number_input("Jumlah lot", min_value=1, value=10, key="pl_lot")
        with c2:
            sell = st.number_input("Harga jual / harga sekarang", min_value=1.0, value=1050.0, key="pl_sell")
            fee_pct = st.number_input("Fee beli+jual total (%)", min_value=0.0, value=0.3, step=0.05, key="pl_fee")
        lembar = int(lots * 100)
        gross = (sell - buy) * lembar
        fee = (buy + sell) * lembar * (fee_pct / 100)
        net = gross - fee
        color = "normal" if net >= 0 else "inverse"
        st.metric("Profit/Loss bersih", fmt_rp(net), delta=f"{(net/(buy*lembar)*100):+.2f}%")
        st.caption(f"Gross {fmt_rp(gross)} · Estimasi fee {fmt_rp(fee)}")

    with tab3:
        st.markdown("**Berapa lot agar fee tertutup + target profit**")
        c1, c2 = st.columns(2)
        with c1:
            entry2 = st.number_input("Entry", value=1000.0, key="be_e")
            target_pct = st.number_input("Target profit %", value=2.0, key="be_t")
        with c2:
            fee2 = st.number_input("Fee total %", value=0.3, key="be_f")
            modal2 = st.number_input("Modal tersedia", value=20_000_000, step=100_000, key="be_m", format="%d")
        need_move = fee2 + target_pct
        st.write(f"Harga perlu bergerak minimal **{need_move:.2f}%** agar fee + target tercapai.")
        max_lot = int(modal2 // (entry2 * 100)) if entry2 > 0 else 0
        st.write(f"Lot maksimal dengan modal ini: **{max_lot:,} lot**")

    # Quick pick from universe
    if len(df):
        st.divider()
        st.markdown("**Isi cepat dari sinyal BELI**")
        picks = df[df["Sinyal"] == "BELI"]["Kode"].tolist()[:10]
        if picks:
            st.write(" · ".join([f"[{c}](https://stockbit.com/symbol/{c})" for c in picks]))


def page_detail(df: pd.DataFrame):
    st.markdown(CSS, unsafe_allow_html=True)
    st.subheader("📄 Detail Saham + Sinyal")
    codes = df["Kode"].tolist() if len(df) else JII70
    code = st.selectbox("Pilih kode JII70", codes)
    row = df[df["Kode"] == code]
    if row.empty:
        st.warning("Data belum tersedia, coba refresh.")
        return
    r = row.iloc[0]

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Harga", f"{r['Harga']:,.0f}" if r["Harga"] else "—", delta=f"{r['%Hari']:+.2f}%" if r["%Hari"] is not None else None)
    h2.metric("Sinyal", r["Sinyal"], delta=f"Conf {r['Confidence']}%")
    h3.metric("RSI", r["RSI"] if r["RSI"] else "—")
    h4.metric("ATR", r["ATR"] if r["ATR"] else "—")

    a, b = st.columns(2)
    with a:
        st.markdown("#### ✅ Alasan mendukung BELI")
        for x in (r["AlasanBeli"] or ["Tidak ada alasan beli kuat"]):
            st.markdown(f"- {x}")
        if r.get("Stop") and r.get("Target"):
            st.info(f"Saran level (ATR-based): Stop **{r['Stop']:,.0f}** · Target **{r['Target']:,.0f}**")
    with b:
        st.markdown("#### ⚠️ Alasan JUAL / waspada")
        for x in (r["AlasanJual"] or ["Tidak ada alasan jual kuat"]):
            st.markdown(f"- {x}")
        st.markdown(f"""
| Indikator | Nilai |
|-----------|------|
| PE | {r['PE'] if r['PE'] else '—'} |
| PBV | {r['PBV'] if r['PBV'] else '—'} |
| SMA20 | {r['SMA20'] if r['SMA20'] else '—'} |
| EMA9 | {r['EMA9'] if r['EMA9'] else '—'} |
| Stoch | {r['Stoch'] if r['Stoch'] else '—'} |
| %B Boll | {r['%B'] if r['%B'] else '—'} |
| Mom 10h | {r['Mom10'] if r['Mom10'] is not None else '—'}% |
| Volume | {fmt_vol(r['Volume'])} |
""")

    st.link_button(f"Analisis {code} di Stockbit ↗", f"https://stockbit.com/symbol/{code}", use_container_width=True)
    st.caption("Sinyal multifaktor (RSI, MACD, MA, Stochastic, Bollinger, PE/PBV, volume). Bukan rekomendasi final.")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.markdown("### 📊 JII70 PRO")
        menu = st.radio(
            "Menu",
            ["Terminal", "Screener", "Kalkulator", "Detail Saham"],
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("Data: Yahoo Finance (delayed ±15 mnt)\nNotif toast saat ada BELI\nExport Excel di Terminal")
        st.divider()
        st.markdown("""
**Legenda sinyal**
- **BELI** — konfluensi kuat
- **AKUMULASI** — cenderung positif
- **HOLD** — netral
- **REDUKSI** — waspada
- **JUAL** — konfluensi negatif

Stop/Target memakai **1.5× / 2.5× ATR**.
        """)

    ihsg = fetch_ihsg()
    df = fetch_universe()

    if df.empty:
        st.error("Gagal memuat data saham. Periksa koneksi internet lalu klik Refresh.")
        return

    if menu == "Terminal":
        page_terminal(df, ihsg)
    elif menu == "Screener":
        page_screener(df)
    elif menu == "Kalkulator":
        page_calculator(df)
    else:
        page_detail(df)

    st.markdown("---")
    st.caption(
        "⚠️ Edukasi saja. Bukan saran investasi / ajakan transaksi. "
        "Selalu DYOR dan sesuaikan dengan profil risiko Anda."
    )


if __name__ == "__main__":
    main()
