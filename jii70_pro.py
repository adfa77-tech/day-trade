#!/usr/bin/env python3
"""
JII70 PRO Terminal — Tampilan Sekuritas Modern (Glassmorphism & 3D Top Menu)
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
    initial_sidebar_state="collapsed",
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

    if rsi is not None:
        if rsi < 30:
            buy_pts += 3
            buy_why.append(f"RSI oversold ({rsi})")
        elif rsi < 40:
            buy_pts += 2
            buy_why.append(f"RSI rendah ({rsi})")
        elif rsi > 75:
            sell_pts += 3
            sell_why.append(f"RSI overbought ({rsi})")
        elif rsi > 65:
            sell_pts += 1
            sell_why.append(f"RSI tinggi ({rsi})")

    if macd_h is not None:
        if macd_h > 0:
            buy_pts += 2
            buy_why.append("MACD positif")
        else:
            sell_pts += 2
            sell_why.append("MACD negatif")

    if sma20 and price:
        if price > sma20:
            buy_pts += 2
            buy_why.append("Diatas SMA20")
        else:
            sell_pts += 2
            sell_why.append("Dibawah SMA20")
    if ema9 and price:
        if price > ema9:
            buy_pts += 1
        else:
            sell_pts += 1

    if stoch is not None:
        if stoch < 20:
            buy_pts += 2
            buy_why.append(f"Stoch oversold")
        elif stoch > 80:
            sell_pts += 2
            sell_why.append(f"Stoch overbought")

    if pct_b is not None:
        if pct_b < 0.2:
            buy_pts += 2
            buy_why.append("Zona bawah Bollinger")
        elif pct_b > 0.95:
            sell_pts += 2
            sell_why.append("Zona atas Bollinger")

    if pe is not None and 0 < pe < 10:
        buy_pts += 2
        buy_why.append(f"PE menarik ({pe})")
    elif pe is not None and pe > 30:
        sell_pts += 1
        sell_why.append(f"PE mahal ({pe})")
    if pb is not None and 0 < pb < 1:
        buy_pts += 2
        buy_why.append(f"PBV diskon ({pb})")

    if mom is not None:
        if 1 < mom < 10:
            buy_pts += 1
            buy_why.append(f"Momentum positif")
        elif mom < -10:
            sell_pts += 1
            sell_why.append(f"Momentum lemah")

    if vol > 50_000_000:
        buy_pts += 1
        buy_why.append("Volume tinggi")

    if chg < -5:
        sell_pts += 1
        sell_why.append(f"Koreksi tajam")

    net = buy_pts - sell_pts
    if buy_pts >= 6 and net >= 3:
        signal = "BELI"
        conf = min(95, 55 + buy_pts * 4)
    elif sell_pts >= 6 and net <= -3:
        signal = "JUAL"
        conf = min(95, 55 + sell_pts * 4)
    elif buy_pts >= 4 and net >= 1:
        signal = "AKUMULASI"
        conf = min(80, 45 + buy_pts * 3)
    elif sell_pts >= 4 and net <= -1:
        signal = "REDUKSI"
        conf = min(80, 45 + sell_pts * 3)
    else:
        signal = "HOLD"
        conf = 50

    stop, target = None, None
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
        "AlasanBeli": buy_why[:4],
        "AlasanJual": sell_why[:4],
        "Stop": stop,
        "Target": target,
        "ATR": atr,
    }

# ═══════════════════════════════════════════════════════════
# DATA FETCHERS
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
            "high": round(float(info.get("dayHigh") or last.get("High") or 0), 2) or None,
            "low": round(float(info.get("dayLow") or last.get("Low") or 0), 2) or None,
        }
    except Exception:
        return {}

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

        _, _, hist_m = calc_macd(close)
        _, _, _, pct_b = calc_bb(close)

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
        }
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner="Memuat data pasar & sinyal JII70...")
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
    for k in ["Sinyal", "Confidence", "AlasanBeli", "AlasanJual", "Stop", "Target", "ATR"]:
        df[k] = [s[k] for s in signals]
    order = {"BELI": 0, "AKUMULASI": 1, "HOLD": 2, "REDUKSI": 3, "JUAL": 4}
    df["_ord"] = df["Sinyal"].map(order)
    df = df.sort_values(["_ord", "Confidence"], ascending=[True, False]).drop(columns="_ord")
    return df.reset_index(drop=True)

def build_excel(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    export = df.copy()
    if "AlasanBeli" in export.columns:
        export["AlasanBeli"] = export["AlasanBeli"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x or ""))
    if "AlasanJual" in export.columns:
        export["AlasanJual"] = export["AlasanJual"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x or ""))
    beli = export[export["Sinyal"] == "BELI"] if "Sinyal" in export.columns else export.head(0)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="Semua JII70", index=False)
        beli.to_excel(writer, sheet_name="Sinyal BELI", index=False)
    return out.getvalue()

# ═══════════════════════════════════════════════════════════
# GLASSMORPHISM & 3D STYLING (CSS)
# ═══════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at 10% 10%, #0f172a 0%, #090d16 100%);
    color: #f1f5f9;
}

/* Sembunyikan sidebar bawaan agar bersih di atas */
section[data-testid="stSidebar"] {
    display: none !important;
}

/* Efek Kaca & 3D untuk Navbar Atas / Container Utama */
.glass-nav {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 14px 20px;
    box-shadow: 0 12px 32px 0 rgba(0, 0, 0, 0.37), inset 0 1px 1px 0 rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}

.glass-card {
    background: rgba(30, 41, 59, 0.55);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
}

/* Tombol / Pilihan Menu Atas */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.metric-font {
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stDataFrame"] {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
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
    # Top Summary & IHSG Banner
    c_title, c_ihsg = st.columns([2, 2])
    with c_title:
        st.markdown("### ⚡ JII70 **PRO Terminal**")
        st.caption("Realtime Analysis & Multifactor Signal Engine")
    with c_ihsg:
        if ihsg.get("price"):
            chg_pct = ihsg.get("changePct", 0)
            chg = ihsg.get("change", 0)
            st.metric("IHSG Index", f"{ihsg['price']:,.2f}", delta=f"{chg_pct:+.2f}% ({chg:+,.2f})")

    # KPI Cards Row
    n_beli = int((df["Sinyal"] == "BELI").sum()) if len(df) else 0
    n_akum = int((df["Sinyal"] == "AKUMULASI").sum()) if len(df) else 0
    n_jual = int((df["Sinyal"].isin(["JUAL", "REDUKSI"])).sum()) if len(df) else 0
    n_hold = int((df["Sinyal"] == "HOLD").sum()) if len(df) else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Universe", f"{len(df)} Saham")
    k2.metric("Sinyal BELI", n_beli)
    k3.metric("Akumulasi", n_akum)
    k4.metric("Hold", n_hold)
    k5.metric("Jual / Reduksi", n_jual)

    st.markdown("---")

    # Alert & Export Panel
    alerts_all = df[df["Sinyal"] == "BELI"]
    ac1, ac2 = st.columns([3, 1])
    with ac1:
        if len(alerts_all) > 0:
            st.success(f"🔔 Ditemukan **{len(alerts_all)} saham** dengan konfluensi sinyal **BELI** kuat.")
        else:
            st.info("Belum ada sinyal BELI kuat saat ini. Pantau menu Screener untuk akumulasi.")
    with ac2:
        xlsx = build_excel(df)
        st.download_button(
            "📥 Export Excel",
            data=xlsx,
            file_name=f"JII70_Terminal_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Watchlist Main Table
    st.markdown("#### 📋 Watchlist & Sinyal JII70")
    show = df.copy()
    show["Vol"] = show["Volume"].apply(fmt_vol)
    show["Link Stockbit"] = show["Kode"].apply(lambda x: f"https://stockbit.com/symbol/{x}")
    show["Alasan Utama"] = show.apply(
        lambda r: " | ".join((r["AlasanBeli"] or [])[:2]) if r["Sinyal"] in ("BELI", "AKUMULASI")
        else (" | ".join((r["AlasanJual"] or [])[:2]) if r["Sinyal"] in ("JUAL", "REDUKSI") else "—"),
        axis=1,
    )

    display = show[[
        "Kode", "Nama", "Harga", "%Hari", "Vol", "PE", "PBV", "RSI",
        "SMA20", "Sinyal", "Confidence", "Stop", "Target", "Alasan Utama", "Link Stockbit"
    ]].copy()

    def style_signal(val):
        colors = {
            "BELI": "background-color:#065f46;color:#34d399;font-weight:700",
            "AKUMULASI": "background-color:#365314;color:#bef264;font-weight:600",
            "HOLD": "background-color:#334155;color:#94a3b8",
            "REDUKSI": "background-color:#7c2d12;color:#fdba74;font-weight:600",
            "JUAL": "background-color:#7f1d1d;color:#fca5a5;font-weight:700",
        }
        return colors.get(val, "")

    styled = (
        display.style
        .map(style_signal, subset=["Sinyal"])
        .format({
            "Harga": "{:,.0f}",
            "%Hari": "{:+.2f}%",
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
        height=450,
        hide_index=True,
        column_config={
            "Link Stockbit": st.column_config.LinkColumn("Stockbit", display_text="Buka ↗"),
            "Confidence": st.column_config.ProgressColumn("Conf", min_value=0, max_value=100, format="%d%%"),
        }
    )

def page_screener(df: pd.DataFrame):
    st.markdown("### 🔍 Advanced Stock Screener")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sigs = st.multiselect("Sinyal Filter", ["BELI", "AKUMULASI", "HOLD", "REDUKSI", "JUAL"], default=["BELI", "AKUMULASI"])
    with c2:
        q = st.text_input("Cari Kode / Nama")
    with c3:
        rsi_rng = st.slider("Rentang RSI", 0, 100, (0, 100))
    with c4:
        pe_max = st.number_input("Maksimal PE", 0.0, 100.0, 0.0)

    view = df[df["Sinyal"].isin(sigs)].copy()
    if q:
        view = view[view["Kode"].str.lower().str.contains(q.lower()) | view["Nama"].str.lower().str.contains(q.lower())]
    view = view[(view["RSI"].fillna(50) >= rsi_rng[0]) & (view["RSI"].fillna(50) <= rsi_rng[1])]
    if pe_max > 0:
        view = view[view["PE"].isna() | (view["PE"] <= pe_max)]

    st.write(f"Menampilkan **{len(view)}** saham sesuai kriteria.")
    for _, r in view.iterrows():
        with st.expander(f"{r['Kode']} — {r['Nama']} | Harga: Rp {r['Harga']:,.0f} | Sinyal: {r['Sinyal']} ({r['Confidence']}%)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Alasan Buy / Pendukung:**")
                for x in (r["AlasanBeli"] or ["-"]):
                    st.markdown(f"- {x}")
            with col2:
                st.markdown("**Level Teknis:**")
                st.write(f"RSI: {r['RSI']} | PE: {r['PE']} | PBV: {r['PBV']}")
                st.write(f"Stop Loss: ~{r.get('Stop', '—')} | Target: ~{r.get('Target', '—')}")

def page_calculator(df: pd.DataFrame):
    st.markdown("### 🧮 Kalkulator Risiko & Ukuran Posisi (Position Sizing)")
    c1, c2 = st.columns(2)
    with c1:
        modal = st.number_input("Total Modal (Rp)", value=50_000_000, step=1_000_000)
        risk_pct = st.slider("Risiko per Trade (%)", 0.5, 3.0, 1.0)
        entry = st.number_input("Harga Entry", value=1000.0)
    with c2:
        stop = st.number_input("Harga Stop Loss", value=950.0)
        target = st.number_input("Harga Target Take Profit", value=1100.0)

    risk_rp = modal * (risk_pct / 100)
    jarak = abs(entry - stop)
    if jarak > 0:
        lembar = int(risk_rp / jarak)
        lot = lembar // 100
        st.success(from_calc := f"Rekomendasi Alokasi: **{lot:,} Lot** ({lembar:,} lembar saham) | Risiko Dana: {fmt_rp(risk_rp)}")
    else:
        st.warning("Harga entry dan stop loss tidak boleh sama.")

def page_detail(df: pd.DataFrame):
    st.markdown("### 📄 Analisis Mendalam Saham")
    codes = df["Kode"].tolist() if len(df) else JII70
    code = st.selectbox("Pilih Emiten JII70", codes)
    row = df[df["Kode"] == code]
    if not row.empty:
        r = row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Harga Terkini", f"Rp {r['Harga']:,.0f}", f"{r['%Hari']:+.2f}%")
        c2.metric("Sinyal", r['Sinyal'], f"Conf: {r['Confidence']}%")
        c3.metric("RSI (14)", r['RSI'])
        c4.metric("ATR Volatility", r['ATR'])

        st.markdown(f"**Alasan Utama Sinyal {r['Sinyal']}:**")
        for b in r["AlasanBeli"]:
            st.markdown(f"✅ {b}")
        for s in r["AlasanJual"]:
            st.markdown(f"⚠️ {s}")

# ═══════════════════════════════════════════════════════════
# MAIN CONTROLLER (TOP NAVBAR 3D & GLASS)
# ═══════════════════════════════════════════════════════════

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    ihsg = fetch_ihsg()
    df = fetch_universe()

    if df.empty:
        st.error("Gagal memuat data dari Yahoo Finance. Periksa koneksi internet Anda.")
        return

    # --- NAVBAR UTAMA DI BAGIAN ATAS (Menggantikan Sidebar) ---
    st.markdown('<div class="glass-nav">', unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([2, 1, 1, 1, 1])

    with nav_col1:
        st.markdown("### 📈 **JII70 PRO** <span style='font-size:0.7rem; color:#38bdf8;'>SEC-TERMINAL</span>", unsafe_allow_html=True)

    # Menggunakan session state untuk navigasi tab atas
    if 'active_menu' not in st.session_state:
        st.session_state.active_menu = "Terminal"

    # Tombol Menu Navigasi Horizontal Atas
    with nav_col2:
        if st.button("📊 Terminal", use_container_width=True):
            st.session_state.active_menu = "Terminal"
    with nav_col3:
        if st.button("🔍 Screener", use_container_width=True):
            st.session_state.active_menu = "Screener"
    with nav_col4:
        if st.button("🧮 Kalkulator", use_container_width=True):
            st.session_state.active_menu = "Kalkulator"
    with nav_col5:
        if st.button("📄 Detail", use_container_width=True):
            st.session_state.active_menu = "Detail Saham"

    st.markdown('</div>', unsafe_allow_html=True)

    # Render Halaman Berdasarkan Pilihan Menu Atas
    menu = st.session_state.active_menu
    if menu == "Terminal":
        page_terminal(df, ihsg)
    elif menu == "Screener":
        page_screener(df)
    elif menu == "Kalkulator":
        page_calculator(df)
    else:
        page_detail(df)

    st.markdown("---")
    st.caption("⚠️ Disclaimer: Alat analisis teknikal otomatis untuk edukasi pasar modal syariah. Selalu lakukan riset mandiri (DYOR) sebelum bertransaksi.")

if __name__ == "__main__":
    main()
