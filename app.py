import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import itertools
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AltutBot Dashboard", layout="wide", page_icon="📈")
st.title("📈 AltutBot Momentum Dashboard")

# --- AYARLAR ---
STOX_FILE = 'top_endeks_hisseleri.txt'
DATA_CACHE_FILE = 'bist_data_cache.pkl'
START_CAPITAL = 100000
COMMISSION_RATE = 0.002 
REBALANCE_FREQ = 'MS'  
STOP_LOSS_RATE = 0.10

@st.cache_data(ttl=3600)
def get_tickers_from_file(file_path):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [t.strip().upper() + ".IS" for t in f.read().splitlines() if t.strip()]

@st.cache_data(ttl=3600)
def load_data(tickers):
    sim_start = (datetime.now() - timedelta(days=365)).replace(day=1)
    download_start = (sim_start - timedelta(days=400)).strftime('%Y-%m-%d')
    if os.path.exists(DATA_CACHE_FILE): 
        return pd.read_pickle(DATA_CACHE_FILE)
    data = yf.download(tickers, start=download_start, auto_adjust=True)
    data.to_pickle(DATA_CACHE_FILE)
    return data

def run_optimization():
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        st.error("Hisse listesi bulunamadı!")
        return None, None, None
        
    all_data = load_data(tickers)
    
    # Check for MultiIndex to safely get Close prices
    if isinstance(all_data.columns, pd.MultiIndex):
        try:
            raw_data = all_data['Close'].dropna(axis=1, how='all')
        except KeyError:
            raw_data = all_data.xs('Close', axis=1, level=0).dropna(axis=1, how='all')
    else:
        raw_data = all_data
        
    available_tickers = raw_data.columns.tolist()
    sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1)
    periods = pd.date_range(start=sim_start_date, end=datetime.now(), freq=REBALANCE_FREQ)
    if periods[-1].date() < datetime.now().date():
        periods = periods.append(pd.DatetimeIndex([pd.Timestamp(datetime.now().date())]))
        
    w_range = np.arange(0.05, 0.16, 0.05) 
    m_range = np.arange(0.10, 0.31, 0.10) 
    y_range = np.arange(0.50, 0.81, 0.15) 
    
    ma_pool = [5, 10, 20, 50, 100, 200]
    ma_combos = list(itertools.combinations(ma_pool, 4))
    
    mx_w = raw_data.pct_change(5)
    mx_m = raw_data.pct_change(21)
    mx_y = raw_data.pct_change(250)
    valid_mask = raw_data.notna().cumsum() >= 250
    
    precalc_ma = {p: raw_data.rolling(p).mean().values for p in ma_pool}
    period_indices = [raw_data.index.get_indexer([p], method='ffill')[0] for p in periods]
    
    num_periods = len(period_indices) - 1
    period_mins = np.zeros((num_periods, raw_data.shape[1]))
    for i in range(num_periods):
        idx_s, idx_e = period_indices[i], period_indices[i+1]
        period_mins[i, :] = np.nanmin(raw_data.values[idx_s:idx_e+1, :], axis=0)
        
    np_close = raw_data.values
    np_w = mx_w.values
    np_m = mx_m.values
    np_y = mx_y.values
    np_valid = valid_mask.values
    
    results = []
    best_profit_so_far = 0.0
    
    progress_text = "Vektörize Optimizasyon Gerçekleştiriliyor..."
    my_bar = st.progress(0, text=progress_text)
    total_iters = len(ma_combos)
    
    for c_idx, combo in enumerate(ma_combos):
        m1, m2, m3, m4 = combo
        arr_m1 = precalc_ma[m1]
        arr_m2 = precalc_ma[m2]
        arr_m3 = precalc_ma[m3]
        arr_m4 = precalc_ma[m4]
        
        np_ma_cond = (np_close > arr_m1) & (arr_m1 > arr_m2) & (arr_m2 > arr_m3) & (arr_m3 > arr_m4)
        
        for w_val in w_range:
            for m_val in m_range:
                for y_val in y_range:
                    temp_balance = START_CAPITAL
                    for i in range(len(period_indices) - 1):
                        idx_start = period_indices[i]
                        idx_end = period_indices[i+1]
                        
                        curr_w = np_w[idx_start, :]
                        curr_m = np_m[idx_start, :]
                        curr_y = np_y[idx_start, :]
                        is_valid = np_valid[idx_start, :]
                        curr_ma_cond = np_ma_cond[idx_start, :]
                        
                        selection = is_valid & curr_ma_cond & (curr_w >= w_val) & (curr_m >= m_val) & (curr_y >= y_val)
                        num_picks = np.sum(selection)
                        
                        if num_picks > 0:
                            entry_prices = np_close[idx_start, selection]
                            exit_prices = np_close[idx_end, selection]
                            min_prices = period_mins[i, selection]
                            
                            stop_prices = entry_prices * (1 - STOP_LOSS_RATE)
                            hit_stop = min_prices <= stop_prices
                            actual_exit_prices = np.where(hit_stop, stop_prices, exit_prices)
                            
                            cash_per_stock = temp_balance / num_picks
                            lots = np.floor(cash_per_stock / entry_prices)
                            cost = lots * entry_prices
                            revenue = lots * actual_exit_prices * (1 - COMMISSION_RATE)
                            unused_cash = cash_per_stock - cost * (1 + COMMISSION_RATE)
                            temp_balance = np.sum(revenue) + np.sum(unused_cash)
                            
                    if temp_balance > best_profit_so_far:
                        best_profit_so_far = temp_balance
                    
                    results.append({'MA_Combo': combo, 'W': w_val, 'M': m_val, 'Y': y_val, 'Final': temp_balance})
        
        my_bar.progress((c_idx + 1) / total_iters, text=f"Aranıyor... Yeni Rekor Bakiye: {best_profit_so_far:,.0f} TL")
        
    my_bar.empty()
    best_series = pd.DataFrame(results).loc[pd.DataFrame(results)['Final'].idxmax()]
    return best_series, raw_data, available_tickers


colA, colB = st.columns([1, 4])
with colA:
    if st.button("🚀 Tarama ve Analizi Başlat!", use_container_width=True, type="primary"):
        st.session_state['run'] = True

if st.session_state.get('run', False):
    with st.spinner("1755 devasa ihtimal matris üzerinden analiz ediliyor..."):
        best, raw_data, available_tickers = run_optimization()
        
    if best is not None:
        bm1, bm2, bm3, bm4 = best['MA_Combo']
        
        st.success("✅ Algoritma en uygun geçmiş stratejisini başarıyla buldu.")
        
        st.markdown(f"### 🏆 Geçen Senenin En Çok Kazandıran Stratejisi")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Bakiye", f"{best['Final']:,.0f} TL", "Başlangıç: 100K TL")
        c2.metric("Hareketli Ortalamalar (Trend)", f"MA({bm1} > {bm2} > {bm3} > {bm4})")
        c3.metric("Momentum Eşikleri", f"Hafta> %{best['W']*100:.0f} | Ay> %{best['M']*100:.0f} | Yıl> %{best['Y']*100:.0f}")
        c4.metric("Risk Sınırı", f"%10 Stop-Loss")
        c5.metric("Komisyon ve Yenileme", f"Binde 2 | Her Ay Başı")
        
        st.divider()
        st.markdown("## 🛒 Bugün Olsa Ne Alırdı?")
        st.markdown("Bot yukarıdaki bulduğu en mükemmel geçmiş ayarları kullanarak **Bugünkü Kapanış Fiyatlarına ve Hacimlerine** göre filtre yaptı. Sonuç aşağıda:")
        
        try:
            last_date = raw_data.index[-1]
            
            ma_cond_final = (raw_data > raw_data.rolling(bm1).mean()) & \
                            (raw_data.rolling(bm1).mean() > raw_data.rolling(bm2).mean()) & \
                            (raw_data.rolling(bm2).mean() > raw_data.rolling(bm3).mean()) & \
                            (raw_data.rolling(bm3).mean() > raw_data.rolling(bm4).mean())
                            
            w_cond = (raw_data.iloc[-1] / raw_data.iloc[-5] - 1) >= best['W']
            m_cond = (raw_data.iloc[-1] / raw_data.iloc[-21] - 1) >= best['M']
            y_cond = (raw_data.iloc[-1] / raw_data.iloc[-250] - 1) >= best['Y']
            
            bugun_alinacak_hisseler = []
            for t in available_tickers:
                if len(raw_data[t].dropna()) >= 250:
                    try:
                        is_ma_valid = bool(ma_cond_final.loc[last_date, t])
                    except:
                        is_ma_valid = False
                        
                    if is_ma_valid and w_cond[t] and m_cond[t] and y_cond[t]:
                        val = raw_data[t].iloc[-1]
                        if not pd.isna(val):
                            ticker_name = str(t[1] if isinstance(t, tuple) else t).replace('.IS', '')
                            bugun_alinacak_hisseler.append({
                                'Potansiyel Hisse': ticker_name,
                                'Giriş Kapanış Fiyatı': f"{val:.2f} TL"
                            })
                            
            if bugun_alinacak_hisseler:
                df_picks = pd.DataFrame(bugun_alinacak_hisseler)
                df_picks.index = df_picks.index + 1
                st.dataframe(df_picks, use_container_width=True)
            else:
                st.warning("⚠️ Orijinal Algoritma Kararı: BIST piyasa şartları şu an en iyi stratejiye uygun değil. Tüm filtrelerden aynı anda geçebilen 'Trendi Mükemmel' hiçbir hisse bulunamadı. Tamamen Nakitte Bekleniyor.")
        except Exception as e:
            st.error(f"Filtreler günümüze uygulanırken hata oluştu: {e}")
            
