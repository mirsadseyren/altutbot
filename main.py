import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import os
import numpy as np

warnings.filterwarnings('ignore')

# --- AYARLAR ---
STOX_FILE = 'top_endeks_hisseleri.txt'
DATA_CACHE_FILE = 'bist_data_cache.pkl'
START_CAPITAL = 100000
COMMISSION_RATE = 0.002 
REBALANCE_FREQ = 'MS'  
STOP_LOSS_RATE = 0.10 # %10 Stop Loss

def get_tickers_from_file(file_path):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [t.strip().upper() + ".IS" for t in f.read().splitlines() if t.strip()]

def load_data(tickers):
    sim_start = (datetime.now() - timedelta(days=365)).replace(day=1)
    download_start = (sim_start - timedelta(days=400)).strftime('%Y-%m-%d')
    if os.path.exists(DATA_CACHE_FILE): 
        return pd.read_pickle(DATA_CACHE_FILE)
    print("Veriler indiriliyor...")
    data = yf.download(tickers, start=download_start, auto_adjust=True)
    data.to_pickle(DATA_CACHE_FILE)
    return data

# 1. HAZIRLIK VE OPTİMİZASYON
tickers = get_tickers_from_file(STOX_FILE)
if not tickers: exit()

all_data = load_data(tickers)

# Çift boyutlu index (MultiIndex) hatasını önleme
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

import itertools
ma_pool = [5, 10, 20, 50, 100, 200]
ma_combos = list(itertools.combinations(ma_pool, 4)) # Örn: (5, 20, 50, 200)

print(f"--- Optimizasyon Başladı (Sıklık: {REBALANCE_FREQ}) ---")
print("Veriler matrikse dönüştürülıyor...")

mx_w = raw_data.pct_change(5)
mx_m = raw_data.pct_change(21)
mx_y = raw_data.pct_change(250)
valid_mask = raw_data.notna().cumsum() >= 250

# MA array'lerini önceden hesapla
precalc_ma = {p: raw_data.rolling(p).mean().values for p in ma_pool}

period_indices = [raw_data.index.get_indexer([p], method='ffill')[0] for p in periods]

# Her periyot için Minimum Fiyat Matrisi (Stop-Loss tespiti için)
num_periods = len(period_indices) - 1
period_mins = np.zeros((num_periods, raw_data.shape[1]))
for i in range(num_periods):
    idx_s, idx_e = period_indices[i], period_indices[i+1]
    # Her periyodun dip fiyatları (idx_s ile idx_e arası)
    period_mins[i, :] = np.nanmin(raw_data.values[idx_s:idx_e+1, :], axis=0)

np_close = raw_data.values
np_w = mx_w.values
np_m = mx_m.values
np_y = mx_y.values
np_valid = valid_mask.values

results = []
total_sims = len(w_range) * len(m_range) * len(y_range) * len(ma_combos)
pbar = tqdm(total=total_sims, desc="Optimization")
best_profit_so_far = 0.0

for combo in ma_combos:
    m1, m2, m3, m4 = combo
    arr_m1 = precalc_ma[m1]
    arr_m2 = precalc_ma[m2]
    arr_m3 = precalc_ma[m3]
    arr_m4 = precalc_ma[m4]
    
    # Koşul: Fiyat > m1 > m2 > m3 > m4
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
                
                selection = (
                    is_valid & 
                    curr_ma_cond &
                    (curr_w >= w_val) & 
                    (curr_m >= m_val) & 
                    (curr_y >= y_val)
                )
                
                num_picks = np.sum(selection)
                if num_picks > 0:
                    entry_prices = np_close[idx_start, selection]
                    exit_prices = np_close[idx_end, selection]
                    min_prices = period_mins[i, selection]  # O dönemin dip fiyatları
                    
                    # Stop-Loss hesaplama
                    stop_prices = entry_prices * (1 - STOP_LOSS_RATE)
                    
                    # Stop patladıysa çıkış fiyatı stop fiyatı olur
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
                pbar.set_postfix({'Max': f'{best_profit_so_far:,.0f}'})
                pbar.update(1)

pbar.close()

best = pd.DataFrame(results).loc[pd.DataFrame(results)['Final'].idxmax()]
bm1, bm2, bm3, bm4 = best['MA_Combo']
print(f"\n✅ EN İYİ AYARLAR BULUNDU: MA({bm1}>{bm2}>{bm3}>{bm4}) | W:%{best['W']*100:.0f} M:%{best['M']*100:.0f} Y:%{best['Y']*100:.0f} | Final Bakiye: {best['Final']:,.2f} TL")

# 2. FİNAL SİMÜLASYONU VE VERİ TOPLAMA
trade_history = []
daily_vals = pd.Series(dtype=float)
curr_total = START_CAPITAL

# Finale özel MA koşulu matrisi
ma_cond_final = (raw_data > raw_data.rolling(bm1).mean()) & \
                (raw_data.rolling(bm1).mean() > raw_data.rolling(bm2).mean()) & \
                (raw_data.rolling(bm2).mean() > raw_data.rolling(bm3).mean()) & \
                (raw_data.rolling(bm3).mean() > raw_data.rolling(bm4).mean())

for i in range(len(periods)-1):
    s_p, e_p = periods[i], periods[i+1]
    hist = raw_data.loc[:s_p]
    if hist.empty: continue
    last_date = hist.index[-1]
    
    picks = [t for t in available_tickers if len(hist[t].dropna()) >= 250 and 
             bool(ma_cond_final.loc[last_date, t]) and
             (hist[t].dropna().iloc[-1]/hist[t].dropna().iloc[-5]-1) >= best['W'] and
             (hist[t].dropna().iloc[-1]/hist[t].dropna().iloc[-21]-1) >= best['M'] and
             (hist[t].dropna().iloc[-1]/hist[t].dropna().iloc[-250]-1) >= best['Y']]
    p_data = raw_data.loc[s_p:e_p]
    
    if picks:
        cash_per = curr_total / len(picks)
        leftover = 0
        active_p = []
        for s in picks:
            # MultiIndex'den döndüğü için hata oluşturmaması adına str güvenliği
            ticker_str = s[1] if isinstance(s, tuple) else s
            ticker_str = str(ticker_str).replace('.IS', '')
            
            b_price = p_data[s].iloc[0]
            if pd.isna(b_price) or b_price <= 0:
                leftover += cash_per
                continue
            
            lots = int(cash_per // b_price)
            if lots > 0:
                cost = lots * b_price
                leftover += (cash_per - cost - (cost * COMMISSION_RATE))
                active_p.append({'t': s, 'l': lots, 'b': b_price, 'is_stopped': False})
                trade_history.append([s_p.strftime('%Y-%m-%d'), ticker_str, lots, f"{b_price:.2f}", "ALIS"])
            else: 
                leftover += cash_per
                
        for date, prices in p_data.iterrows():
            current_val = 0
            for item in active_p:
                if not item['is_stopped']:
                    cp = prices[item['t']]
                    if not pd.isna(cp) and cp <= (item['b'] * (1 - STOP_LOSS_RATE)):
                        sell_price = item['b'] * (1 - STOP_LOSS_RATE)
                        leftover += (item['l'] * sell_price) * (1 - COMMISSION_RATE)
                        item['is_stopped'] = True
                        
                        ticker_str = str(item['t'][1] if isinstance(item['t'], tuple) else item['t']).replace('.IS', '')
                        trade_history.append([date.strftime('%Y-%m-%d'), ticker_str, item['l'], f"{sell_price:.2f}", "STOP"])
                    else:
                        current_val += item['l'] * (cp if not pd.isna(cp) else item['b'])
                        
            daily_vals[date] = current_val + leftover
            
        total_sell = sum([item['l'] * p_data[item['t']].iloc[-1] for item in active_p if not item['is_stopped']])
        curr_total = (total_sell * (1 - COMMISSION_RATE)) + leftover
    else:
        for date in p_data.index: daily_vals[date] = curr_total

# 3. GÖRSELLEŞTİRME
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(daily_vals.index, daily_vals.values, color='springgreen', lw=2)
ax1.axhline(y=START_CAPITAL, color='red', ls='--', alpha=0.5)
ax1.set_title(f"Optimum Portföy | H:%{best['W']*100:.0f} M:%{best['M']*100:.0f} Y:%{best['Y']*100:.0f}")
ax1.text(0.5, 1.05, f"FİNAL BAKİYE: {daily_vals.iloc[-1]:,.2f} TL", transform=ax1.transAxes, ha="center", fontsize=18, color='gold', fontweight='bold')
ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} TL'))
ax1.grid(True, alpha=0.1)

ax2.axis('off')
columns = ('Tarih', 'Hisse', 'Lot', 'Fiyat', 'Islem')
table_data = trade_history[-15:] if trade_history else [["-","-","-","-","-"]]

the_table = ax2.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1, 1.6)

for (row, col), cell in the_table.get_celld().items():
    cell.set_text_props(color='black', fontweight='bold')
    if row == 0:
        cell.set_facecolor('#2ecc71')
        cell.set_text_props(color='white')
    else:
        cell.set_facecolor('#ecf0f1')

df_trade = pd.DataFrame(trade_history, columns=columns)
df_trade.to_excel("islem_gecmisi.xlsx", index=False)
print(f"\n✅ İşlem geçmişi 'islem_gecmisi.xlsx' dosyasına kaydedildi.")

# AYLIK GETİRİ RAPORU EKLENTİSİ
monthly_resampled = daily_vals.resample('M').last()
monthly_returns = monthly_resampled.pct_change() * 100

print("\n--- AYLIK GETİRİ RAPORU ---")
for date, ret in monthly_returns.items():
    if pd.notna(ret):
        status = "🚀" if ret > 0 else "🔻"
        print(f"{date.strftime('%Y-%m')}: %{ret:>6.2f} {status}")

plt.tight_layout()
plt.show()