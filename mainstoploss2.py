import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timedelta
import warnings
import os
import numpy as np
from tqdm import tqdm  # İlerleme çubuğu için

warnings.filterwarnings('ignore')

# ==========================================================
# GENEL AYARLAR
# ==========================================================
STOX_FILE = 'top_endeks_hisseleri.txt'
DATA_CACHE_FILE = 'bist_data_cache.pkl'
START_CAPITAL = 100000
COMMISSION_RATE = 0.002

# OPTİMİZASYON ARALIKLARI (Genişletildi)
w_range = np.arange(0.05, 0.16, 0.05)
m_range = np.arange(0.10, 0.31, 0.10)
y_range = np.arange(0.50, 0.81, 0.15)
stop_loss_range = np.arange(0.05, 0.21, 0.05)  # %5, %10, %15, %20
freq_range = ['W', 'MS', '2MS']  # Haftalık, Aylık, 2 Aylık


# ... (get_tickers_from_file ve load_data fonksiyonları aynı) ...
def get_tickers_from_file(file_path):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [t.strip().upper() + ".IS" for t in f.read().splitlines() if t.strip()]


def load_data(tickers):
    sim_start = (datetime.now() - timedelta(days=365)).replace(day=1)
    if os.path.exists(DATA_CACHE_FILE): return pd.read_pickle(DATA_CACHE_FILE)
    print("Veriler indiriliyor...")
    data = yf.download(tickers, start=(sim_start - timedelta(days=400)).strftime('%Y-%m-%d'), auto_adjust=True)['Close']
    data = data.dropna(axis=1, how='all')
    data.to_pickle(DATA_CACHE_FILE)
    return data


tickers = get_tickers_from_file(STOX_FILE)
raw_data = load_data(tickers)
available_tickers = raw_data.columns.tolist()
sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1)

# 1. MEGA OPTİMİZASYON DÖNGÜSÜ
results = []
total_combinations = len(w_range) * len(m_range) * len(y_range) * len(stop_loss_range) * len(freq_range)

print(f"--- Toplam {total_combinations} kombinasyon test ediliyor... ---")

with tqdm(total=total_combinations, desc="Optimizasyon", unit="it") as pbar:
    for freq in freq_range:
        periods = pd.date_range(start=sim_start_date, end=datetime.now(), freq=freq)
        if periods[-1].date() < datetime.now().date():
            periods = periods.append(pd.DatetimeIndex([pd.Timestamp(datetime.now().date())]))
        for sl in stop_loss_range:
            for w in w_range:
                for m in m_range:
                    for y in y_range:
                        temp_bal = START_CAPITAL
                        for i in range(len(periods) - 1):
                            hist = raw_data.loc[:periods[i]]
                            picks = [t for t in available_tickers if len(hist[t].dropna()) >= 250 and
                                     (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-5] - 1) >= w and
                                     (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-21] - 1) >= m and
                                     (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-250] - 1) >= y]

                            if picks:
                                p_data = raw_data[picks].loc[periods[i]:periods[i + 1]]
                                cash_per = temp_bal / len(picks)
                                current_period_val = 0
                                for s in picks:
                                    s_series = p_data[s].dropna()
                                    if len(s_series) < 2:
                                        current_period_val += cash_per  # Periyotta yeterli veri yok
                                        continue
                                    b_p = s_series.iloc[0]
                                    s_p = s_series.iloc[-1]
                                    # BUG FIX 1: NaN veya sıfır alış fiyatı kontrolü
                                    if pd.isna(b_p) or b_p <= 0 or pd.isna(s_p):
                                        current_period_val += cash_per  # Pozisyon açılamadı, nakit kalsın
                                        continue
                                    # Stop loss kontrolü — exit_p gerçek stop fiyatı
                                    stop_price = b_p * (1 - sl)
                                    min_p = p_data[s].min()
                                    if (not pd.isna(min_p)) and min_p <= stop_price:
                                        exit_p = stop_price  # Stop-limit fiyatından çık
                                    else:
                                        exit_p = s_p

                                    lots = int(cash_per // b_p)
                                    if lots <= 0:
                                        current_period_val += cash_per  # Lot alınamadı, nakit kalsın
                                        continue
                                    # Alış komisyonu: lots*b_p*COMMISSION_RATE, Satış komisyonu: lots*exit_p*COMMISSION_RATE
                                    buy_cost = lots * b_p * (1 + COMMISSION_RATE)
                                    sell_proceeds = lots * exit_p * (1 - COMMISSION_RATE)
                                    leftover_cash = cash_per - buy_cost
                                    val = sell_proceeds + leftover_cash
                                    current_period_val += val
                                # BUG FIX 2: picks boşsa temp_bal değişmemeli, doluysa güncelle
                                temp_bal = current_period_val
                            # picks boşsa temp_bal olduğu gibi kalır (nakit saklanır)

                        results.append({'FREQ': freq, 'SL': sl, 'W': w, 'M': m, 'Y': y, 'Final': temp_bal})
                        pbar.update(1)

# 2. EN İYİSİNİ SEÇ VE FİNAL SİMÜLASYONU
df_res = pd.DataFrame(results)
best = df_res.loc[df_res['Final'].idxmax()]
print(f"\n✅ EN İYİ AYARLAR BULUNDU: {best['FREQ']} frekans, %{best['SL'] * 100} Stop-Loss")

# Final simülasyonu (Grafik için en iyi ayarlarla tekrar çalıştır)
trade_history = []
daily_vals = pd.Series(dtype=float)
curr_total = START_CAPITAL
final_periods = pd.date_range(start=sim_start_date, end=datetime.now(), freq=best['FREQ'])
if final_periods[-1].date() < datetime.now().date():
    final_periods = final_periods.append(pd.DatetimeIndex([pd.Timestamp(datetime.now().date())]))

for i in range(len(final_periods) - 1):
    s_p, e_p = final_periods[i], final_periods[i + 1]
    hist = raw_data.loc[:s_p]
    picks = [t for t in available_tickers if len(hist[t].dropna()) >= 250 and
             (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-5] - 1) >= best['W'] and
             (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-21] - 1) >= best['M'] and
             (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-250] - 1) >= best['Y']]

    p_data = raw_data.loc[s_p:e_p]
    if picks:
        cash_per = curr_total / len(picks)
        leftover = 0
        active_p = []
        for s in picks:
            s_series = p_data[s].dropna()
            if len(s_series) < 1:
                leftover += cash_per  # Periyotta veri yok, nakit kalsın
                continue
            b_price = s_series.iloc[0]
            if pd.isna(b_price) or b_price <= 0:
                leftover += cash_per
                continue
            lots = int(cash_per // b_price)
            if lots > 0:
                leftover += (cash_per - (lots * b_price) - (lots * b_price * COMMISSION_RATE))
                active_p.append({'t': s, 'l': lots, 'b': b_price, 'is_stopped': False})
                trade_history.append([s_p.strftime('%Y-%m-%d'), s.replace('.IS', ''), lots, f"{b_price:.2f}", "ALIS"])
            else:
                leftover += cash_per

        for date, prices in p_data.iterrows():
            current_val = 0
            for item in active_p:
                if not item['is_stopped']:
                    cp = prices[item['t']]
                    if not pd.isna(cp) and cp <= (item['b'] * (1 - best['SL'])):
                        leftover += (item['l'] * cp) * (1 - COMMISSION_RATE)
                        item['is_stopped'] = True
                        trade_history.append(
                            [date.strftime('%Y-%m-%d'), item['t'].replace('.IS', ''), item['l'], f"{cp:.2f}", "STOP"])
                    else:
                        current_val += item['l'] * (cp if not pd.isna(cp) else item['b'])
            daily_vals[date] = current_val + leftover
        curr_total = daily_vals.iloc[-1]
    else:
        for date in p_data.index: daily_vals[date] = curr_total

# 3. GÖRSELLEŞTİRME
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
final_val = daily_vals.values[-1]

ax1.plot(daily_vals.index, daily_vals.values, color='springgreen', lw=2)
ax1.set_title(
    f"Mega Optimum Portföy\nFreq: {best['FREQ']} | SL: %{best['SL'] * 100:.0f} | W:%{best['W'] * 100:.0f} M:%{best['M'] * 100:.0f} Y:%{best['Y'] * 100:.0f}",
    fontsize=12)
ax1.text(0.5, 1.05, f"FİNAL BAKİYE: {final_val:,.2f} TL", transform=ax1.transAxes, ha="center", fontsize=18,
         color='gold', fontweight='bold')
ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} TL'))

ax2.axis('off')
table_data = trade_history[-15:] if trade_history else [["-"] * 5]
the_table = ax2.table(cellText=table_data, colLabels=('Tarih', 'Hisse', 'Lot', 'Fiyat', 'Tip'), loc='center',
                      cellLoc='center')
for (r, c), cell in the_table.get_celld().items():
    cell.set_text_props(color='black', fontweight='bold')
    cell.set_facecolor('#3498db' if r == 0 else '#ecf0f1')

plt.tight_layout()
plt.show()