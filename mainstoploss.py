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

# ==========================================================
# GENEL AYARLAR
# ==========================================================
STOX_FILE = 'top_endeks_hisseleri.txt'
DATA_CACHE_FILE = 'bist_data_cache.pkl'
START_CAPITAL = 19000
COMMISSION_RATE = 0.002  # Binde 2 komisyon
REBALANCE_FREQ = 'M'  # 'W': Haftalık, 'MS': Ay Başı
STOP_LOSS_RATE = 0.30  # %15 Zarar durdur
VOLUME_STOP_RATIO = 5.0  # Hacim ortalamasının kaç katına çıkarsa satılsın


def get_tickers_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} dosyası bulunamadı!")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [t.strip().upper() + ".IS" for t in f.read().splitlines() if t.strip()]


def load_data(tickers):
    sim_start = (datetime.now() - timedelta(days=365)).replace(day=1)
    download_start = (sim_start - timedelta(days=400)).strftime('%Y-%m-%d')
    
    if os.path.exists(DATA_CACHE_FILE):
        data = pd.read_pickle(DATA_CACHE_FILE)
        # Check if we have volume data (needed for new stop loss)
        if hasattr(data, 'columns') and 'Volume' in data.columns:
            return data
        # If cache exists but doesn't have Volume (e.g. old cache), we fall through to download
    
    print("Veriler indiriliyor (Close + Volume)...")
    # yfinance download returns a MultiIndex DataFrame if multiple tickers (Price, Ticker)
    data = yf.download(tickers, start=download_start, auto_adjust=True)
    data.to_pickle(DATA_CACHE_FILE)
    return data

# 1. HAZIRLIK
tickers = get_tickers_from_file(STOX_FILE)
if not tickers:
    exit()

all_data = load_data(tickers)

# Handle yfinance MultiIndex or Single Index structure
if isinstance(all_data.columns, pd.MultiIndex):
    # Expected: Levels (Price, Ticker) -> access as data['Close']
    try:
        raw_data = all_data['Close'].dropna(axis=1, how='all')
        raw_volume = all_data['Volume']
    except KeyError:
        # Fallback if structure is different
        raw_data = all_data.xs('Close', axis=1, level=0).dropna(axis=1, how='all')
        raw_volume = all_data.xs('Volume', axis=1, level=0)
else:
    # If flat dataframe (unlikely with multiple tickers but possible)
    # Assuming it might be just Close prices if simple download
    raw_data = all_data # Risk here if volume not present, but we checked in load_data
    raw_volume = pd.DataFrame(0, index=raw_data.index, columns=raw_data.columns) # Dummy

available_tickers = raw_data.columns.tolist()

# Pre-calculate Volume Average (1 Month ~ 21 Days)
vol_avg = raw_volume.rolling(21).mean()
sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1)
periods = pd.date_range(start=sim_start_date, end=datetime.now(), freq=REBALANCE_FREQ)
if periods[-1].date() < datetime.now().date():
    periods = periods.append(pd.DatetimeIndex([pd.Timestamp(datetime.now().date())]))

# 2. OPTİMİZASYON (Vektörize Edilmiş Hızlı Hesaplama)
# 2. OPTİMİZASYON (Vektörize Edilmiş Hızlı Hesaplama)
w_range = np.arange(0.05, 0.16, 0.05)
m_range = np.arange(0.10, 0.31, 0.10)
q_range = np.arange(0.20, 0.51, 0.10) # 3 Aylık: %20 - %50 arası
y_range = np.arange(0.50, 0.81, 0.15)

print(f"--- Optimizasyon Başladı (Sıklık: {REBALANCE_FREQ}) ---")

# A) Matris Ön Hesaplamaları (Dynamic Programming / Vectorization)
print("Veriler matrikse dönüştürülüyor...")
# Fiyat değişimlerini önceden hesapla: (Güncel Fiyat / Eski Fiyat) - 1
# raw_data bir DataFrame (Index: Tarih, Columns: Hisseler)

# 1. Momentum Matrisleri (Tüm tarihler için tek seferde hesapla)
# pct_change(n) = (Price_t / Price_{t-n}) - 1
mx_w = raw_data.pct_change(5)
mx_m = raw_data.pct_change(21)
mx_q = raw_data.pct_change(63) # 3 Aylık (Quarterly) yaklaşık 63 işlem günü
mx_y = raw_data.pct_change(250)

# 2. Geçerlilik Maskesi (En az 250 bar verisi olanlar)
# notna().cumsum() > 250
valid_mask = raw_data.notna().cumsum() >= 250

# 3. Periyotlar için dilimleme hazırlığı
period_indices = [raw_data.index.get_indexer([p], method='ffill')[0] for p in periods]
# period_indices, periods listesindeki tarihlerin raw_data içindeki row indexleridir.

# NumPy arraylerine geçiş (Çok daha hızlı işlem için)
np_close = raw_data.values
np_w = mx_w.values
np_m = mx_m.values
np_q = mx_q.values
np_y = mx_y.values
np_valid = valid_mask.values

results = []
total_sims = len(w_range) * len(m_range) * len(q_range) * len(y_range)
pbar = tqdm(total=total_sims, desc="Optimization")
best_profit_so_far = 0.0

# B) Grid Search Loop
for w_val in w_range:
    for m_val in m_range:
        for q_val in q_range:
            for y_val in y_range:
                temp_balance = START_CAPITAL
                
                # Her periyot için vektörize seçim ve getiri hesabı
                for i in range(len(period_indices) - 1):
                    idx_start = period_indices[i]
                    idx_end = period_indices[i+1]
                    
                    # O tarihteki momentum değerleri
                    # row = idx_start
                    curr_w = np_w[idx_start, :]
                    curr_m = np_m[idx_start, :]
                    curr_q = np_q[idx_start, :]
                    curr_y = np_y[idx_start, :]
                    is_valid = np_valid[idx_start, :]
                    
                    # Seçim Kriterleri (Boolean Mask)
                    # w, m, y kriterleri ve geçerlilik
                    # NaN check is implicit usually false in comparisons, but safe to be explicit if needed
                    # np_w vs nan -> False
                    selection = (
                        is_valid & 
                        (curr_w >= w_val) & 
                        (curr_m >= m_val) & 
                        (curr_q >= q_val) &
                        (curr_y >= y_val)
                    )
                    
                    # Seçilen hisse sayısı
                    # selection bir boolean array
                    num_picks = np.sum(selection)
                    
                    if num_picks > 0:
                        # Getiriler: (Dönem Sonu Fiyat / Dönem Başı Fiyat)
                        # Alış Fiyatı: np_close[idx_start]
                        # Satış Fiyatı: np_close[idx_end]
                        # Getiri Carpanı: price_end / price_start
                        
                        entry_prices = np_close[idx_start, selection]
                        exit_prices = np_close[idx_end, selection]
                        
                        # Basit getiri hesabı (Komisyonlu)
                        # Nakit Bölu Hisse Sayısı = Hisse Başına Ayrılan TL
                        cash_per_stock = temp_balance / num_picks
                        
                        # Lot Sayısı (Tam Sayı)
                        # lots = floor(cash_per_stock / entry_price) -> Ama vektörize yaptığımız için float kalabilir simülasyon hızı için.
                        # Tam simülasyon için:
                        lots = np.floor(cash_per_stock / entry_prices)
                        
                        # Maliyet
                        cost = lots * entry_prices
                        
                        # Satış Geliri (Bitiş Fiyatı * Lot * (1-Kom))
                        revenue = (lots * exit_prices * (1 - COMMISSION_RATE))
                        
                        # Artan Nakit (Cash Per Stock - Cost - Cost*Kom)
                        unused_cash = (cash_per_stock - cost * (1 + COMMISSION_RATE))
                        # unused_cash negatif olamaz teorik olarak floor aldık ama float hatası vs olursa diye:
                        # Bu basit hesapta unused_cash toplamı:
                        
                        total_revenue = np.sum(revenue)
                        total_unused = np.sum(unused_cash)
                        
                        temp_balance = total_revenue + total_unused

                if temp_balance > best_profit_so_far:
                    best_profit_so_far = temp_balance
                    
                results.append({'W': w_val, 'M': m_val, 'Q': q_val, 'Y': y_val, 'Final': temp_balance})
                pbar.set_postfix({'Max Profit': f'{best_profit_so_far:,.0f}'})
                pbar.update(1)

pbar.close()

best = pd.DataFrame(results).loc[pd.DataFrame(results)['Final'].idxmax()]

# 3. STOP LOSS DAHİL DETAYLI FİNAL SİMÜLASYONU
trade_history = []
daily_vals = pd.Series(dtype=float)
curr_total = START_CAPITAL

for i in range(len(periods) - 1):
    s_p, e_p = periods[i], periods[i + 1]
    hist = raw_data.loc[:s_p]

    picks = [t for t in available_tickers if len(hist[t].dropna()) >= 250 and
             (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-5] - 1) >= best['W'] and
             (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-21] - 1) >= best['M'] and
             (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-63] - 1) >= best['Q'] and
             (hist[t].dropna().iloc[-1] / hist[t].dropna().iloc[-250] - 1) >= best['Y']]

    p_data = raw_data.loc[s_p:e_p]
    if picks:
        cash_per = curr_total / len(picks)
        leftover_cash = 0
        active_p = []

        for s in picks:
            b_price = p_data[s].iloc[0]
            
            # Check for valid price
            if pd.isna(b_price) or b_price <= 0:
                lots = 0
            else:
                lots = int(cash_per // b_price)

            if lots > 0:
                cost = lots * b_price
                leftover_cash += (cash_per - cost - (cost * COMMISSION_RATE))
                active_p.append({'t': s, 'l': lots, 'b': b_price, 'max_p': b_price, 'is_stopped': False})
                stock_val_init = sum([x['l'] * p_data[x['t']].iloc[0] for x in active_p if not x.get('is_stopped', False)])
                total_val_init = leftover_cash + stock_val_init
                trade_history.append([s_p.strftime('%Y-%m-%d'), s.replace('.IS', ''), lots, f"{b_price:.2f}", "ALIS", 
                                      f"{leftover_cash:,.2f}", f"{total_val_init:,.2f}"])
            else:
                leftover_cash += cash_per

        for date, prices in p_data.iterrows():
            current_stocks_value = 0
            for item in active_p:
                if not item['is_stopped']:
                    cp = prices[item['t']]
                    cv = raw_volume.loc[date, item['t']]
                    av = vol_avg.loc[date, item['t']]
                    
                    stop_reason = None
                    
                    # 1. Stop Loss Kontrolü (Fiyat)
                    # Max Price Update
                    if not pd.isna(cp) and cp > item.get('max_p', item['b']):
                         item['max_p'] = cp

                    # Trailing Stop: Max fiyattan %X düşerse sat
                    if not pd.isna(cp) and cp <= (item.get('max_p', item['b']) * (1 - STOP_LOSS_RATE)):
                        stop_reason = "STOP LOSS"
                    
                    # 2. Hacim Stop Kontrolü (Hacim Patlaması - Satış Sinyali Olabilir)
                    elif not pd.isna(cv) and not pd.isna(av) and av > 0 and cv >= (av * VOLUME_STOP_RATIO):
                        stop_reason = "HACIM_STOP"

                    if stop_reason:
                        sell_val = (item['l'] * cp) * (1 - COMMISSION_RATE)
                        leftover_cash += sell_val
                        item['is_stopped'] = True
                        
                        # Toplam Varlık Loglama
                        stock_val_now = sum([x['l'] * (prices[x['t']] if not pd.isna(prices[x['t']]) and prices[x['t']] > 0 else x['b']) for x in active_p if not x['is_stopped']])
                        total_val_now = leftover_cash + stock_val_now
                        
                        trade_history.append(
                            [date.strftime('%Y-%m-%d'), item['t'].replace('.IS', ''), item['l'], f"{cp:.2f}",
                             stop_reason, f"{leftover_cash:,.2f}", f"{total_val_now:,.2f}"])
                    else:
                        # Eğer fiyat yoksa veya 0 ise (veri hatası), alış fiyatını baz al (Grafigi çökertmemek için)
                        price_to_use = cp if not pd.isna(cp) and cp > 0 else item['b']
                        current_stocks_value += item['l'] * price_to_use

            # 2. Besleme (Boştaki Nakdi En Güçlüye Yatır - Eğer aktif hisse varsa)
            # Stop olanlardan gelen nakit: leftover_cash
            active_items = [it for it in active_p if not it['is_stopped']]
            # Only consider items with valid current price
            feedable_items = [it for it in active_items if not pd.isna(prices[it['t']]) and prices[it['t']] > 0]

            if leftover_cash > 2000 and feedable_items: 
                # En iyi performansı gösteren (Fiyat / Maliyet)
                winner = max(feedable_items, key=lambda x: prices[x['t']] / x['b'])
                p_win = prices[winner['t']]
                
                extra_lots = int(leftover_cash // (p_win * (1 + COMMISSION_RATE)))
                if extra_lots > 0:
                    cost_extra = extra_lots * p_win
                    leftover_cash -= (cost_extra * (1 + COMMISSION_RATE))
                    winner['l'] += extra_lots
                    
                    # Loglama öncesi günlük değeri güncelle (Ekranda drop olmaması için)
                    current_stocks_value += (extra_lots * p_win)
                    
                    # Toplam Varlık (Nakit + Hisse) Loglama
                    stock_val_now = sum([it['l'] * (prices[it['t']] if not pd.isna(prices[it['t']]) and prices[it['t']] > 0 else it['b']) for it in active_p if not it['is_stopped']])
                    total_val_now = leftover_cash + stock_val_now
                    
                    trade_history.append([date.strftime('%Y-%m-%d'), winner['t'].replace('.IS',''), extra_lots, 
                                          f"{p_win:.2f}", "BESLEME", f"{leftover_cash:,.2f}", f"{total_val_now:,.2f}"])

            daily_vals[date] = current_stocks_value + leftover_cash

        # Dönem sonu aktif olanları sat ve yeni döneme nakitle gir
        final_sell_val = sum([(item['l'] * p_data[item['t']].iloc[-1] * (1 - COMMISSION_RATE)) for item in active_p if
                              not item['is_stopped']])
        curr_total = final_sell_val + leftover_cash
    else:
        for date in p_data.index: daily_vals[date] = curr_total

# 4. GÖRSELLEŞTİRME
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

final_money = daily_vals.values[-1]
final_money_str = f"{final_money:,.2f} TL"



# Üst Grafik
ax1.plot(daily_vals.index, daily_vals.values, color='springgreen', lw=2.5, label="Portföy Değeri")
ax1.axhline(y=START_CAPITAL, color='white', ls='--', alpha=0.3)
ax1.set_title(f"Momentum | SL:%{STOP_LOSS_RATE*100} | W:%{best['W']*100:.0f} M:%{best['M']*100:.0f} Q:%{best['Q']*100:.0f} Y:%{best['Y']*100:.0f}", fontsize=12)
ax1.text(0.5, 1.02, f"FİNAL BAKİYE: {final_money_str}", transform=ax1.transAxes, ha="center", fontsize=16, color='gold',
         fontweight='bold')

# Y ekseni formatı (Binlik ayırıcı)
ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} TL'))
ax1.grid(True, alpha=0.1)

# Alt Tablo (Son 15 işlem - Besleme hariç)
ax2.axis('off')
columns = ('Tarih', 'Hisse', 'Lot', 'Fiyat', 'İşlem', 'Nakit', 'Toplam Varlık')
# Besleme olmayan son 15 işlemi göster
filtered_history = [x for x in trade_history if x[4] != "BESLEME"]
table_data = filtered_history[-15:] if filtered_history else [["-"] * 7]
the_table = ax2.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')

for (row, col), cell in the_table.get_celld().items():
    cell.set_text_props(color='black', fontweight='bold')
    if row == 0:
        cell.set_facecolor('#3498db')
        cell.set_text_props(color='white')
    else:
        cell.set_facecolor('#ecf0f1')

# Dosya Kaydı
pd.DataFrame(trade_history, columns=columns).to_excel("islem_gecmisi_final.xlsx", index=False)

plt.tight_layout()
print(f"Momentum | SL:%{STOP_LOSS_RATE*100} | W:%{best['W']*100:.0f} M:%{best['M']*100:.0f} Q:%{best['Q']*100:.0f} Y:%{best['Y']*100:.0f}")
print(f"\n🎯 İşlem Tamamlandı. Son Bakiye: {final_money_str}")


# ==========================================================
# 5. AYLIK KAZANÇ ANALİZİ (YENİ BÖLÜM)
# ==========================================================
monthly_resampled = daily_vals.resample('M').last()
monthly_returns = monthly_resampled.pct_change() * 100

print("\n--- AYLIK GETİRİ RAPORU ---")
for date, ret in monthly_returns.items():
    if pd.notna(ret):
        status = "🚀" if ret > 0 else "🔻"
        print(f"{date.strftime('%Y-%m')}: %{ret:>6.2f} {status}")

# Aylık Getiri Görselleştirme
plt.figure(figsize=(12, 5))
colors = ['green' if x > 0 else 'red' for x in monthly_returns.dropna()]
monthly_returns.dropna().plot(kind='bar', color=colors)
plt.title("Aylık Yüzdelik (%) Getiri Değişimi", fontsize=14)
plt.ylabel("Değişim (%)")
plt.xlabel("Ay")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()