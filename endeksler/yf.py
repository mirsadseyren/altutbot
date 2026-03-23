import yfinance as yf


# 2. Endeks için (BIST 100)
# Türkiye için sembolün sonuna .IS eklemeyi unutmayın
bist100 = yf.Ticker("XHARZ.IS")

# 3. Tarihsel Veri Çekme (Örn: Son 1 yıl)
data = bist100.history(period="1y", interval="1d")
print(data.tail())