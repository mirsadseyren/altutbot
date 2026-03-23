try:
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    uc = None
    HAS_SELENIUM = False
    print("WARNING: undetected_chromedriver or selenium not found. Scraping will not work.")
import time
import json
import os

def get_indices_and_stocks():
    """
    Scrapes Turkish indices and their constituent stocks from Investing.com.
    Returns a dictionary tree: { "Index Name": { "url": "...", "stocks": ["SYM1", "SYM2"] } }
    """
    
    # Configuration
    indices_url = "https://tr.investing.com/indices/turkey-indices?include-major-indices=true&include-additional-indices=true&include-primary-sectors=true&include-other-indices=true"
    
    if not HAS_SELENIUM:
        print("Selenium not installed, cannot scrape.")
        return

    # Initialize Driver
    print("Initializing Browser...")
    options = uc.ChromeOptions()
    # options.add_argument('--headless') # Headless often gets detected more easily, keep it off for debugging if needed, or on for speed.
    # User requested to just write the program, let's try headless first to be less intrusive, 
    # but if it fails we might need to remove it. For now, let's keep it visible so user can see it working?
    # Actually, headless is better for background tasks unless debugging.
    # Let's use headless but be ready to switch.
    # options.add_argument('--headless') 
    # Disable headless to avoid detection
    # options.add_argument('--headless') 
    
    # Fix for version mismatch 146
    driver = uc.Chrome(options=options, version_main=146)
    
    results = {}
    try:
        print("Fetching indices list...")
        driver.get(indices_url)
        # Wait for table to load by waiting for a known element (Normal button)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, "//span[text()='Normal']"))) 
        
        try:
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "(//span[text()='Normal'])[1]"))).click()
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Geniş']"))).click()
        except:
             pass

    except Exception as e:
        print(f"Error loading page: {e}")

    # Use find_elements to get all rows/items at once instead of looping by index blindly
    try:
        # Wait for the specific list items to be present
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, "//span[contains(@class,'block overflow-hidden')]")))
        
        names_elements = driver.find_elements(By.XPATH, "//span[contains(@class,'block overflow-hidden')]")
        urls_elements = driver.find_elements(By.XPATH, "//a[contains(@class,'overflow-hidden text-ellipsis whitespace-nowrap font-semibold text-primary hover:text-link')]")
        
        min_len = min(len(names_elements), len(urls_elements))
        
        for i in range(min_len):
            try:
                fonadi = names_elements[i].text.strip()
                fonurl = urls_elements[i].get_attribute("href")
                
                if fonadi and fonurl:
                    print(fonadi, fonurl)
                    results[i+1] = [fonadi, fonurl]
            except Exception as e:
                print(f"Error extracting item {i}: {e}")
                
    except Exception as e:
        print(f"Critical Error during scraping loop: {e}")
                
    # Save Results
    driver.quit()
    print(f"Fetched {len(results)} indices.")
    
    # Save as CSV
    with open("endeksler/endeksler.csv", "w", encoding="utf-8") as f:
        f.write("Index,URL\n")
        for index, data in results.items():
            # data is [name, url]
            f.write(f"{data[0]},{data[1]}\n")
            
    # Also save as JSON for the other function to use
    with open("endeksler/endeksler.json", "w", encoding="utf-8") as f:
         json.dump(results, f, indent=4)


def tickers_in_indices():
    settings = {
        'headless': False,
        'version_main': 146
    }
    endeksler = "endeksler/endeksler.json"
    with open(endeksler, "r", encoding="utf-8") as f:
        endeksler = json.load(f)
    all_tickers = {}
    
    # Load existing data if possible to resume or append
    output_file = "endeksler/endeks_hisseleri.json"
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_tickers = json.load(f)
        except:
            pass

    driver = uc.Chrome(options=uc.ChromeOptions(), **settings)
    
    # Use zip to iterate names and urls together
    # Filter out indices we already have if needed, or just overwrite. 
    # For now, we overwrite or update.
    
    items = list(endeksler.values())
    
    for item in items:
        index_name = item[0]
        url = item[1]
        
        print(f"Processing {index_name}...")
        
        try:
            driver.get(url)
            # random sleep to be human-like
            time.sleep(3) 
            
            # Click components tab
            # We use try/except for navigation steps to be robust
            try:
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//li[@data-test='Bileşenler']//a[1]"))).click()
            except Exception as e:
                print(f"Could not click Components for {index_name}, might be empty or different layout. Error: {e}")
                continue
                
            time.sleep(2)
            
            # Try to expand list
            try:
                WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Normal']"))).click()
                WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Geniş']"))).click()
            except:
                # Layout might be already correct or selection not needed
                pass

            # Method 1: Iterating with index (as originally intended but broken)
            # The original code was missing the index in the xpath string.
            # But finding all elements at once is better.
            
            # This xpath selects ALL stock names on the card view
            # Wait for elements to be visible first
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, "//span[contains(@class,'block overflow-hidden')]")))
            except:
                print(f"Time out waiting for stocks for {index_name}")

            stock_elements = driver.find_elements(By.XPATH, "//span[contains(@class,'block overflow-hidden')]")
            
            current_index_stocks = []
            for el in stock_elements:
                text = el.text.strip()
                if text:
                    current_index_stocks.append(text)
            
            # Fallback if the previous xpath didn't work or if it's a table view
            if not current_index_stocks:
                 print(f"No stocks found for {index_name} with primary selector.")

            all_tickers[index_name] = current_index_stocks
            print(current_index_stocks)
            print(f"Found {len(current_index_stocks)} stocks for {index_name}")

            # Save after each index to prevent data loss
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_tickers, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"Error processing {index_name}: {e}")
            
def indices_performance():
    settings = {
        'headless': False,
        'version_main': 146
    }
    endeksler = "endeksler/endeksler.json"
    with open(endeksler, "r", encoding="utf-8") as f:
        endeksler = json.load(f)
    all_tickers = {}
    
    # Load existing data if possible to resume or append
    output_file = "endeksler/endeks_performans.json"
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_tickers = json.load(f)
        except:
            pass

    driver = uc.Chrome(options=uc.ChromeOptions(), **settings)
    
    # Use zip to iterate names and urls together
    # Filter out indices we already have if needed, or just overwrite. 
    # For now, we overwrite or update.
    
    items = list(endeksler.values())
    
    for item in items:
        index_name = item[0]
        url = item[1]
        
        print(f"Processing {index_name}...")
        
        try:
            driver.get(url)
            # random sleep to be human-like
            time.sleep(3) 
            
            # Click components tab
            # We use try/except for navigation steps to be robust
            gunluk = driver.find_element(By.XPATH, "(//div[contains(@class,'text-3xs/3.5 font-semibold')])[1]").text
            haftalik = driver.find_element(By.XPATH, "(//div[contains(@class,'text-3xs/3.5 font-semibold')])[2]").text
            aylik = driver.find_element(By.XPATH, "(//div[contains(@class,'text-3xs/3.5 font-semibold')])[3]").text
            uc_aylik = driver.find_element(By.XPATH, "(//div[contains(@class,'text-3xs/3.5 font-semibold')])[4]").text
            alti_aylik = driver.find_element(By.XPATH, "(//div[contains(@class,'text-3xs/3.5 font-semibold')])[5]").text
            bir_yillik = driver.find_element(By.XPATH, "(//div[contains(@class,'text-3xs/3.5 font-semibold')])[6]").text
            bes_yillik = driver.find_element(By.XPATH, "(//div[contains(@class,'text-3xs/3.5 font-semibold')])[7]").text
            maximum = driver.find_element(By.XPATH, "(//div[contains(@class,'text-3xs/3.5 font-semibold')])[8]").text

            all_tickers[index_name] = {
                "gunluk": gunluk,
                "haftalik": haftalik,
                "aylik": aylik,
                "uc_aylik": uc_aylik,
                "alti_aylik": alti_aylik,
                "bir_yillik": bir_yillik,
                "bes_yillik": bes_yillik,
                "maximum": maximum
            }
            print(all_tickers)
            time.sleep(1)
             

            # Save after each index to prevent data loss
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_tickers, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"Error processing {index_name}: {e}")


if __name__ == "__main__":
    indices_performance()
    tickers_in_indices()