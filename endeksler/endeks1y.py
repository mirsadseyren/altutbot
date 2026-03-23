import json
import pandas as pd
import os
import re
import sys
# from endekshisseleri import indices_performance

def parse_percentage(value):
    if not isinstance(value, str):
        return None
    if value == '-':
        return None
        
    # Clean up the string
    # Expected formats: "+%6,51", "-%0,40", "+15,38Mln%", "+1,30B%"
    
    # Remove '%', '+'
    val = value.replace('%', '').replace('+', '')
    
    multiplier = 1.0
    if 'Mln' in val:
        multiplier = 1_000_000.0
        val = val.replace('Mln', '')
    elif 'B' in val:
        multiplier = 1_000_000_000.0
        val = val.replace('B', '')
    elif 'K' in val:
        multiplier = 1_000.0
        val = val.replace('K', '')
        
    # Replace comma with dot
    val = val.replace(',', '.')
    
    try:
        return float(val) * multiplier
    except ValueError:
        return None

def leaderboard(leaders=5):
    json_path = os.path.join(os.path.dirname(__file__), 'endeks_performans.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} could not be found.")
        return

    rows = []
    for index_name, metrics in data.items():
        row = {'Index': index_name}
        # Parse fields
        for key in ['gunluk', 'haftalik', 'aylik', 'uc_aylik', 'alti_aylik', 'bir_yillik']:
             str_val = metrics.get(key, '-')
             row[key] = parse_percentage(str_val)
             row[f'{key}_str'] = str_val # Keep original for display if needed
        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Columns to prioritize for "success": Monthly, Weekly, 3-Month
    # Let's sort by Monthly descending order for "momentum"
    
    df_sorted = df.sort_values(by='bir_yillik', ascending=False)
    
    print("\n--- En Başarılı Endeksler (Aylık Yüzde Artışa Göre) ---")
    
    # Select columns for display
    display_cols = ['Index', 'gunluk', 'haftalik', 'aylik', 'uc_aylik', 'bir_yillik']
    
    # Format floats to 2 decimal places for better reading
    df_display = df_sorted[display_cols].copy()
    
    # Reset index for ranking
    df_display.reset_index(drop=True, inplace=True)
    df_display.index += 1  # 1-based ranking
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    print(df_display)
    
    # Also show top for Weekly
    print("\n\n--- En Başarılı Endeksler (Haftalık Yüzde Artışa Göre) ---")
    df_sorted_weekly = df.sort_values(by='haftalik', ascending=False).head(10)
    df_display_weekly = df_sorted_weekly[display_cols].copy().reset_index(drop=True)
    df_display_weekly.index += 1
    print(df_display_weekly)
    
    # --- TASK: Extract tickers for top 5 monthly indices ---
    top_5_indices = df_sorted.head(leaders)['Index'].tolist()
    print(f"\nTop 5 Indices (Monthly): {top_5_indices}")
    
    try:
        hisseler_json_path = os.path.join(os.path.dirname(__file__), 'endeks_hisseleri.json')
        with open(hisseler_json_path, 'r', encoding='utf-8') as f:
            hisseler_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {hisseler_json_path} could not be found.")
        return

    unique_tickers = set()
    
    for idx_name in top_5_indices:
        # Check if index exists in the hisseler file
        if idx_name in hisseler_data:
            tickers = hisseler_data[idx_name]
            # Some entries might be list of strings, some empty
            if isinstance(tickers, list):
                for t in tickers:
                    if isinstance(t, str) and t.strip():
                        unique_tickers.add(t.strip())
            else:
                 print(f"Warning: Data for {idx_name} is not a list.")
        else:
            print(f"Warning: {idx_name} not found in endeks_hisseleri.json")
            
    # Save to txt
    output_txt_path = os.path.join(os.path.dirname(__file__), '../top_endeks_hisseleri.txt')
    
    sorted_unique_tickers = sorted(list(unique_tickers))
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for ticker in sorted_unique_tickers:
            f.write(f"{ticker}\n")
            
    print(f"\n{len(sorted_unique_tickers)} unique tickers saved to: {output_txt_path}")

    # Save selected indices names to a file
    selected_indices_path = os.path.join(os.path.dirname(__file__), '../selected_indices.json')
    with open(selected_indices_path, 'w', encoding='utf-8') as f:
        json.dump(top_5_indices, f, ensure_ascii=False, indent=4)
    print(f"Selected indices saved to: {selected_indices_path}")

def get_top_indices_tickers(n_leaders=10):
    """
    Returns a list of unique tickers for the top n_leaders indices.
    """
    json_path = os.path.join(os.path.dirname(__file__), 'endeks_performans.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return []

    rows = []
    for index_name, metrics in data.items():
        row = {'Index': index_name}
        for key in ['gunluk', 'haftalik', 'aylik', 'uc_aylik', 'alti_aylik', 'bir_yillik']:
             str_val = metrics.get(key, '-')
             row[key] = parse_percentage(str_val)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by monthly return
    df_sorted = df.sort_values(by='bir_yillik', ascending=False)
    
    top_indices = df_sorted.head(n_leaders)['Index'].tolist()
    
    try:
        hisseler_json_path = os.path.join(os.path.dirname(__file__), 'endeks_hisseleri.json')
        with open(hisseler_json_path, 'r', encoding='utf-8') as f:
            hisseler_data = json.load(f)
    except FileNotFoundError:
        return []

    unique_tickers = set()
    for idx_name in top_indices:
        if idx_name in hisseler_data:
            tickers = hisseler_data[idx_name]
            if isinstance(tickers, list):
                for t in tickers:
                    if isinstance(t, str) and t.strip():
                        unique_tickers.add(t.strip() + ".IS") # Add .IS suffix here as expected by regression
            
    return sorted(list(unique_tickers))

if __name__ == "__main__":
    # indices_performance()
    
    # Check for command line argument
    num_leaders = 10
    if len(sys.argv) > 1:
        try:
            num_leaders = int(sys.argv[1])
        except ValueError:
            pass
            
    leaderboard(num_leaders)
