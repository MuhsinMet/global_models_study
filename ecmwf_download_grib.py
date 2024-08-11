import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import argparse

# Function to calculate the latest forecast date and time
def get_latest_forecast_datetime():
    now = datetime.utcnow()
    latest_forecast_time = now - timedelta(hours=8, minutes=45)
    forecast_date = latest_forecast_time.strftime('%Y%m%d')
    forecast_time = '00z' if latest_forecast_time.hour < 12 else '12z'
    return forecast_date, forecast_time

# Function to download files with progress bar
def download_file(url):
    local_filename = os.path.join(save_folder, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # Open the file for writing binary data
        with open(local_filename, 'wb') as f:
            # Progress bar initialization
            with tqdm(
                desc=local_filename,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                # Download in chunks
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    return local_filename

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Download ECMWF forecast files.')
parser.add_argument('--max-files', type=int, default=None, help='Maximum number of files to download (default is all).')
args = parser.parse_args()

# Calculate the latest forecast date and time
forecast_date, forecast_time = get_latest_forecast_datetime()

# URL of the directory containing .grib2 files
base_url = f"https://data.ecmwf.int/forecasts/{forecast_date}/{forecast_time}/ifs/0p25/enfo/"

# Local base directory to save files
base_save_folder = "/mnt/datawaha/hyex/msn/trial/"

# Extract date and time from the URL for folder naming
date_time_str = f"{forecast_date}_{forecast_time.replace('z', '')}"  # Format: YYYYMMDD_HH
save_folder = os.path.join(base_save_folder, date_time_str)

# Create the directory if it does not exist
os.makedirs(save_folder, exist_ok=True)

# Get the page content
response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all links with .grib2 extension
grib2_files = [urljoin(base_url, link['href']) for link in soup.find_all('a') if link['href'].endswith('.grib2')]

# Limit the number of files if specified
if args.max_files is not None:
    grib2_files = grib2_files[:args.max_files]

# Total files to download
total_files = len(grib2_files)
print(f"Total files to download: {total_files}")

# Number of parallel downloads
max_workers = min(total_files, 8)  # Adjust as needed

# Download files in parallel
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_url = {executor.submit(download_file, grib2_file): grib2_file for grib2_file in grib2_files}
    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
            print(f"Downloaded {os.path.basename(result)} successfully.")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

print("\nAll downloads completed.")
