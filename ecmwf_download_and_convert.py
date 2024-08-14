import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import glob
import time
import pygrib
import re
import sys
import importlib
from netCDF4 import Dataset
from ecmwf.opendata import Client

# Function to calculate the latest forecast date and time
def get_latest_forecast_datetime():
    now = datetime.utcnow()
    latest_forecast_time = now - timedelta(hours=8, minutes=45)
    forecast_date = latest_forecast_time.strftime('%Y%m%d')
    forecast_time = '00z' if latest_forecast_time.hour < 12 else '12z'
    return forecast_date, forecast_time

# Function to download files with progress bar
def download_file(url, save_folder):
    local_filename = os.path.join(save_folder, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            with tqdm(
                desc=local_filename,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    return local_filename

# Download function using Script A's methodology
def download_grib_files(forecast_date, forecast_time, base_save_folder):
    base_url = f"https://data.ecmwf.int/forecasts/{forecast_date}/{forecast_time}/ifs/0p25/enfo/"
    date_time_str = f"{forecast_date}_{forecast_time.replace('z', '')}"
    save_folder = os.path.join(base_save_folder, date_time_str)
    os.makedirs(save_folder, exist_ok=True)

    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    grib2_files = [urljoin(base_url, link['href']) for link in soup.find_all('a') if link['href'].endswith('.grib2')]
    total_files = len(grib2_files)
    print(f"Total files to download: {total_files}")

    max_workers = min(total_files, 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_file, grib2_file, save_folder): grib2_file for grib2_file in grib2_files}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                print(f"Downloaded {os.path.basename(result)} successfully.")
            except Exception as e:
                print(f"Failed to download {url}: {e}")

    print("\nAll downloads completed.")
    return save_folder

# Conversion and processing function using Script B's methodology
def process_grib_files(save_folder, config, vars, ensemble_members, hours, date_6h):
    client = Client()

    def process_variable(vv):
        params1 = vars[vv][12]
        params2 = vars[vv][13]

        outfolder = os.path.join(config['dir_converted'], 'ECMWF_IFS_open_ensemble_forecasts', vars[vv][0], date_6h.strftime('%Y%m%d_%H'), str(ensemble_members[-1]).zfill(3), 'Daily')
        files = glob.glob(os.path.join(outfolder, '*.nc'))
        if len(files) > 4:
            print(f"Forecast {date_6h} {vars[vv][0]} already processed, skipping")
            return

        print(f"Downloading raw forecast {date_6h} {vars[vv][0]}")
        t2 = time.time()

        grib_path = os.path.join(save_folder, vars[vv][0] + ".grib2")

        client.retrieve(
            stream="enfo",
            type="pf",
            number=ensemble_members.tolist(),
            date=date_6h.strftime('%Y%m%d'),
            time=date_6h.hour,
            step=hours.tolist(),
            param=params1,
            target=grib_path,
        )

        elapsed_time = time.time() - t2
        print(f"Time elapsed is {elapsed_time} sec")

        t2 = time.time()
        print(f"Opening {date_6h} {vars[vv][0]} {grib_path}")

        grbs = pygrib.open(grib_path)
        datacubes = {}
        for param2 in params2:
            datacubes[param2] = np.zeros((720,1440,len(hours),len(ensemble_members)),dtype=np.single)*np.nan
            for grb in grbs:
                if param2.lower() in grb.name.lower():
                    hour_index = np.where(hours==grb.endStep)[0][0]
                    ensemble_member_index = np.where(ensemble_members==grb.number)[0][0]
                    datacubes[param2][:, :, hour_index, ensemble_member_index] = grb.values[:-1,:]
        grbs.close()

        print(f"Time elapsed is {time.time() - t2} sec")

        # (The rest of the process_variable function remains the same as in Script B)

    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        executor.map(process_variable, range(len(vars)))

    print(f"Forecast {date_6h} downloaded and processed successfully.")

# Main function
if __name__ == "__main__":
    script = sys.argv[0]
    settings_file = sys.argv[1]

    config = importlib.import_module(settings_file, package=None)
    config = config.config

    sys.path.insert(1, config['dir_modules'])
    import library as lb

    lb.mkdir(config['dir_raw'])  # Ensure raw directory is created
    free_space = lb.get_free_space_mb(config['dir_converted'])
    if free_space < 5000:
        raise ValueError('Not enough disk space, terminating to avoid file corruption')

    ensemble_members = np.array([i for i in range(1, 6, 1)])

    vars = np.array([
        ['P','','',1000,0,'sum','precipitation','mm/d','mm/6h','mm/3h','mm/h',1,['tp'],['Total Precipitation']],
        ['Temp','','',1,-273.15,'mean','air_temperature','degree_Celsius','degree_Celsius','degree_Celsius','degree_Celsius',1,['2t'],['2 metre temperature']],
        ['Wind','','',1,0,'mean','wind_speed','m s-1','m s-1','m s-1','m s-1',1,['10u','10v'],['10 metre U wind component','10 metre V wind component']],
        ['Pres','','',1,0,'mean','surface_pressure','Pa','Pa','Pa','Pa',0,['sp'],['Surface pressure']], 
        ['SpecHum','','',1,0,'mean','specific_humidity','kg kg-1','kg kg-1','kg kg-1','kg kg-1',5,['q'],['Specific humidity']],
        ['RelHum','','',1,0,'mean','relative_humidity','%','%','%','%',1,['r'],['Relative humidity']],
        ['SWd','','',1,0,'mean','downward_shortwave_radiation','W m-2','W m-2','W m-2','W m-2',1,['ssrd'],['Surface short-wave (solar) radiation downwards']],
        ['LWd','','',1,0,'mean','downward_longwave_radiation','W m-2','W m-2','W m-2','W m-2',1,['strd'],['Surface long-wave (thermal) radiation downwards']]
    ], dtype=object)

    forecast_date, forecast_time = get_latest_forecast_datetime()
    save_folder = download_grib_files(forecast_date, forecast_time, config['dir_raw'])

    date_6h = pd.to_datetime(pd.Timestamp.now() - timedelta(hours=8, minutes=45)).floor('6H')
    hours = np.array([i for i in range(0, 145, 3)] + [i for i in range(150, 361, 6)]) if date_6h.hour in [0, 12] else np.array([i for i in range(0, 145, 3)])

    process_grib_files(save_folder, config, vars, ensemble_members, hours, date_6h)
