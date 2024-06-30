# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 08:02:07 2024

@author: popul
"""

import os
import numpy as np
import pandas as pd
import time
import xarray as xr
from datetime import datetime, timedelta
import cdsapi
from concurrent.futures import ThreadPoolExecutor
import shutil

# Configuration (replace with paths and settings)
config = {
    'dir_temp': 'D:/KAUST/Global/temp', #Location need to change
    'dir_raw': 'D:/KAUST/Global/raw', #Location need to change
    'dir_converted': 'D:/KAUST/Global/converted', #Location need to change
}

# Ensure directories exist
os.makedirs(config['dir_raw'], exist_ok=True)
os.makedirs(config['dir_converted'], exist_ok=True)

# Check free space
def get_free_space_mb(folder):
    total, used, free = shutil.disk_usage(folder)
    return free / 1024 / 1024

free_space = get_free_space_mb(config['dir_converted'])
if free_space < 5000:
    raise ValueError('Not enough disk space, terminating to avoid file corruption')

# Variables to download
variables = {
    'precipitation': 'total_precipitation',
    'temperature': '2m_temperature'
}

# Define the time range
start_date = datetime(2023, 1, 1)  # Define start date
end_date = datetime(2023, 1, 31)  # Define end date
dates = pd.date_range(start_date, end_date, freq='D')

# Initialize CDS API client
c = cdsapi.Client()

# Function to download data using CDS API
def download_cds_data(date, variable):
    target_dir = os.path.join(config['dir_temp'], 'ECMWF', date.strftime('%Y%m%d'), variable)
    os.makedirs(target_dir, exist_ok=True)
    
    filename = f'ecmwf_{variable}_{date.strftime("%Y%m%d")}.nc'
    filepath = os.path.join(target_dir, filename)
    
    if not os.path.isfile(filepath):
        print(f'Downloading {variable} data for {date.strftime("%Y-%m-%d")}')
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variable,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'time': '00:00',
                'format': 'netcdf',
            },
            filepath
        )

# Download data for each date and variable
with ThreadPoolExecutor(max_workers=3) as executor:
    for date in dates:
        for variable in variables.keys():
            executor.submit(download_cds_data, date, variable)

# Process downloaded data
def process_data(date, variable):
    raw_dir = os.path.join(config['dir_temp'], 'ECMWF', date.strftime('%Y%m%d'), variable)
    raw_file = f'ecmwf_{variable}_{date.strftime("%Y%m%d")}.nc'
    
    if not os.path.isfile(os.path.join(raw_dir, raw_file)):
        print(f'Skipping {variable} data for {date.strftime("%Y-%m-%d")} - file not found')
        return

    # Load data using xarray
    ds = xr.open_dataset(os.path.join(raw_dir, raw_file))
    
    # Perform any necessary processing, e.g., unit conversion
    if variable == 'temperature':
        ds[variable] = ds[variable] - 273.15  # Convert from K to °C

    # Save processed data
    processed_dir = os.path.join(config['dir_converted'], 'ECMWF', variable, date.strftime('%Y%m%d'))
    os.makedirs(processed_dir, exist_ok=True)
    processed_file = os.path.join(processed_dir, raw_file)
    
    ds.to_netcdf(processed_file)

    # Clean up raw data
    os.remove(os.path.join(raw_dir, raw_file))

# Process data for each date and variable
for date in dates:
    for variable in variables.keys():
        process_data(date, variable)

print('Data download and processing complete.')
