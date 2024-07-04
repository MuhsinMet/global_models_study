# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 07:27:06 2024

@author: popul
"""

from ecmwfapi import ECMWFDataServer
import xarray as xr

# Replace these with your actual API key and email
api_key = '0e63dade9cebd2822176cb17accf1beb'
api_email = 'hylke.beck@kaust.edu.sa'

# Initialize the ECMWF server with API key and email
server = ECMWFDataServer(url='https://api.ecmwf.int/v1', key=api_key, email=api_email)

# Define the parameters for the data request
request_params = {
    'class': 'od',  # Operational data class for medium-range forecasts
    'dataset': 'tigge',  # Specify TIGGE dataset
    'expver': 'prod',  # Experiment version
    'date': '2017-08-24',  # Forecast initialization date
    'step': '0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240',  # Up to 10 days, every 6 hours
    'levtype': 'sfc',  # Surface level
    'number': '1/2',  # Limit to the first two ensemble members
    'param': '167.128',  # 2m temperature
    'stream': 'enfo',  # Ensemble forecast
    'time': '00:00:00',  # Forecast time
    'type': 'cf',  # Control forecast
    'target': '/home/puthiyma/msn/trial.nc',  # Output file
}

# Retrieve the data
try:
    server.retrieve(request_params)
    print("Data retrieval successful.")
except Exception as e:
    print(f"Error during data retrieval: {e}")

# Load the downloaded NetCDF file
try:
    ds = xr.open_dataset('/home/puthiyma/msn/trial.nc', engine='netcdf4')
    print(ds)
except Exception as e:
    print(f"Error opening the NetCDF file: {e}")

# For minimal trial run, slice the data to include only a small subset
# Limiting to the first few steps and ensemble members
if 'ds' in locals():
    sliced_ds = ds.isel(step=slice(0, 5), number=slice(0, 2))

    # Save the sliced data to a new NetCDF file
    sliced_ds.to_netcdf('/home/puthiyma/msn/sliced_trial.nc')

    print("Data downloaded and sliced successfully. The sliced data is saved in '/home/puthiyma/msn/sliced_trial.nc'")
else:
    print("Failed to retrieve and open the NetCDF file.")