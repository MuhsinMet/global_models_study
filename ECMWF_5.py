# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:27:16 2024

@author: popul
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import importlib
from datetime import timedelta
from subprocess import call
from concurrent.futures import ThreadPoolExecutor
import pygrib
import glob
import shutil
import re

if len(sys.argv) < 2:
    print("Usage: python ecmwf_4.py <settings_file>")
    sys.exit(1)

script = sys.argv[0]
settings_file = sys.argv[1]

config = importlib.import_module(settings_file, package=None)
config = config.config

sys.path.insert(1, config['dir_modules'])
import library as lb

lb.mkdir(os.path.join(config['dir_raw'], 'ECMWF_IFS_open_forecast'))
free_space = lb.get_free_space_mb(config['dir_converted'])
if free_space < 5000:
    raise ValueError('Not enough disk space, terminating to avoid file corruption')

vars = np.array([
    ['P', '', '', 1, 0, 'sum', 'precipitation', 'mm/d', 'mm/3h', 'mm/h', 1, ['tot_prec']],
    ['Temp', '', '', 1, -273.15, 'mean', 'air_temperature', 'degree_Celsius', 'degree_Celsius', 'degree_Celsius', 1, ['t_2m']],
], dtype=object)

dates_6h = [pd.to_datetime(pd.Timestamp.now() - timedelta(hours=8)).floor('6H')]

for date_6h in dates_6h:
    t1 = time.time()

    print('=================================================================================')
    if os.path.isfile(os.path.join(config['dir_temp'], "ECMWF_filelist.txt")):
        os.remove(os.path.join(config['dir_temp'], "ECMWF_filelist.txt"))
    url = f'https://data.ecmwf.int/forecasts/{date_6h.strftime("%Y%m%d")}/{date_6h.strftime("%H")}z/ifs/0p25/oper/'
    command = f"wget -q -nH -nd '{url}' -O - | grep -oP '(?<=oper/)[^\" ]*\\.grib2(?=\")' > '{os.path.join(config['dir_temp'], 'ECMWF_filelist.txt')}'"
    print(command)
    call(command, shell=True)

    with open(os.path.join(config['dir_temp'], 'ECMWF_filelist.txt'), 'r') as f:
        file_contents = f.read()
        print("Contents of ECMWF_filelist.txt:")
        print(file_contents)

    try:
        filelist = pd.read_csv(os.path.join(config['dir_temp'], "ECMWF_filelist.txt"), header=None).iloc[:, 0].tolist()
        if len(filelist) == 0:
            print(f"No files found for {str(date_6h)}, skipping")
            continue
    except Exception as e:
        print(f"Unable to obtain file list for {str(date_6h)}, error: {e}")
        continue

    rawoutdir = os.path.join(config["dir_temp"], "ECMWF_IFS_open_forecast", date_6h.strftime("%Y%m%d_%H"))
    if not os.path.exists(rawoutdir):
        os.makedirs(rawoutdir)

    def process_file(filename, rawoutdir, url):
        if os.path.isfile(os.path.join(rawoutdir, filename)):
            return

        print(f"Downloading {filename}")

        download_command = (
            f'wget {url + filename} --directory-prefix={rawoutdir} '
            f'--cut-dirs=20 --no-verbose --no-host-directories --mirror --no-parent --timestamping '
            f'--retry-connrefused --no-check-certificate'
        )
        call(download_command, shell=True)
        call(download_command, shell=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_file, filelist, [rawoutdir] * len(filelist), [url] * len(filelist))

    datacubes = {}
    hours = {}
    data_shape = None

    for var in vars:
        param = var[0]
        print('=================================================================================')
        print(f'Loading all data into a massive 3D datacube for variable {param}')
        files = sorted(glob.glob(os.path.join(rawoutdir, '*.grib2')))

        print(f"Files found for variable {param}: {files}")

        if len(files) == 0:
            print(f"No files found for variable {param}, skipping")
            continue

        for ii, file in enumerate(files):
            print(f"Processing file {file}")
            grbs = pygrib.open(file)
            grb = grbs.read(1)[0]
            if grb.values is None:
                print(f"No values found in {file}, skipping")
                continue

            if data_shape is None:
                data_shape = grb.values.shape
                datacubes[param] = np.zeros((data_shape[0], data_shape[1], len(files)), dtype=np.single) * np.NaN
                hours[param] = np.zeros((len(files),))

            if param not in datacubes:
                datacubes[param] = np.zeros((data_shape[0], data_shape[1], len(files)), dtype=np.single) * np.NaN
                hours[param] = np.zeros((len(files),))

            datacubes[param][:, :, ii] = np.flipud(grb.values)
            grbs.close()
            match = re.search(r'_(\d{3})_', file)
            hours[param][ii] = int(match.group(1)) if match else 0

        datacubes[param] = datacubes[param] * var[3] + var[4]

    for vv, var in enumerate(vars):
        param = var[0]
        if param not in datacubes or datacubes[param] is None:
            continue

        if len(var[11]) > 1:
            if not datacubes[list(datacubes.keys())[0]].shape == datacubes[list(datacubes.keys())[1]].shape:
                print('ERROR: Datacubes for parameters', var[11], 'have different sizes, skipping this variable')
                continue

        hours_combined = hours[list(hours.keys())[0]] if len(var[11]) == 1 else hours[list(hours.keys())[0]]
        if len(var[11]) > 1:
            hours1 = hours[list(hours.keys())[0]]
            hours2 = hours[list(hours.keys())[1]]
            if not (hours1 == hours2).all():
                print('ERROR: Hours for parameters', var[11], 'are inconsistent, skipping this variable')
                continue
            hours_combined = hours1

        if param == 'Wind':
            datacube = np.sqrt(datacubes['u_10m'] ** 2 + datacubes['v_10m'] ** 2)
        else:
            datacube = datacubes[param]

        if var[5] == 'sum':
            print('=================================================================================')
            print(f'Converting accumulations to netCDF for {param}')
            hours_combined = np.concatenate((np.array([0]), hours_combined))
            datacube = np.concatenate((np.zeros((data_shape[0], data_shape[1], 1)), datacube), axis=2)

            for ii_end in np.arange(datacube.shape[2]):
                ii_start = np.where(hours_combined == hours_combined[ii_end] - 1)[0]
                if len(ii_start) > 0:
                    ii_start = ii_start[0]
                    data = datacube[:, :, ii_end] - datacube[:, :, ii_start]
                    ts = date_6h + timedelta(hours=float(hours_combined[ii_start]))
                    filename = os.path.basename(filelist[ii_end]).replace('.grib2', '.nc')
                    outfolder = os.path.join(config['dir_converted'], 'ECMWF', var[0], date_6h.strftime('%Y%m%d_%H'), 'Hourly')
                    print(f"Saving file to: {os.path.join(outfolder, filename)}")
                    os.makedirs(outfolder, exist_ok=True)
                    lb.save_netcdf(os.path.join(outfolder, filename), var[6], data, var[9], ts, float(var[10]), float(var[4]), lat=None, lon=None)

                if hours_combined[ii_end] % 3 == 0:
                    ii_start = np.where(hours_combined == hours_combined[ii_end] - 3)[0]
                    if len(ii_start) > 0:
                        ii_start = ii_start[0]
                        data = datacube[:, :, ii_end] - datacube[:, :, ii_start]
                        ts = date_6h + timedelta(hours=float(hours_combined[ii_start]))
                        filename = os.path.basename(filelist[ii_end]).replace('.grib2', '.nc')
                        outfolder = os.path.join(config['dir_converted'], 'ECMWF', var[0], date_6h.strftime('%Y%m%d_%H'), '3hourly')
                        print(f"Saving file to: {os.path.join(outfolder, filename)}")
                        os.makedirs(outfolder, exist_ok=True)
                        lb.save_netcdf(os.path.join(outfolder, filename), var[6], data, var[8], ts, float(var[10]), float(var[4]), lat=None, lon=None)

                if (date_6h.hour + hours_combined[ii_end]) % 24 == 0:
                    ii_start = np.where(hours_combined == hours_combined[ii_end] - 24)[0]
                    if len(ii_start) > 0:
                        ii_start = ii_start[0]
                        data = datacube[:, :, ii_end] - datacube[:, :, ii_start]
                        ts = date_6h + timedelta(hours=float(hours_combined[ii_start]))
                        filename = os.path.basename(filelist[ii_end]).replace('.grib2', '.nc')
                        outfolder = os.path.join(config['dir_converted'], 'ECMWF', var[0], date_6h.strftime('%Y%m%d_%H'), 'Daily')
                        print(f"Saving file to: {os.path.join(outfolder, filename)}")
                        os.makedirs(outfolder, exist_ok=True)
                        lb.save_netcdf(os.path.join(outfolder, filename), var[6], data, var[7], ts, float(var[10]), float(var[4]), lat=None, lon=None)

        if var[5] == 'mean':
            print('=================================================================================')
            print(f'Converting means to netCDF for {param}')
            if hours_combined[0] != 0:
                hours_combined = np.concatenate((np.array([0]), hours_combined))
                datacube = np.concatenate((datacube[:, :, 0:1], datacube), axis=2)

            for ii_start in np.arange(datacube.shape[2]):
                ii_end = np.where(hours_combined == hours_combined[ii_start] + 1)[0]
                if len(ii_end) > 0:
                    data = datacube[:, :, ii_start]
                    ts = date_6h + timedelta(hours=float(hours_combined[ii_start]))
                    filename = os.path.basename(filelist[ii_start]).replace('.grib2', '.nc')
                    outfolder = os.path.join(config['dir_converted'], 'ECMWF', var[0], date_6h.strftime('%Y%m%d_%H'), 'Hourly')
                    print(f"Saving file to: {os.path.join(outfolder, filename)}")
                    os.makedirs(outfolder, exist_ok=True)
                    lb.save_netcdf(os.path.join(outfolder, filename), var[6], data, var[9], ts, float(var[10]), float(var[4]), lat=None, lon=None)

                if hours_combined[ii_start] % 3 == 0:
                    ii_end = np.where(hours_combined == hours_combined[ii_start] + 3)[0]
                    if len(ii_end) > 0:
                        ii_end = ii_end[0]
                        data = (datacube[:, :, ii_start] + datacube[:, :, ii_end]) / 2
                        ts = date_6h + timedelta(hours=float(hours_combined[ii_start]))
                        filename = os.path.basename(filelist[ii_start]).replace('.grib2', '.nc')
                        outfolder = os.path.join(config['dir_converted'], 'ECMWF', var[0], date_6h.strftime('%Y%m%d_%H'), '3hourly')
                        print(f"Saving file to: {os.path.join(outfolder, filename)}")
                        os.makedirs(outfolder, exist_ok=True)
                        lb.save_netcdf(os.path.join(outfolder, filename), var[6], data, var[8], ts, float(var[10]), float(var[4]), lat=None, lon=None)

                if (date_6h.hour + hours_combined[ii_start]) % 24 == 0:
                    indices = np.where((hours_combined >= hours_combined[ii_start]) & (hours_combined <= hours_combined[ii_start] + 23))[0]
                    if len(indices) >= 8:
                        data = np.mean(datacube[:, :, indices], axis=2)
                        ts = date_6h + timedelta(hours=float(hours_combined[ii_start]))
                        filename = os.path.basename(filelist[ii_start]).replace('.grib2', '.nc')
                        outfolder = os.path.join(config['dir_converted'], 'ECMWF', var[0], date_6h.strftime('%Y%m%d_%H'), 'Daily')
                        print(f"Saving file to: {os.path.join(outfolder, filename)}")
                        os.makedirs(outfolder, exist_ok=True)
                        lb.save_netcdf(os.path.join(outfolder, filename), var[6], data, var[7], ts, float(var[10]), float(var[4]), lat=None, lon=None)

    print('=================================================================================')
    print(f'Forecast downloaded and processed in {time.time() - t1} sec')
    # shutil.rmtree(os.path.join(config["dir_temp"], "ECMWF_IFS_open_forecast", date_6h.strftime("%Y%m%d_%H")))
