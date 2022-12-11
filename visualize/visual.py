import glob
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D, axes3d
from pygsp import graphs

plt.rcParams['image.cmap'] = 'jet'
# plt.rcParams['image.cmap'] = 'gist_ncar'

ROOT_DIR = '/path/to/dir'
# WORD_DIR = os.getcwd()
FILE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, 'data')


def draw_map(signal: np.ndarray, save_name: str, vmin_: float, vmax_: float, pdf: bool=True):
    ''' drawing function for u.s. temperature

    Parameters
    ----------
    signal:
        base signal
    save_name:
        name of pdf file that will be saved as ${save_file}.pdf, if pdf = True
    vmin_:
        minimum value of the colorbar that will be put on the left
    vmin_:
        maximum value of the colorbar that will be put on the left
    pdf:
        if true, map will be saved as `pdf`, if else it will be `png`

    '''

    plt.rcParams['image.cmap'] = 'jet'
    temp_dir = os.path.join(DATA_DIR, 'temp_new')
    stations = os.path.join(temp_dir, 'final_station_info.txt')
    df_station = pd.read_csv(stations, dtype=str)

    # 表示する図形のboxの範囲を指定
    extent = [-120, -74, 25, 50]  # Show US Map
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    fig = plt.figure(figsize=(8, 4))  # make map

    ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax.set_extent(extent)

    ax.add_feature(cfeature.LAND, facecolor='none', zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor=cfeature.COLORS['water'], zorder=1)
    ax.add_feature(cfeature.LAKES, facecolor=cfeature.COLORS['water'], edgecolor='lightgray', zorder=2)

    # ax.add_feature(cfeature.RIVERS,zorder=2)
    ax.add_feature(cfeature.BORDERS, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray')

    states_10m = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m',
        edgecolor='lightgray', facecolor='none',  # no filled color
        zorder=2)
    ax.add_feature(states_10m)

    # 緯度・経度のラインを表示する => off
    # ax.gridlines()

    # 各ポイントの緯度・経度を指定する
    lonArr = df_station.Longitude.astype(np.float)
    latArr = df_station.Latitude.astype(np.float)

    # 各ポイントの色を指定できる
    # satLngArr = signal/max(signal)

    cs = ax.scatter(
        lonArr, latArr,
        s=signal*3,  # size
        marker="o",
        c=signal,
        # cmap=plt.cm.jet,
        alpha=0.9,
        transform=ccrs.Geodetic(),
        zorder=10,
        edgecolor='gray',
        linewidths=0.2,
        vmin=vmin_,
        vmax=vmax_,
    )  # Plot

    # set colorbar
    plt.colorbar(cs, pad=0.01, shrink=1)
    if pdf == True:
        plt.savefig(f"{save_name}.pdf", bbox_inches='tight', pad_inches=0.1)
    else:
        plt.savefig(f'{save_name}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.clf()
    plt.close()


def draw_map_diff(dif_signal: np.ndarray, save_name: str, pdf=True):
    """Original scale
    """
    draw_map(dif_signal, save_name, vmin_=0, vmax_=15, pdf=pdf)


def draw_map_sqerror(dif_signal: np.ndarray, save_name: str, pdf=True):
    """This is the result figure used in the paper.
    """
    draw_map(dif_signal**2, save_name, vmin_=0, vmax_=200, pdf=pdf)
