import typing as t
import matplotlib.pyplot as plt

ColorMap = t.Literal['jet']
FontFamily = t.Literal[
    # default
    'DejaVu Sans',
    # serif
    'Times' , 'Palatino' , 'Charter' , 'Computer Modern Roman',
    # sans-serif
    'Helvetica' , 'Avant Garde' , 'Computer Modern Serif']

def setup(color: ColorMap, font: FontFamily, font_size: int=10):
    plt.rcParams['image.cmap'] = color
    plt.rcParams["font.family"] = font
    plt.rcParams["font.size"] = font_size
    return plt

plt = setup(color='jet', font='Palatino', font_size=10)
