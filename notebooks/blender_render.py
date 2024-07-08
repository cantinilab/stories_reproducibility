import pandas as pd
import numpy as np
import blender_plots as bplt
from matplotlib import cm
from scipy.interpolate import RBFInterpolator

"""
This script belongs in a Blender project. Set the render engine to Cycles
in order to render the scatterplot.
"""

df = pd.read_csv("/home/ghuizing/Documents/midbrain_potential.csv")
points = df[["iso_1", "iso_2", "potential"]].to_numpy()
palette = {"GlioB": "#008941ff", "NeuB": "#ff34ffff", "RGC": "#00bfffff"}
for k in palette.keys():
    h = palette[k][1:]
    palette[k] = list(int(h[i : i + 2], 16) for i in (0, 2, 4))
colors = np.array([palette[k] for k in df["annotation"].values]) / 255

points[:, 2] -= points[:, 2].min()
points[:, 2] *= 3e3
radius = 0.4

points *= 0.035
radius *= 0.03

points[:, 2] -= 32
# Define the kernel type
kernel = "cubic"
smoothing = 0.025

# Define the interpolator.
interpolator = RBFInterpolator(
    points[:, :2],
    points[:, 2],
    smoothing=smoothing,
    kernel=kernel,
)

# Define the plotting limits
xmin, xmax = (
    points[:, 0].max(),
    points[:, 0].min(),
)
ymin, ymax = (
    points[:, 1].min(),
    points[:, 1].max(),
)
zmin, zmax = np.quantile(points[:, 2], [0.01, 0.995])

# Define a grid
x = np.linspace(xmin, xmax, 100)
y = np.linspace(ymin, ymax, 100)
xx, yy = np.meshgrid(x, y)

# Interpolate
zz = interpolator(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Project cells on the interpolation
zz_pred = interpolator(points[:, :2])

# Clip the values
zz = np.clip(zz, zmin, zmax)
zz_pred = np.clip(zz_pred, zmin, zmax)
points[:, 2] = zz_pred + radius

bplt.Surface(xx, yy, zz, name="interp")

bplt.Scatter(points, color=colors, marker_type="spheres", radius=radius)
