'''
Plot colormaps with data and model as examples.
'''

import numpy as np
import matplotlib.pyplot as plt
import tracpy
import tracpy.plotting
import netCDF4 as netCDF
import cmocean.cm as cmo
import matplotlib as mpl
from datetime import datetime
from matplotlib.pylab import find
from scipy import spatial
from matplotlib import colors, colorbar, cm
from matplotlib.path import Path
import op


mpl.rcParams.update({'font.size': 10})
mpl.rcParams['font.sans-serif'] = 'Arev Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Helvetica, Avant Garde, sans-serif'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.cal'] = 'cursive'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.tt'] = 'monospace'
mpl.rcParams['mathtext.it'] = 'sans:italic'
mpl.rcParams['mathtext.bf'] = 'sans:bold'
mpl.rcParams['mathtext.sf'] = 'sans'
mpl.rcParams['mathtext.fallback_to_cm'] = 'True'


## Get data ##
fname = 'MS03_L15.txt'
d = np.loadtxt(fname, comments='*')
var = ['lat', 'lon', 'depth', 'temp', 'density', 'sigma', 'oxygen',
       'voltage 2', 'voltage 3', 'voltage 4', 'fluorescence-CDOM', 'fluorescence-ECO',
       'turbidity', 'pressure', 'salinity', 'RINKO temperature',
       'RINKO DO - CTD temp', 'RINKO DO - RINKO temp', 'bottom', 'PAR']
lat = d[:, 0]
lon = d[:, 1]
z = d[:, 2]
h = d[:, 18]
pressure = d[:, var.index('pressure')]

# check when pressure change is positive vs. negative to select points to plot
pdiff = np.diff(pressure)
idown = pdiff > 0  # going down
iup = pdiff < 0  # going up

z0line = True
gradient = False

## Get model output ##
# loc = 'http://barataria.tamu.edu:6060/thredds/dodsC/NcML/txla_nesting6.nc'
loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_nesting6.nc'
proj = tracpy.tools.make_proj('nwgom', llcrnrlat=27, urcrnrlat=30.5, llcrnrlon=-98)
grid = tracpy.inout.readgrid(loc, proj)
xpsi = grid.x_psi
ypsi = grid.y_psi
xr = grid.x_rho[1:-1, 1:-1]
yr = grid.y_rho[1:-1, 1:-1]
m = netCDF.Dataset(loc)
t = m.variables['ocean_time']
# datetime
datetimedata = datetime(2011, 6, 26, 5, 5)  # UTC
tdata = netCDF.date2num(datetimedata, t.units)
# dates = netCDF.num2date(t[:], t.units)
ind = find(t[:] < tdata)[-1]
# Free Surface
ssh = np.squeeze(m.variables['zeta'][ind, 1:-1, 1:-1])

# Calculate along-track distance
x, y = proj(lon, lat)
dist = np.sqrt((x-x[0])**2 + (y-y[0])**2)/1000.  # in km
dist = dist.max() - dist  # reverse to keep plot orientation the same

# Path around the dataset to limit voronoi regions
minz = []
maxz = []
distz = []
dinds = 1000  # looking for the max values over this many indices
counter = 0
while counter < z.size:
    # distz does not align exactly, just taking the mean of the values
    distz.append(dist[counter:counter+dinds].mean())
    minz.append(z[counter:counter+dinds].min())
    maxz.append(z[counter:counter+dinds].max())
    counter += dinds
pathpts = np.concatenate((np.vstack((distz, minz)).T, np.vstack((distz[::-1], maxz[::-1])).T))
path = Path(pathpts)

# Make Voronoi diagram for plotting
pts = np.vstack((dist[1:][iup], z[1:][iup])).T
vor = spatial.Voronoi(pts)

## Set up figure ##
if gradient:
    ncols = 3
    fig = plt.figure(figsize=(15, 10))
    # row 1: TXLA top half with Free Surface and Temperature
    ax1a = fig.add_subplot(4, ncols, 1)  # Free Surface
    ax1b = fig.add_subplot(4, ncols, 2)  # Free Surface, jet
    ax1c = fig.add_subplot(4, ncols, 3)  # Free Surface gradient
    # row 2
    ax2a = fig.add_subplot(4, ncols, 4)  # Temperature
    ax2b = fig.add_subplot(4, ncols, 5)  # Temperature, jet
    ax2c = fig.add_subplot(4, ncols, 6)  # Temperature gradient
    # row 3
    ax3a = fig.add_subplot(4, ncols, 7)  # Salinity
    ax3b = fig.add_subplot(4, ncols, 8)  # Salinity, jet
    ax3c = fig.add_subplot(4, ncols, 9)  # Salinity gradient
    # row 4
    ax4a = fig.add_subplot(4, ncols, 10)  # Oxygen
    ax4b = fig.add_subplot(4, ncols, 11)  # Oxygen, jet
    ax4c = fig.add_subplot(4, ncols, 12)  # Oxygen gradient
else:
    ncols = 2
    fig = plt.figure(figsize=(10, 10))
    # row 1: TXLA top half with Free Surface and Temperature
    ax1a = fig.add_subplot(4, ncols, 1)  # Free Surface
    ax1b = fig.add_subplot(4, ncols, 2)  # Free Surface, jet
    # row 2
    ax2a = fig.add_subplot(4, ncols, 3)  # Temperature
    ax2b = fig.add_subplot(4, ncols, 4)  # Temperature, jet
    # row 3
    ax3a = fig.add_subplot(4, ncols, 5)  # Salinity
    ax3b = fig.add_subplot(4, ncols, 6)  # Salinity, jet
    # row 4
    ax4a = fig.add_subplot(4, ncols, 7)  # Oxygen
    ax4b = fig.add_subplot(4, ncols, 8)  # Oxygen, jet
fig.subplots_adjust(left=0.05, bottom=0.03, right=0.99, top=0.99, wspace=0.04, hspace=0.08)


## TXLA Plot of top half of domain in jet ##
tracpy.plotting.background(grid, ax=ax1a, outline=[0, 0, 0, 0], mers=np.arange(-96, -86, 2), hlevs=[0])
map1a = ax1a.pcolormesh(xpsi, ypsi, ssh, cmap='jet', vmin=-0.1, vmax=0.1)
# overlay contours
ax1a.contour(xr, yr, ssh, levels=[-0.025, -0.05], colors='k', linestyles='-.', alpha=0.5, linewidths=0.5)
ax1a.contour(xr, yr, ssh, levels=[0.025, 0.05], colors='k', linestyles='-', alpha=0.5, linewidths=0.35)
ax1a.set_frame_on(False)  # kind of like it without the box
ax1a.text(0.9, 0.01, 'a', color='0.1', transform=ax1a.transAxes, fontsize=12)  # Label subplot
# Plot the track
ax1a.plot(x, y, '0.2', lw=2)
n = x.size/2
ax1a.plot([x[n], 891975], [y[n], 146818], '-', color='0.2', lw=0.5)
ax1a.text(891975-10000, 146818-10000, 'Ship track', color='0.1')
if gradient:
    cax1a = fig.add_axes([0.0625, 0.94, 0.065, 0.015])  # colorbar axes
else:
    cax1a = fig.add_axes([0.07, 0.94, 0.1, 0.015])  # colorbar axes
cb1a = fig.colorbar(map1a, cax=cax1a, orientation='horizontal')
cb1a.set_label('Free surface [m]', color='0.2')
cb1a.ax.tick_params(labelsize=10, length=2, color='0.2', labelcolor='0.2')
cb1a.set_ticks(np.arange(-0.1, 0.2, 0.1))

## TXLA Plot of top half of domain ##
tracpy.plotting.background(grid, ax=ax1b, outline=[0, 0, 0, 0], mers=np.arange(-96, -86, 2), parslabels=[0,0,0,0], hlevs=[0])
map1b = ax1b.pcolormesh(xpsi, ypsi, ssh, cmap=cmo.balance, vmin=-0.1, vmax=0.1)
# overlay contours
ax1b.contour(xr, yr, ssh, levels=[-0.025, -0.05], colors='k', linestyles='-.', alpha=0.5, linewidths=0.5)
ax1b.contour(xr, yr, ssh, levels=[0.025, 0.05], colors='k', linestyles='-', alpha=0.5, linewidths=0.35)
ax1b.set_frame_on(False)  # kind of like it without the box
ax1b.text(0.9, 0.01, 'b', color='0.1', transform=ax1b.transAxes, fontsize=12)  # Label subplot
# Plot the track
ax1b.plot(x, y, '0.2', lw=2)
n = x.size/2
ax1b.plot([x[n], 891975], [y[n], 146818], '-', color='0.2', lw=0.5)
ax1b.text(891975-10000, 146818-10000, 'Ship track', color='0.1')
if gradient:
    cax1b = fig.add_axes([0.38, 0.94, 0.065, 0.015])  # colorbar axes
else:
    cax1b = fig.add_axes([0.55, 0.94, 0.1, 0.015])  # colorbar axes
cb1b = fig.colorbar(map1b, cax=cax1b, orientation='horizontal')
cb1b.set_label('Free surface [m]', color='0.2')
cb1b.ax.tick_params(labelsize=10, length=2, color='0.2', labelcolor='0.2')
cb1b.set_ticks(np.arange(-0.1, 0.2, 0.1))

if gradient:
    ## TXLA Plot of top half of domain - gradients
    grad1 = (ssh[1:,:] - ssh[:-1,:])/(yr[1:,:] - yr[:-1,:])
    grad2 = (ssh[:,1:] - ssh[:,:-1])/(xr[:,1:] - xr[:,:-1])
    grad = abs(np.sqrt(op.resize(grad1,1)**2 + op.resize(grad2,0)**2))
    tracpy.plotting.background(grid, ax=ax1c, outline=[0, 0, 0, 0], mers=np.arange(-96, -86, 2), parslabels=[0,0,0,0], hlevs=[0])
    map1c = ax1c.pcolormesh(xr, yr, grad, cmap=cmo.matter, vmin=0, vmax=0.000008)
    ax1c.set_frame_on(False)  # kind of like it without the box
    ax1c.text(0.9, 0.01, 'c', color='0.1', transform=ax1c.transAxes, fontsize=12)  # Label subplot

# Temperature, in jet
cmap = cm.jet
data = d[:, var.index('temp')]
# http://matplotlib.org/1.2.1/examples/pylab_examples/hist_colormapped.html
# we need to normalize the data to 0..1 for the full range of the colormap
fracs = data[1:][iup]  # /data[1:][iup].max()
norm = colors.Normalize(fracs.min(), fracs.max())
# loop through the voronoi regions by data point and associated color fraction
for frac, point_region in zip(fracs, vor.point_region):
    color = cmap(norm(frac))  # which color in cmap to use
    indices = vor.regions[point_region]  # get relevant region indices
    if not indices: continue     # check for empty regions
    if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
    # exclude vertices that exit data path
    regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
    # require all to be within path to plot
    if not path.contains_points(regionpts).sum() == len(indices):
        continue
    # plot voronoi regions with temp data
    ax2a.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')
if z0line:
    ax2a.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)
ax2a.plot(dist, h, '0.2', lw=1)
ax2a.set_xticklabels([])
ax2a.set_ylabel('Depth [m]')
ax2a.tick_params(right='off', top='off')
ax2a.set_frame_on(False)  # kind of like it without the box
ax2a.set_ylim(h.max(), 0)
ax2a.set_xlim(dist.min(), dist.max())
# colorbar
if gradient:
    ax2a.text(0.94, 0.05, 'd', color='0.1', transform=ax2a.transAxes, fontsize=12)  # Label subplot
    cax2a = fig.add_axes([0.19, 0.57, 0.13, 0.02])  # colorbar axes
else:
    ax2a.text(0.94, 0.05, 'c', color='0.1', transform=ax2a.transAxes, fontsize=12)  # Label subplot
    cax2a = fig.add_axes([0.265, 0.57, 0.175, 0.02])  # colorbar axes
cb2a = colorbar.ColorbarBase(cax2a, cmap=cmap, norm=norm, orientation='horizontal')
cb2a.set_label('Temperature [C]', color='0.2')
cb2a.ax.tick_params(labelsize=10, length=2, color='0.2', labelcolor='0.2')
cb2a.set_ticks(np.arange(22, 32, 2))

## Temperature ##
data = d[:, var.index('temp')]
cmap = cmo.thermal
# http://matplotlib.org/1.2.1/examples/pylab_examples/hist_colormapped.html
# we need to normalize the data to 0..1 for the full range of the colormap
fracs = data[1:][iup] #/data[1:][iup].max()
norm = colors.Normalize(fracs.min(), fracs.max())
# loop through the voronoi regions by data point and associated color fraction
for frac, point_region in zip(fracs, vor.point_region):
    color = cmap(norm(frac))  # which color in cmap to use
    indices = vor.regions[point_region]  # get relevant region indices
    if not indices: continue     # check for empty regions
    if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
    # exclude vertices that exit data path
    regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
    # require all to be within path to plot
    if not path.contains_points(regionpts).sum() == len(indices):
        continue
    # plot voronoi regions with temp data
    ax2b.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')

if z0line:
    ax2b.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)
ax2b.plot(dist, h, '0.2', lw=1)
ax2b.set_xticklabels([])
ax2b.tick_params(right='off', top='off')
ax2b.set_frame_on(False)  # kind of like it without the box
ax2b.set_ylim(h.max(), 0)
ax2b.set_yticklabels([])
ax2b.set_xlim(dist.min(), dist.max())
# colorbar
if gradient:
    ax2b.text(0.94, 0.05, 'e', color='0.1', transform=ax2b.transAxes, fontsize=12)  # Label subplot
    cax2b = fig.add_axes([0.51, 0.57, 0.13, 0.02])  # colorbar axes
else:
    ax2b.text(0.94, 0.05, 'd', color='0.1', transform=ax2b.transAxes, fontsize=12)  # Label subplot
    cax2b = fig.add_axes([0.75, 0.57, 0.175, 0.02])  # colorbar axes
# http://matplotlib.org/examples/api/colorbar_only.html
cb2b = colorbar.ColorbarBase(cax2b, cmap=cmap, norm=norm, orientation='horizontal')
cb2b.set_label('Temperature [C]', color='0.2')
cb2b.ax.tick_params(labelsize=10, length=2, color='0.2', labelcolor='0.2')
cb2b.set_ticks(np.arange(22, 32, 2))

if gradient:
    # Temperature gradients
    # calculating gradients in the data
    data = d[:, var.index('temp')]
    zdiff = np.diff(z)
    grad = abs(np.diff(data)/zdiff)
    cmap = cmo.matter
    fracs = grad[iup]  # /data[1:][iup].max()
    norm = colors.Normalize(0, 4)  # fracs.min(), fracs.max())
    # loop through the voronoi regions by data point and associated color fraction
    for frac, point_region in zip(fracs, vor.point_region):
        color = cmap(norm(frac))  # which color in cmap to use
        indices = vor.regions[point_region]  # get relevant region indices
        if not indices: continue     # check for empty regions
        if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
        # exclude vertices that exit data path
        regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
        # require all to be within path to plot
        if not path.contains_points(regionpts).sum() == len(indices):
            continue
        # plot voronoi regions with temp data
        ax2c.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')
    if z0line:
        ax2c.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)
    ax2c.plot(dist, h, '0.2', lw=1)
    ax2c.set_xticklabels([])
    ax2c.set_yticklabels([])
    ax2c.tick_params(right='off', top='off')
    ax2c.set_frame_on(False)  # kind of like it without the box
    ax2c.text(0.94, 0.05, 'f', color='0.1', transform=ax2c.transAxes, fontsize=12)  # Label subplot
    ax2c.set_ylim(h.max(), 0)
    ax2c.set_xlim(dist.min(), dist.max())

# Salinity, jet
cmap = cm.jet
data = d[:, var.index('salinity')]
# http://matplotlib.org/1.2.1/examples/pylab_examples/hist_colormapped.html
# we need to normalize the data to 0..1 for the full range of the colormap
# fracs = data[1:][iup]/34.
fracs = data[1:][iup]  # /data[1:][iup].max()
norm = colors.Normalize(fracs.min(), fracs.max())
# loop through the voronoi regions by data point and associated color fraction
for frac, point_region in zip(fracs, vor.point_region):
    color = cmap(norm(frac))  # which color in cmap to use
    indices = vor.regions[point_region]  # get relevant region indices
    if not indices: continue     # check for empty regions
    if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
    # exclude vertices that exit data path
    regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
    # require all to be within path to plot
    if not path.contains_points(regionpts).sum() == len(indices):
        continue
    # plot voronoi regions with temp data
    ax3a.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')
if z0line:
    ax3a.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)
ax3a.plot(dist, h, '0.2', lw=1)
ax3a.set_xticklabels([])
ax3a.set_ylabel('Depth [m]')
ax3a.tick_params(right='off', top='off')
ax3a.set_frame_on(False)  # kind of like it without the box
ax3a.set_ylim(h.max(), 0)
ax3a.set_xlim(dist.min(), dist.max())
if gradient:
    ax3a.text(0.94, 0.05, 'g', color='0.1', transform=ax3a.transAxes, fontsize=12)  # Label subplot
    cax3a = fig.add_axes([0.19, 0.32, 0.13, 0.02])  # colorbar axes
else:
    ax3a.text(0.94, 0.05, 'e', color='0.1', transform=ax3a.transAxes, fontsize=12)  # Label subplot
    cax3a = fig.add_axes([0.265, 0.32, 0.175, 0.02])  # colorbar axes
cb3a = colorbar.ColorbarBase(cax3a, cmap=cmap, norm=norm, orientation='horizontal')
cb3a.set_label('Practical Salinity [PSS-78]', color='0.2')
cb3a.ax.tick_params(labelsize=10, length=2, color='0.2', labelcolor='0.2')

## Salinity ##
cmap = cmo.haline
data = d[:, var.index('salinity')]
# http://matplotlib.org/1.2.1/examples/pylab_examples/hist_colormapped.html
# we need to normalize the data to 0..1 for the full range of the colormap
# fracs = data[1:][iup]/34.
fracs = data[1:][iup] #/data[1:][iup].max()
norm = colors.Normalize(fracs.min(), fracs.max())
# loop through the voronoi regions by data point and associated color fraction
for frac, point_region in zip(fracs, vor.point_region):
    color = cmap(norm(frac))  # which color in cmap to use
    indices = vor.regions[point_region]  # get relevant region indices
    if not indices: continue     # check for empty regions
    if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
    # exclude vertices that exit data path
    regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
    # require all to be within path to plot
    if not path.contains_points(regionpts).sum() == len(indices):
        continue
    # plot voronoi regions with temp data
    ax3b.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')
if z0line:
    ax3b.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)
ax3b.plot(dist, h, '0.2', lw=1)
ax3b.set_xticklabels([])
ax3b.set_yticklabels([])
ax3b.tick_params(right='off', top='off')
ax3b.set_frame_on(False)  # kind of like it without the box
ax3b.set_ylim(h.max(), 0)
ax3b.set_xlim(dist.min(), dist.max())
if gradient:
    ax3b.text(0.94, 0.05, 'h', color='0.1', transform=ax3b.transAxes, fontsize=12)  # Label subplot
    cax3b = fig.add_axes([0.51, 0.32, 0.13, 0.02])  # colorbar axes
else:
    ax3b.text(0.94, 0.05, 'f', color='0.1', transform=ax3b.transAxes, fontsize=12)  # Label subplot
    cax3b = fig.add_axes([0.75, 0.32, 0.175, 0.02])  # colorbar axes
cb3b = colorbar.ColorbarBase(cax3b, cmap=cmap, norm=norm, orientation='horizontal')
cb3b.set_label('Practical Salinity [PSS-78]', color='0.2')
cb3b.ax.tick_params(labelsize=10, length=2, color='0.2', labelcolor='0.2')

if gradient:
    # Salinity gradient
    # calculating gradients in the data
    data = d[:, var.index('salinity')]
    grad = abs(np.diff(data)/zdiff)
    cmap = cmo.matter
    fracs = grad[iup]  # /data[1:][iup].max()
    norm = colors.Normalize(0, 15)  # fracs.min(), fracs.max())
    # loop through the voronoi regions by data point and associated color fraction
    for frac, point_region in zip(fracs, vor.point_region):
        color = cmap(norm(frac))  # which color in cmap to use
        indices = vor.regions[point_region]  # get relevant region indices
        if not indices: continue     # check for empty regions
        if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
        # exclude vertices that exit data path
        regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
        # require all to be within path to plot
        if not path.contains_points(regionpts).sum() == len(indices):
            continue
        # plot voronoi regions with temp data
        ax3c.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')
    if z0line:
        ax3c.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)
    ax3c.plot(dist, h, '0.2', lw=1)
    ax3c.set_xticklabels([])
    ax3c.set_yticklabels([])
    ax3c.tick_params(right='off', top='off')
    ax3c.set_frame_on(False)  # kind of like it without the box
    ax3c.text(0.94, 0.05, 'i', color='0.1', transform=ax3c.transAxes, fontsize=12)  # Label subplot
    ax3c.set_ylim(h.max(), 0)
    ax3c.set_xlim(dist.min(), dist.max())

# Oxygen in jet
cmap = cm.jet
data = d[:, var.index('oxygen')]
# http://matplotlib.org/1.2.1/examples/pylab_examples/hist_colormapped.html
# we need to normalize the data to 0..1 for the full range of the colormap
fracs = data[1:][iup]  # /data[1:][iup].max()
norm = colors.Normalize(fracs.min(), fracs.max())
# loop through the voronoi regions by data point and associated color fraction
for frac, point_region in zip(fracs, vor.point_region):
    color = cmap(norm(frac))  # which color in cmap to use
    indices = vor.regions[point_region]  # get relevant region indices
    if not indices: continue     # check for empty regions
    if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
    # exclude vertices that exit data path
    regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
    # require all to be within path to plot
    if not path.contains_points(regionpts).sum() == len(indices):
        continue
    # plot voronoi regions with temp data
    ax4a.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')
ax4a.plot(dist, h, '0.2', lw=1)
ax4a.tick_params(right='off', top='off')
ax4a.set_xlabel('Distance [km]')
ax4a.set_ylabel('Depth [m]')
ax4a.set_frame_on(False)  # kind of like it without the box
ax4a.set_ylim(h.max(), 0)
ax4a.set_xlim(dist.min(), dist.max())
if z0line:
    ax4a.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)
if gradient:
    ax4a.text(0.94, 0.05, 'j', color='0.1', transform=ax4a.transAxes, fontsize=12)  # Label subplot
    cax4a = fig.add_axes([0.19, 0.075, 0.13, 0.02])  # colorbar axes
else:
    ax4a.text(0.94, 0.05, 'g', color='0.1', transform=ax4a.transAxes, fontsize=12)  # Label subplot
    cax4a = fig.add_axes([0.265, 0.075, 0.175, 0.02])  # colorbar axes
cb4a = colorbar.ColorbarBase(cax4a, cmap=cmap, norm=norm, orientation='horizontal')
cb4a.set_label('Oxygen [ml/l]', color='0.2')
cb4a.ax.tick_params(labelsize=10, length=2, color='0.2', labelcolor='0.2')
locs = np.linspace(data[1:][iup].min(), data.max(), 6)
cb4a.set_ticks(locs)
cb4a.set_ticklabels(np.arange(0, 12, 2))

# Oxygen
cmap = cmo.oxy
data = d[:, var.index('oxygen')]
# http://matplotlib.org/1.2.1/examples/pylab_examples/hist_colormapped.html
# we need to normalize the data to 0..1 for the full range of the colormap
fracs = data[1:][iup]  # /data[1:][iup].max()
norm = colors.Normalize(fracs.min(), fracs.max())
# loop through the voronoi regions by data point and associated color fraction
for frac, point_region in zip(fracs, vor.point_region):
    color = cmap(norm(frac))  # which color in cmap to use
    indices = vor.regions[point_region]  # get relevant region indices
    if not indices: continue     # check for empty regions
    if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
    # exclude vertices that exit data path
    regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
    # require all to be within path to plot
    if not path.contains_points(regionpts).sum() == len(indices):
        continue
    # plot voronoi regions with temp data
    ax4b.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')
if z0line:
    ax4b.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)
ax4b.plot(dist, h, '0.2', lw=1)
ax4b.tick_params(right='off', top='off')
ax4b.set_xlabel('Distance [km]')
ax4b.set_yticklabels([])
ax4b.set_frame_on(False)  # kind of like it without the box
ax4b.set_ylim(h.max(), 0)
ax4b.set_xlim(dist.min(), dist.max())
if gradient:
    ax4b.text(0.94, 0.05, 'k', color='0.1', transform=ax4b.transAxes, fontsize=12)  # Label subplot
    cax4b = fig.add_axes([0.51, 0.075, 0.13, 0.02])  # colorbar axes
else:
    ax4b.text(0.94, 0.05, 'h', color='0.1', transform=ax4b.transAxes, fontsize=12)  # Label subplot
    cax4b = fig.add_axes([0.75, 0.075, 0.175, 0.02])  # colorbar axes
cb4b = colorbar.ColorbarBase(cax4b, cmap=cmap, norm=norm, orientation='horizontal')
cb4b.set_label('Oxygen [ml/l]', color='0.2')
cb4b.ax.tick_params(labelsize=10, length=2, color='0.2', labelcolor='0.2')
locs = np.linspace(data[1:][iup].min(), data.max(), 6)
cb4b.set_ticks(locs)
cb4b.set_ticklabels(np.arange(0, 12, 2))

if gradient:
    # Oxygen gradients
    # calculating gradients in the data
    data = d[:, var.index('oxygen')]
    grad = abs(np.diff(data)/zdiff)
    cmap = cmo.matter
    fracs = grad[iup]  # /data[1:][iup].max()
    norm = colors.Normalize(0, 8)  # fracs.min(), fracs.max())
    # loop through the voronoi regions by data point and associated color fraction
    for frac, point_region in zip(fracs, vor.point_region):
        color = cmap(norm(frac))  # which color in cmap to use
        indices = vor.regions[point_region]  # get relevant region indices
        if not indices: continue     # check for empty regions
        if -1 in indices: continue   # region includes a vertex out of the diagram (the region goes to infinity)
        # exclude vertices that exit data path
        regionpts = np.vstack((vor.vertices[indices, 0], vor.vertices[indices, 1])).T
        # require all to be within path to plot
        if not path.contains_points(regionpts).sum() == len(indices):
            continue
        # plot voronoi regions with temp data
        ax4c.fill(vor.vertices[indices, 0], vor.vertices[indices, 1], color=color, edgecolor='none')
    ax4c.plot(dist, h, '0.2', lw=1)
    ax4c.set_yticklabels([])
    ax4c.tick_params(right='off', top='off')
    ax4c.set_xlabel('Distance [km]')
    ax4c.set_frame_on(False)  # kind of like it without the box
    ax4c.text(0.94, 0.05, 'l', color='0.1', transform=ax4c.transAxes, fontsize=12)  # Label subplot
    ax4c.set_ylim(h.max(), 0)
    ax4c.set_xlim(dist.min(), dist.max())
    if z0line:
        ax4c.plot([dist.min(), dist.max()], [0, 0], '0.1', linestyle=':', lw=0.5, alpha=0.7)

fig.savefig('examples.png', bbox_inches='tight', dpi=300)
