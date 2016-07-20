'''
Make demonstration figure for colormap comparisons.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
from colorspacious import cspace_converter


mpl.rcParams.update({'font.size': 14})
mpl.rcParams['font.sans-serif'] = 'Arev Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Helvetica, Avant Garde, sans-serif'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.cal'] = 'cursive'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.tt'] = 'monospace'
mpl.rcParams['mathtext.it'] = 'sans:italic'
mpl.rcParams['mathtext.bf'] = 'sans:bold'
mpl.rcParams['mathtext.sf'] = 'sans'
mpl.rcParams['mathtext.fallback_to_cm'] = 'True'


## Set up figure ##
fig = plt.figure(figsize=(12,10))
# row 1: colormaps
ax1a = plt.subplot2grid((10,10), (0,0), colspan=2, rowspan=1)
ax1b = plt.subplot2grid((10,10), (0,2), colspan=2, rowspan=1)
ax1c = plt.subplot2grid((10,10), (0,4), colspan=2, rowspan=1)
ax1d = plt.subplot2grid((10,10), (0,6), colspan=2, rowspan=1)
ax1e = plt.subplot2grid((10,10), (0,8), colspan=2, rowspan=1)
# row 2: colormaps in grayscale
ax2a = plt.subplot2grid((10,10), (1,0), colspan=2, rowspan=1)
ax2b = plt.subplot2grid((10,10), (1,2), colspan=2, rowspan=1)
ax2c = plt.subplot2grid((10,10), (1,4), colspan=2, rowspan=1)
ax2d = plt.subplot2grid((10,10), (1,6), colspan=2, rowspan=1)
ax2e = plt.subplot2grid((10,10), (1,8), colspan=2, rowspan=1)
# row 3: lightness
ax3a = plt.subplot2grid((10,10), (2,0), colspan=2, rowspan=2)
ax3b = plt.subplot2grid((10,10), (2,2), colspan=2, rowspan=2)
ax3c = plt.subplot2grid((10,10), (2,4), colspan=2, rowspan=2)
ax3d = plt.subplot2grid((10,10), (2,6), colspan=2, rowspan=2)
ax3e = plt.subplot2grid((10,10), (2,8), colspan=2, rowspan=2)
# row 4: perceptual deltas
ax4a = plt.subplot2grid((10,10), (4,0), colspan=2, rowspan=2)
ax4b = plt.subplot2grid((10,10), (4,2), colspan=2, rowspan=2)
ax4c = plt.subplot2grid((10,10), (4,4), colspan=2, rowspan=2)
ax4d = plt.subplot2grid((10,10), (4,6), colspan=2, rowspan=2)
ax4e = plt.subplot2grid((10,10), (4,8), colspan=2, rowspan=2)
# row 5: cone plots
ax5a = plt.subplot2grid((10,10), (6,0), colspan=2, rowspan=2, projection='polar')
ax5b = plt.subplot2grid((10,10), (6,2), colspan=2, rowspan=2, projection='polar')
ax5c = plt.subplot2grid((10,10), (6,4), colspan=2, rowspan=2, projection='polar')
ax5d = plt.subplot2grid((10,10), (6,6), colspan=2, rowspan=2)
ax5e = plt.subplot2grid((10,10), (6,8), colspan=2, rowspan=2, projection='polar')
# row 6: cone plots for color blindness
ax6a = plt.subplot2grid((10,10), (8,0), colspan=2, rowspan=2, projection='polar')
ax6b = plt.subplot2grid((10,10), (8,2), colspan=2, rowspan=2, projection='polar')
ax6c = plt.subplot2grid((10,10), (8,4), colspan=2, rowspan=2, projection='polar')
ax6d = plt.subplot2grid((10,10), (8,6), colspan=2, rowspan=2)
ax6e = plt.subplot2grid((10,10), (8,8), colspan=2, rowspan=2, projection='polar')

## row 1: Plot colormap ##
x = np.linspace(0, 10, 256)
X, _ = np.meshgrid(x, x)
ax1a.text(-0.25, 0.1, 'a', rotation='horizontal', transform=ax1a.transAxes)
# Grayscale
ax1a.pcolor(X, cmap=cmocean.cm.gray)
ax1a.set_xticks([])
ax1a.set_yticks([])
ax1a.axis('tight')
ax1a.set_title('gray')
# Jet
ax1b.pcolor(X, cmap=plt.cm.jet)
ax1b.set_xticks([])
ax1b.set_yticks([])
ax1b.axis('tight')
ax1b.set_title('jet')
# Good sequential colormap: haline
ax1c.pcolor(X, cmap=cmocean.cm.haline)
ax1c.set_xticks([])
ax1c.set_yticks([])
ax1c.axis('tight')
ax1c.set_title('haline')
# Good diverging colormap: balance
ax1d.pcolor(X, cmap=cmocean.cm.balance)
ax1d.set_xticks([])
ax1d.set_yticks([])
ax1d.axis('tight')
ax1d.set_title('balance')
# Circular colormap: phase
ax1e.pcolor(X, cmap=cmocean.cm.phase)
ax1e.set_xticks([])
ax1e.set_yticks([])
ax1e.axis('tight')
ax1e.set_title('phase')


## row 2: Plot colormap in grayscale ##
x = np.linspace(0, 1.0, 256)
ax2a.text(-0.25, 0.1, 'b', rotation='horizontal', transform=ax2a.transAxes)
# Grayscale
# jch = cspace_converter("sRGB1", "JCh")(cmocean.cm.gray(x)[np.newaxis, :, :3])
jch = cspace_converter("sRGB1", "CAM02-UCS")(cmocean.cm.gray(x)[np.newaxis, :, :3])
L = jch[0, :, 0]
L = np.float32(np.vstack((L, L, L)))
ax2a.imshow(L, aspect='auto', cmap=cmocean.cm.gray, vmin=0, vmax=100.)
ax2a.set_xticks([])
ax2a.set_yticks([])
# Jet
jch = cspace_converter("sRGB1", "CAM02-UCS")(plt.cm.jet(x)[np.newaxis, :, :3])
L = jch[0, :, 0]
L = np.float32(np.vstack((L, L, L)))
ax2b.imshow(L, aspect='auto', cmap=cmocean.cm.gray, vmin=0, vmax=100.)
ax2b.set_xticks([])
ax2b.set_yticks([])
# haline
jch = cspace_converter("sRGB1", "CAM02-UCS")(cmocean.cm.haline(x)[np.newaxis, :, :3])
L = jch[0, :, 0]
L = np.float32(np.vstack((L, L, L)))
ax2c.imshow(L, aspect='auto', cmap=cmocean.cm.gray, vmin=0, vmax=100.)
ax2c.set_xticks([])
ax2c.set_yticks([])
# balance
jch = cspace_converter("sRGB1", "CAM02-UCS")(cmocean.cm.balance(x)[np.newaxis, :, :3])
L = jch[0, :, 0]
L = np.float32(np.vstack((L, L, L)))
ax2d.imshow(L, aspect='auto', cmap=cmocean.cm.gray, vmin=0, vmax=100.)
ax2d.set_xticks([])
ax2d.set_yticks([])
# Phase
jch = cspace_converter("sRGB1", "CAM02-UCS")(cmocean.cm.phase(x)[np.newaxis, :, :3])
L = jch[0, :, 0]
L = np.float32(np.vstack((L, L, L)))
ax2e.imshow(L, aspect='auto', cmap=cmocean.cm.gray, vmin=0, vmax=100.)
ax2e.set_xticks([])
ax2e.set_yticks([])

## row 3: Lightness
ax3a.text(-0.25, 0.1, 'c', rotation='horizontal', transform=ax3a.transAxes)
ax3a.text(0.1, 0.9, 'Lightness', transform=ax3a.transAxes)
ymin = -10
ymax = 110
ax3a.set_xticks([])
x = np.linspace(0.0, 1.0, 256)
# Grays
cmap = cmocean.cm.gray
rgb = cmap(x)[np.newaxis, :, :3]
lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
L = lab[0, :, 0]
ax3a.scatter(x, L, c=x, cmap=cmap, s=200, linewidths=0.)
ax3a.set_frame_on(False)
ax3a.set_ylim(ymin, ymax)
ax3a.set_xlim(0, 1)
ax3a.tick_params('y', labelsize=10, right=False)
# Jet
cmap = plt.cm.jet
rgb = cmap(x)[np.newaxis, :, :3]
lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
L = lab[0, :, 0]
ax3b.scatter(x, L, c=x, cmap=cmap, s=200, linewidths=0.)
ax3b.set_frame_on(False)
ax3b.set_ylim(ymin, ymax)
ax3b.set_xlim(0, 1)
ax3b.set_xticks([])
ax3b.set_yticks([])
ax3b.set_yticklabels([])
# haline
cmap = cmocean.cm.haline
rgb = cmap(x)[np.newaxis, :, :3]
lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
L = lab[0, :, 0]
ax3c.scatter(x, L, c=x, cmap=cmap, s=200, linewidths=0.)
ax3c.set_frame_on(False)
ax3c.set_ylim(ymin, ymax)
ax3c.set_xlim(0, 1)
ax3c.set_xticks([])
ax3c.set_yticks([])
ax3c.set_yticklabels([])
# balance
cmap = cmocean.cm.balance
rgb = cmap(x)[np.newaxis, :, :3]
lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
L = lab[0, :, 0]
ax3d.scatter(x, L, c=x, cmap=cmap, s=200, linewidths=0.)
ax3d.set_frame_on(False)
ax3d.set_ylim(ymin, ymax)
ax3d.set_xlim(0, 1)
ax3d.set_xticks([])
ax3d.set_yticks([])
ax3d.set_yticklabels([])
# phase
cmap = cmocean.cm.phase
rgb = cmap(x)[np.newaxis, :, :3]
lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
L = lab[0, :, 0]
ax3e.scatter(x, L, c=x, cmap=cmap, s=200, linewidths=0.)
ax3e.set_xlim(0, 1)
ax3e.set_frame_on(False)
ax3e.set_ylim(ymin, ymax)
ax3e.set_xticks([])
ax3e.set_yticks([])
ax3e.set_yticklabels([])


## row 4: Perceptual deltas
ax4a.text(-0.25, 0.1, 'd', rotation='horizontal', transform=ax4a.transAxes)
ymin = -100
ymax = 700
_sRGB1_to_uniform = cspace_converter("sRGB1", "CAM02-UCS")
x = np.linspace(0.0, 1.0, 256)
ax4a.set_xticks([])
ax4a.set_yticks([])
ax4a.set_ylim(ymin, ymax)
ax4a.set_xlim(0, 1)
ax4a.text(0.03, 0.8, 'Perceptual changes', transform=ax4a.transAxes)
# gray
cmap = cmocean.cm.gray
rgb = cmap(x)[np.newaxis, :, :3]
Jpapbp = _sRGB1_to_uniform(rgb)[0]
local_deltas = 256 * np.sqrt(np.sum((Jpapbp[:-1, :] - Jpapbp[1:, :]) ** 2, axis=-1))
ax4a.scatter(x[1:], local_deltas+100, c=x[1:], cmap=cmap, s=200, linewidths=0.)
ax4a.set_frame_on(False)
# Jet
cmap = plt.cm.jet
rgb = cmap(x)[np.newaxis, :, :3]
Jpapbp = _sRGB1_to_uniform(rgb)[0]
local_deltas = 256 * np.sqrt(np.sum((Jpapbp[:-1, :] - Jpapbp[1:, :]) ** 2, axis=-1))
ax4b.scatter(x[1:], local_deltas, c=x[1:], cmap=cmap, s=200, linewidths=0.)
ax4b.set_xticks([])
ax4b.set_yticks([])
ax4b.set_ylim(ymin, ymax)
ax4b.set_xlim(0, 1)
ax4b.set_frame_on(False)
# haline
cmap = cmocean.cm.haline
rgb = cmap(x)[np.newaxis, :, :3]
Jpapbp = _sRGB1_to_uniform(rgb)[0]
local_deltas = 256 * np.sqrt(np.sum((Jpapbp[:-1, :] - Jpapbp[1:, :]) ** 2, axis=-1))
ax4c.scatter(x[1:], local_deltas+80, c=x[1:], cmap=cmap, s=200, linewidths=0.)
ax4c.set_xticks([])
ax4c.set_yticks([])
ax4c.set_ylim(ymin, ymax)
ax4c.set_xlim(0, 1)
ax4c.set_frame_on(False)
# balance
cmap = cmocean.cm.balance
rgb = cmap(x)[np.newaxis, :, :3]
Jpapbp = _sRGB1_to_uniform(rgb)[0]
local_deltas = 256 * np.sqrt(np.sum((Jpapbp[:-1, :] - Jpapbp[1:, :]) ** 2, axis=-1))
ax4d.scatter(x[1:], local_deltas, c=x[1:], cmap=cmap, s=200, linewidths=0.)
ax4d.set_xticks([])
ax4d.set_yticks([])
ax4d.set_ylim(ymin, ymax)
ax4d.set_xlim(0, 1)
ax4d.set_frame_on(False)
# phase
cmap = cmocean.cm.phase
rgb = cmap(x)[np.newaxis, :, :3]
Jpapbp = _sRGB1_to_uniform(rgb)[0]
local_deltas = 256 * np.sqrt(np.sum((Jpapbp[:-1, :] - Jpapbp[1:, :]) ** 2, axis=-1))
ax4e.scatter(x[1:], local_deltas, c=x[1:], cmap=cmap, s=200, linewidths=0.)
ax4e.set_xticks([])
ax4e.set_yticks([])
ax4e.set_ylim(ymin, ymax)
ax4e.set_xlim(0, 1)
ax4e.set_frame_on(False)

## row 5
ax5a.text(-0.35, 0.1, 'e', rotation='horizontal', transform=ax5a.transAxes)
# cone plot
r = np.linspace(0, 1, 50)
theta = np.linspace(0, 2*np.pi, 100)
z = np.ones((r.size, theta.size)).T*np.linspace(2, 1, r.size)
# Tent plot for diverging colormap
xx = np.linspace(-10, 10, 100)
yy = np.linspace(0, 15, 50)
zz = xx[np.newaxis, :].repeat(yy.size, axis=0)
# Phase plot
azimuths = np.arange(0, 361, 1)
zeniths = np.arange(30, 70, 1)
values = azimuths * np.ones((40, 361))
# Grays
ax5a.pcolormesh(theta, r, z.T, cmap=cmocean.cm.gray)
ax5a.set_xticks([])
ax5a.set_yticks([])
ax5a.set_frame_on(False)
# Jet
ax5b.pcolormesh(theta, r, z.T, cmap=plt.cm.jet)
ax5b.set_xticks([])
ax5b.set_yticks([])
ax5b.set_frame_on(False)
# haline
ax5c.pcolormesh(theta, r, z.T, cmap=cmocean.cm.haline)
ax5c.set_xticks([])
ax5c.set_yticks([])
ax5c.set_frame_on(False)
# balance
ax5d.pcolormesh(xx, yy, zz, cmap=cmocean.cm.balance)
ax5d.set_xticks([])
ax5d.set_yticks([])
ax5d.set_frame_on(False)
ax5d.axis('equal')
# Phase
ax5e.pcolormesh(azimuths*np.pi/180.0, zeniths, values, cmap=cmocean.cm.phase)
ax5e.set_xticks([])
ax5e.set_yticks([])
ax5e.set_frame_on(False)

## row 6: cone plot for color blindness
x = np.linspace(0.0, 1.0, 256)
_deuter50_space = {"name": "sRGB1+CVD",
                   "cvd_type": "deuteranomaly",
                   "severity": 50}
ax6a.text(-0.35, 0.1, 'f', rotation='horizontal', transform=ax6a.transAxes)
# Grays
deut = np.clip(cspace_converter(_deuter50_space, "sRGB1")(cmocean.cm.gray(x)[:, :3]), 0, 1)
cmap = cmocean.tools.cmap(deut)
ax6a.pcolormesh(theta, r, z.T, cmap=cmap)
ax6a.set_xticks([])
ax6a.set_yticks([])
ax6a.set_frame_on(False)
# Jet
deut = np.clip(cspace_converter(_deuter50_space, "sRGB1")(plt.cm.jet(x)[:, :3]), 0, 1)
cmap = cmocean.tools.cmap(deut)
ax6b.pcolormesh(theta, r, z.T, cmap=cmap)
ax6b.set_xticks([])
ax6b.set_yticks([])
ax6b.set_frame_on(False)
# haline
deut = np.clip(cspace_converter(_deuter50_space, "sRGB1")(cmocean.cm.haline(x)[:, :3]), 0, 1)
cmap = cmocean.tools.cmap(deut)
ax6c.pcolormesh(theta, r, z.T, cmap=cmap)
ax6c.set_xticks([])
ax6c.set_yticks([])
ax6c.set_frame_on(False)
# balance
deut = np.clip(cspace_converter(_deuter50_space, "sRGB1")(cmocean.cm.balance(x)[:, :3]), 0, 1)
cmap = cmocean.tools.cmap(deut)
ax6d.pcolormesh(xx, yy, zz, cmap=cmap)
ax6d.set_xticks([])
ax6d.set_yticks([])
ax6d.set_frame_on(False)
ax6d.axis('equal')
# Phase
deut = np.clip(cspace_converter(_deuter50_space, "sRGB1")(cmocean.cm.phase(x)[:, :3]), 0, 1)
cmap = cmocean.tools.cmap(deut)
ax6e.pcolormesh(azimuths*np.pi/180.0, zeniths, values, cmap=cmap)
ax6e.set_xticks([])
ax6e.set_yticks([])
ax6e.set_frame_on(False)

plt.show()
fig.savefig('cmapcomp.png', bbox_inches='tight', dpi=300)
