# Andrew Gloster
# Februay 2018
# Plot stills of 2D Cahn-Hilliard

#------------------------------------
# Import relevant modules
#------------------------------------

import os
from subprocess import Popen, PIPE
import h5py
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import re
import math
from scipy.integrate import simps
from matplotlib import animation

# ---------------------------------------------------------------------
# Begin Program
# ---------------------------------------------------------------------

output = 'output/'

nx = 512

# Get filenames and times
hdfiles = os.listdir(output)
t = [float(re.findall("\d+\.\d+", file)[0]) for file in hdfiles]

# Reorder depending on times
hdfilesSorted = [x for _, x in sorted(zip(t,hdfiles))]
t.sort()

# Create x and y
x = np.linspace(0.0, 2.0 * math.pi, nx)
y = np.linspace(0.0, 2.0 * math.pi, nx)

# Create the figure
fig = plt.figure()

def Plot(i):
	file = hdfilesSorted[i]
	f = h5py.File(output + file, 'r')
	a_group_key = f.keys()[0]
	data = np.array(f[a_group_key])

	plt.clf()
	CS = plt.contourf(x, y, data, 50, vmin = - 1.0, vmax = 1.0, cmap=cm.seismic)
	plt.xlabel('X')
	plt.ylabel('Y')
	m = plt.cm.ScalarMappable(cmap=cm.seismic)
	m.set_array(data)
	m.set_clim(-1.0, 1.0)
	plt.colorbar(m)
	plt.savefig('contour' + str(i) + '.png', dpi = 300)

	print(file)

	return CS

Plot(10)
Plot(30)
Plot(50)
Plot(70)
