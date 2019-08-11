# Andrew Gloster
# February 2019
# Program to make a movie of a Cahn-Hilliard numerical solution

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

#------------------------------------
# Program begin
#------------------------------------

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

def animate(i):
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

	return CS


anim = animation.FuncAnimation(fig, animate, frames=len(hdfilesSorted))

anim.save('basic_animation10.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
anim.save('basic_animation20.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
anim.save('basic_animation30.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
anim.save('basic_animation30.mp4', fps=40, extra_args=['-vcodec', 'libx264'])