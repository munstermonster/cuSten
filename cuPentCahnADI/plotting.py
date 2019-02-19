# Andrew Gloster
# February 2019
# Program to plot the s(t) and k_1 functions of a Cahn-Hilliard numerical solution

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

#------------------------------------
# Program begin
#------------------------------------

output = 'output/'

nx = 1024

s_t = []
k1 = []

# Get filenames and times
hdfiles = os.listdir(output)
t = [float(re.findall("\d+\.\d+", file)[0]) for file in hdfiles]

# Reorder depending on times
hdfilesSorted = [x for _, x in sorted(zip(t,hdfiles))]
t.sort()

# Find powers of t
tThird = [x ** (1.0 / 3.0) for x in t]
tTwoThird = [x ** (2.0 / 3.0) for x in t]

k = np.fft.fftfreq(nx, 1.0 / nx)

kx_array = np.zeros((nx - 1, nx - 1), dtype = float)
ky_array = np.zeros((nx - 1, nx - 1), dtype = float)

for row in xrange(nx - 1):

	for column in xrange(nx - 1):

		kx_array[row][column] = k[column + 1]
		ky_array[row][column] = k[row + 1]

modK_inv = 1.0 / np.sqrt(np.square(kx_array) + np.square(ky_array))

for file in hdfilesSorted:
	f = h5py.File(output + file, 'r')
	a_group_key = f.keys()[0]
	data = np.array(f[a_group_key])

	x = np.linspace(0, 2.0 * math.pi, 2 ** 10)
	y = np.linspace(0, 2.0 * math.pi, 2 ** 10)
	avg = (1.0 / ((2.0 * math.pi) ** 2)) * simps(simps(np.square(data), y), x)
	s_t.append(1.0 / (1.0 - avg))

	FT = np.square(np.abs(np.fft.fft2(data)))[1:nx, 1:nx]

	numer = np.sum(FT)

	denom = np.sum(np.multiply(modK_inv, FT))

	k1.append(denom / numer)

plt.loglog(t, s_t, label = 's(t)')
plt.loglog(t, tThird, label = 't^{1/3}')
plt.loglog(t, k1, label = '1 / k_1')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.savefig('analysis.png')









