'''generic.py: Module containing generic data manipulation tools.'''

import numpy as np
import math
import h5py

def averageOverBins(x, y, bins):
    '''Smooth data by averaging over given bins. Essentially a customized
    version of numpy.histogram. Bins anchored on RHS.

    Parameters
    ----------
    x : list(float)
        List of x data.
    y : list(float)
        List of y data.
    bins : list(float)
        RHS location of bins to average over.
	'''
    increasing = True
	if x[-1] < x[0]:
		increasing = False

	yav = np.zeros(len(bins))

	binIndex = 0
	pointsPerBin = 0
	for xIndex,xPoint in enumerate(x):
		if (bins[binIndex] < xPoint and increasing) \
			or (bins[binIndex] > xPoint and not increasing):
			if pointsPerBin != 0:
				yav[binIndex] /= pointsPerBin
			pointsPerBin = 0
			while binIndex < len(bins)-1 \
				and ((bins[binIndex] < xPoint and increasing) \
				or (bins[binIndex] > xPoint and not increasing)):
				binIndex += 1

		yav[binIndex] += y[xIndex]
		pointsPerBin += 1

		if binIndex == len(bins)-1 and xIndex == len(x)-1 and pointsPerBin != 0:
			yav[binIndex] /= pointsPerBin

	return bins, yav

def averageOverBinsLinear(x, y, Nsamples):
	'''Smooth data by averaging over bins spaced linearly in x. Bins are
	anchored on the RHS. Same as "averageOverBinsPowerLaw(x, y, Nsamples, pow=1.0)".
	'''
	dx = float(x[-1] - x[0])/float(Nsamples)
	bins = np.linspace(x[0]+dx, x[-1], Nsamples) # RHS of bins

	bins, yav = averageOverBins(x, y, bins)

	return bins, yav

def averageOverBinsLog(x, y, Nsamples, base=10):
	'''Smooth data by averaging over bins spaced logarithmically in x.'''
	logBins = []
	yav = []
	if x[0] < 0:
		raise Exception('Error in averageOverBinsLog: Can\'t have negative x values.')
	elif x[0] == 0:
		logBins, yav = averageOverBinsLinear(np.log(x[1:])/np.log(base), y[1:], Nsamples)
	else:
		logBins, yav = averageOverBinsLinear(np.log(x)/np.log(base), y, Nsamples)
	return base**logBins, yav

def averageOverBinsPowerLaw(x, y, Nsamples, pow):
	'''Smooth data by averaging over bins spaced using a power law.'''
	N = Nsamples+1
	A = (x[-1] - x[0])/(1.0 - 1.0/N**pow)
	B = x[0] - A/N**pow
	fBins = (A*np.power(np.arange(1,N+1) / float(N), pow) + B)[1:]
	yav = []

	fBins, yav = averageOverBins(x, y, fBins)
	return fBins, yav

def normalize(x, y):
	'''Normalize the function y(x) using a middle-point Riemann sum.'''
	if len(x) != len(y):
		raise Exception('Error in normalize: x and y must have the same length.')
	area = 0
	for i in range(len(x)-1):
		yav = abs(y[i+1]+y[i])/2.0
		area += (x[i+1]-x[i])*yav

	return y/area

def powerSpectrum3D(vx, vy, vz, L=1.0):
	'''Calculate power spectrum E(k), where k is the radial wavenumber. The 
	returned dataset will have many more points for high k than for low k, 
	so it helps to average using averageOverBinsLinear.

    Parameters
    ----------
    vx, vy, vz : numpy.ndarray
        3D arrays containing the x, y, and z components of the velocity at each cell.
	'''
	N = len(vx[0,0])
	dx = L/float(N)
	freq = 2*math.pi*np.fft.fftfreq(N, dx) # 2pi factor converts freq to wavenumber
	Etrans = 0.5*(np.fft.rfftn(vx)**2 + np.fft.rfftn(vy)**2 + np.fft.rfftn(vz)**2)

	kList = []
	EtransList = []
	for k in range(Etrans.shape[0]):
		for j in range(Etrans.shape[1]):
			for i in range(Etrans.shape[2]):
				kmag = math.sqrt(freq[i]**2 + freq[j]**2 + freq[k]**2)
				kList.append(kmag)
				EtransList.append(4*math.pi*kmag**2*np.abs(Etrans[k,j,i]))

    # sorts by increasing radial wavenumber
	sortedList = sorted(zip(kList,EtransList))
	kList = np.array([x for x,_ in sortedList])
	EtransList = np.array([x for _,x in sortedList])

	return kList, EtransList


# Reading and writing data.

def save(fileName, data, descr=None, *attrs):
	'''Saves data and metadata describing its contents in a .hdf file.'''
	if descr is None:
		descrText = ''
	else:
		descrText = descr

	dataset = h5py.File(fileName, 'w')
	dataset['Data'] = data
	dataset.attrs['Description'] = descrText

	# Add an arbitrary number of other attributes
	for x in attrs:
		dataset.attrs[x[0]] = x[1]

	dataset.close()

def load(fileName):
	'''Loads data from .hdf file, using same format as generic.save. Ignores 
	attributes other than "Description".
	'''
	dataset = h5py.File(fileName, 'r')
	data = dataset['Data'][()]
	descr = dataset.attrs['Description']
	dataset.close()
	return descr, data
