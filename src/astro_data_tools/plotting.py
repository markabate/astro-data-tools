'''plotting.py: Module containing helper functions for plotting.'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import athena
import generic

def standardFormat():
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

def athenaPlotHeatMap(fileName, var, meshblocksDim, sliceAxis, sliceCoord=0.0,
                      L=1.0, title='', log=False, ax=None):
	'''Plot heat map directly from an athdf file. Loads mesh and then calls
	plotHeatMap.
	'''
	mesh = athena.loadFullMesh(fileName, var)
	return plotHeatMap(mesh, sliceAxis, slicecoord, L, title, log, ax)

def plotHeatMap(mesh, sliceAxis, sliceCoord=0.0, L=1.0, title='', log=False, ax=None):
	'''Plot heat map of a square 3-dimensional mesh, sliced along some axis.
	'''
	N = len(mesh[0,0])
	if sliceAxis == 'x':
		tmpMesh = np.transpose(mesh, [2,0,1])
		axesLabels = [r'$y$', r'$z$']
	elif sliceAxis == 'y':
		tmpMesh = np.transpose(mesh, [1,0,2])
		axesLabels = [r'$x$', r'$z$']
	elif sliceAxis == 'z':
		tmpMesh = np.transpose(mesh, [0,1,2])
		axesLabels = [r'$x$', r'$y$']
	else:
		raise Exception('Error in plotHeatMap: Invalid slice axis.')

	sliceCoordAdj = sliceCoord + L/2.0
	sliceIndex = int(sliceCoordAdj*N/L)
	if sliceIndex <= 0:
		grid2D = tmpMesh[0]
	elif sliceIndex >= N-1:
		grid2D = tmpMesh[-1]
	elif sliceCoordAdj*N/L % 1 == 0:
		grid2D = (tmpMesh[sliceIndex-1] + tmpMesh[sliceIndex])/2.0
	else:
		grid2D = tmpMesh[sliceIndex]

	if ax is None:
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
	else:
		ax1 = ax

	# Handle ticks
	ax1.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)

	axisRange = np.linspace(-0.5, N-0.5, 6)
	axisTickLabels = [('%.2f' % x).rstrip('0').rstrip('.') for x in np.linspace(-L/2.0, L/2.0, 6)]

	if log:
		heatmap = ax1.imshow(grid2D, origin='lower', norm=matplotlib.colors.LogNorm())  # imshow plots first axis along y, second along x
	else:
		heatmap = ax1.imshow(grid2D, origin='lower')
	ax1.set_xticks(axisRange)
	ax1.set_xticklabels(axisTickLabels, fontdict={'fontsize':7})
	ax1.set_yticks(axisRange)
	ax1.set_yticklabels(axisTickLabels, fontdict={'fontsize':7})

	ax1.set_xlabel(axesLabels[0], fontsize=12)
	ax1.set_ylabel(axesLabels[1], fontsize=12)
	ax1.set_title(title, fontsize=14)

	return heatmap

def plotHistogram(inputs, curveLabels=None, colors=None, axesLabels=None, xlim=None,
                  ylim=None, log=False, loglog=False, fontsize=14, linewidth=1.5,
                  expectedValue=None, normX=1.0, normY=1.0, mode=None, ax=None):
    '''Plots data from histogram files.

    Parameters
    ----------
    inputs : list
        List with format [filename, xcolumn, ycolumn, ynormcolumn (optional)]
	curveLabels : list
        Curve labels on legend.
	colors : list
        Curve colors.
    mode : None, 'sqrt'
        'sqrt' -> Plot the square root of the y data.
        (other modes can be added)

	:Common color schemes:
	[ "#0000ff","#33dd00","#ff0000","#ff8800"]
    [(.2,.2,1), (0,.5,.5), (.9,.7,0), (1,.2,.2)]   primary colors
	[(.9,0,0),(0,0,.9),(0,.7,0)]                   rbg
	'''
    data = []
    XN = []
    YN = []
    normYN = []

    for inputList in inputs:
        data.append(np.loadtxt(inputList[0], skiprows=1))
        XN.append(int(inputList[1]))
        YN.append(int(inputList[2]))
        if len(inputList) == 4:
            normYN.append(int(inputList[3]))
        else:
            normYN.append(-1)

    if ax is None:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
    else:
        ax1 = ax

    if axesLabels is not None:
		ax1.set_xlabel(axesLabels[0], fontsize=fontsize)
		ax1.set_ylabel(axesLabels[1], fontsize=fontsize)

	# Handle axes
    if log:
        ax1.set_yscale('log', basey=10)
    elif loglog:
		ax1.set_xscale('log', basex=10)
		ax1.set_yscale('log', basey=10)
    if xlim is not None:
		ax1.set_xlim(xlim)
    if ylim is not None:
		ax1.set_ylim(ylim)

	# Handle ticks
    ax1.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)

    for i in range(len(inputs)):
        # Normalize and implement mode
        if mode is None:
            xdata = data[i][:,XN[i]]/normX
            ydata = data[i][:,YN[i]]/normY
        elif mode == "sqrt":
            xdata = data[i][:,XN[i]]/normX
            ydata = np.sqrt(data[i][:,YN[i]])/normY

        # Normalization column
        if normYN[i] != -1:
            ydata /= data[i][:,normYN[i]]

        if curveLabels is not None:
            if colors is not None:
				ax1.plot(xdata, ydata, color=colors[i], label=curveLabels[i], lw=linewidth)
            else:
                ax1.plot(xdata, ydata, label=curveLabels[i], lw=linewidth)
        else:
            if colors is not None:
				ax1.plot(xdata, ydata, color=colors[i], lw=linewidth)
            else:
				ax1.plot(xdata, ydata, lw=linewidth)

	# Expected value dotted line
	if expectedValue is not None:
		windowXLim = ax1.get_xlim()
		windowXRange = np.arange(windowXLim[0]-.5, windowXLim[1]+.5, (windowXLim[1]-windowXLim[0])/100.0)
		ax1.plot(windowXRange, np.ones(len(windowXRange))*expectedValue, 'k--', lw=1)

	ax1.legend(frameon=False, loc="lower right")
	