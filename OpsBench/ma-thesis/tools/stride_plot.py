###############################################################################
### Plotting script taken and modified from C2070 microbenchmarks #############
###############################################################################

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import sys

def main(argv):
	results = np.loadtxt(argv[1], dtype=np.int32, delimiter=',')
	nbytes = results[:,0]
	stride = results[:,1]
	cycles = results[:,4]
	niters = results[:,5]
	avg_time = np.float64(cycles) / niters
	
	# set color map
	curves = set()
	for i in xrange(len(nbytes)):
		curves.add(nbytes[i])	
	num_colors = len(curves)
	cm = plt.get_cmap('gist_rainbow')
	cNorm = colors.Normalize(vmin=0, vmax=num_colors-1)
	scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
	cycle = [scalarMap.to_rgba(i) for i in xrange(num_colors)]
	plt.rc(('axes'), color_cycle = cycle)
	
	# configure the plot
	fig, ax = plt.subplots()
	fig.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	
	# draw the data
	current_start = 0
	current_size  = nbytes[0]
	for i in xrange(len(nbytes)):
		if not nbytes[i] == current_size:
			ax.plot(stride[current_start: i], avg_time[current_start: i], label=str(current_size) + ' B')      
			current_start = i
			current_size  = nbytes[i]
	ax.plot(stride[current_start:], avg_time[current_start:], label=str(current_size) + ' B')

	# draw the labels
	ax.set_title('Latency vs. Stride Length')
	ax.set_ylabel('Latency (clock cycles)')
	ax.set_xlabel('Stride length (bytes)')
	ax.set_xscale('log', basex=2)
	ax.legend(loc='best')
	plt.show()
	
if __name__ == '__main__':
	if not len(sys.argv) == 2:
		print 'Usage: python stride_plot.py stride.py_output_file_name'
		sys.exit(0)
	main(sys.argv)

