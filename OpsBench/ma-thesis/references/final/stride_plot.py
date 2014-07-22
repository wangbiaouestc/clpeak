import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Usage: python stride_plot.py stride.py_output_file_name'
    exit()
    
  results = np.loadtxt(sys.argv[1], dtype=np.int32)
  nbytes = results[:,0]
  stride = results[:,1] * 4
  cycles = results[:,4]
  niters = results[:,5]
  avg_time = np.float64(cycles) / niters
  fig = plt.figure()

  current_start = 0
  current_size  = nbytes[0]
  for i in xrange(len(nbytes)):
    if nbytes[i] != current_size:
      # Plot.
      plt.plot(
          stride[current_start: i],
          avg_time[current_start: i],
          label=str(current_size) + ' B')
      
      current_start = i
      current_size  = nbytes[i]

  # Plot the last line.
  plt.plot(
    stride[current_start:],
    avg_time[current_start:],
    label=str(current_size) + ' B')

  plt.title('Latency vs. Stride Length')
  plt.ylabel('Latency (clock cycles)')
  plt.xlabel('Stride length (bytes)')
  plt.xscale('log', basex=2)
  plt.legend(loc='best')
  plt.show()

