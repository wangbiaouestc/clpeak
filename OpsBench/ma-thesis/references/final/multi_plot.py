import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Usage: python multi_stride.py multi_stride.py_output_file_name'
    exit()
    
  results = np.loadtxt(sys.argv[1], dtype=np.int32)
  nbytes = results[:,0]
  stride = results[:,1] * 4
  threads = results[:,3]
  cycles = results[:,4]
  niters = results[:,5]
  avg_time = np.float64(cycles) / niters
  fig = plt.figure()

  current_start = 0
  current_bpg  = threads[0]
  for i in xrange(len(threads)):
    if threads[i] != current_bpg or i == len(threads) - 1:
      # Plot.
      plt.plot(
          stride[current_start: i],
          avg_time[current_start: i],
          label=str(current_bpg) + ' threads')
      
      current_start = i
      current_bpg  = threads[i]

  plt.title('Latency vs. Stride Length')
  plt.ylabel('Latency (clock cycles)')
  plt.xlabel('Stride length (bytes)')
  plt.xscale('log', basex=2)
  plt.legend()
  plt.show()

