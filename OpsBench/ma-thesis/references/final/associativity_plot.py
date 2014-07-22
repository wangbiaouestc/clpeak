'''
Plots latency vs. array size to find the associativity of the cache.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
  
  if len(sys.argv) != 2:
    print 'Usage: python associativity_plot.py associativity.py_output_file_name'
    exit()
    
  results = np.loadtxt(sys.argv[1], dtype=np.int32)
  nbytes = results[:,0]
  stride = results[:,1] * 4
  cycles = results[:,4]
  niters = results[:,5]
  avg_time = np.float64(cycles) / niters
  fig = plt.figure()
      
  # Plot.
  plt.title('Latency vs. Array Size')
  plt.ylabel('Latency (clock cycles)')
  plt.xlabel('Array Size (bytes)')
  plt.plot(nbytes, avg_time)
  plt.show()

