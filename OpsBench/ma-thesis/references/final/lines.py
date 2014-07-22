'''
Calculates effective working set size.
'''
import numpy as np

def calc(nbytes, stride, num_iters):
  '''
  Simulates the elements the GPU accesses; records how many lines are read from.
  '''
  line_size = 32 # Number of elements that fit per cache line
  lines = set()

  num_elems = nbytes / 4
  array = np.arange(stride, num_elems*stride+1, stride, dtype=np.int64)
  array = array % num_elems
  array = np.int32(array)

  k = 0
  for _ in xrange(num_iters):
    lines.add(k / line_size)
    k = array[k]
  print '%d\t%d' % (stride*4,len(lines))

if __name__ == '__main__':
  calc(200*2**10, 2**1-1, 1024)
  calc(200*2**10, 2**2-1, 1024)
  calc(200*2**10, 2**3-1, 1024)
  calc(200*2**10, 2**4-1, 1024)
  calc(200*2**10, 2**5-1, 1024)
  calc(200*2**10, 2**6-1, 1024)
  calc(200*2**10, 2**7-1, 1024)
  calc(200*2**10, 2**8-1, 1024)
  calc(200*2**10, 2**9-1, 1024)
  calc(200*2**10, 2**10-1, 1024)

