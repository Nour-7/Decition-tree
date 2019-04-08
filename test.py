import pylab
import numpy as np

a = [[1, 2, 3],
     [4, 3, 1],
     [1, 2, 3]]
unique_elements, counts_elements = np.unique(a, axis=0, return_counts=True)
print(unique_elements)
print(counts_elements)
