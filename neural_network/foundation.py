import numpy as np

a=np.array([[2,3,5],[0,3,-3]])
b=np.array([[1,2],[0,1],[2,4]])

ab=np.dot(a,b)
ab_flat=ab.flatten()
print(ab)
print(ab_flat)
# print(a.shape)
# print(a.ndim)
# print(a.size)
# print(a.itemsize)
# print(a.dtype)



