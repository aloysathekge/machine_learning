# import numpy as np

# a=np.array([[2,3,5],[0,3,-3]])
# b=np.array([[1,2],[0,1],[2,4]])

# ab=np.dot(a,b)
# ab_flat=ab.flatten()
# print(ab)
# print(ab_flat)
# # print(a.shape)
# # print(a.ndim)
# # print(a.size)
# # print(a.itemsize)
# # print(a.dtype)

# Derivative 
from typing import Callable
import numpy as np
from numpy import ndarray

def deriv(func:Callable[ndarray,ndarray],input_:ndarray, delta:float=0.001)->ndarray:
    """
    Evaluates the derivative of a function func at every element in input_ array

    """
    return (func(input_ + delta) - func(input_-delta))/(2*delta)

def f(x):
   return x**2 + 3*x

input=np.array([2,6,8])

gradients=deriv(f,input)

print(gradients)
