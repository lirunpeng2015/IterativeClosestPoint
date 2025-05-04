import numpy as np

x_ref = np.array([1,2,3])
y_ref = np.array([4,5,6])
x_src = np.array([7,8,9])
y_src = np.array([10,11,12])

ref_points = np.vstack((x_ref, y_ref)).T

R = np.eye(2)
t = np.zeros((2,1))
print(t)


