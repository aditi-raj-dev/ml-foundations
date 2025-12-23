import numpy as np

print ("===VECTOR BASICS===")

# 1D vector (row vector )
v=np.array([3,4])
print("vector v:",v)
print("shape of v:", v.shape)

#In ML , a vector represents one data point(features of one sample)

print ("\n===COLUMN VECTOR===")

#Convert to column vector 
v_col=v.reshape(-1,1)
print("column vector:\n", v_col)
print("shape of column vector:", v_col.shape)

#Column vectors are required for matrix multiplication

print ("\n===VECTOR MAGNITUDE===")

#Magnitude (length of vector)
magnitude=np.linalg.norm(v)
print("Magnitude of v:", magnitude)

#Magnitude tells how large the vector is

print ("\n===UNIT VECTOR===")
unit_v=v/magnitude
print("unit vector:", unit_v)
print("Magnitude of unit vector:", np.linalg.norm(unit_v))

#Normalization keeps direction but removes scale

print ("\n===ANGLE INTUTION===")

# IF TWO VECTORS POINT IN THE SAME DIRECTION , ANGLE IS SMALL
#IF THEY ARE PERPENDICULAR, ANGLE IS 90 DEGREES
#ML USES ANGLES TO MEASURE SIMILARITY BETWEEN DATA POINTS
