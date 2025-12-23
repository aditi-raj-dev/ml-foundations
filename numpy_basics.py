import numpy as np 

print ("ARRAY FROM LIST")
a=np.array([1,2,3])
print(a,a.shape,a.dtype)

print("\n Zeroes , Ones , Full")
print(np.zeros(3))
print(np.ones(3))
print(np.full(3,5))

print("\n Arange and Linspace")
print(np.arange(0,10,2))
print(np.linspace(0,1,5))

print("\n Dtype control")
b= np.array([1,2,3],dtype=np.int32)
c= np.array([1,2,3], dtype=np.float64)
print(b,b.dtype)
print(c,c.dtype)

print("\n Reshape")
d=np.arange(6)
print(d.reshape(2,3))

print("\n Flatten vs Ravel")
print(d.flatten())
print(d.ravel())

print("\n Indexing and slicing")
e=np.array([[1,2],[3,4]])
print(e[0])
print(e[:,1])

print("\n Boolean masking")
f=np.array([-1,2,-3,4])
print(f[f>0])

