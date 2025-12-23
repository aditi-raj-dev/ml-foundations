import numpy as np 

#element - wise operations
print("===ELEMENT-WISE OPERATIONS ===")
a=np.array([1,2,3])
b=np.array([4,5,6])
print("a:",a,"b:",b)

print("addition:", a+b)
print("subtraction:", a-b)
print("multiplication:", a*b)
print("division:", a/b)

#broadcasting example(2D+1D)
print("===BROADCASTING EXAMPLE ===")
A=np.array([[1,2,3],[4,5,6]])
B=np.array([1,0,-1])
print("A: \n",A)
print("B: \n",B)
print("A+B: \n",A+B)

# AGGREGATION
print("===AGGREGATION ===")
print("sum axis=0:", np.sum(A, axis=0))
print("sum axis=1:", np.sum(A, axis=1))
print("Mean:", np.mean(A))
print("STD DEV:", np.std(A))

#PYTHON SUM VS NP SUM
py_sum=sum([1,2,3])
np_sum=np.sum(np.array([1,2,3]))
print("\n Python sum:", py_sum, "NumPy sum:", np_sum)



