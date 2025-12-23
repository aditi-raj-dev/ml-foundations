import numpy as np
import time

#1. normalize a vector manually
print("=== NORMALIZE A VECTOR ===")
v=np.array([3,4])
norm=np.sqrt(np.sum(v**2))
v_normalized=v/norm
print("original vector:",v)
print("normalized vector:",v_normalized)


#2. MIN-MAX SCALING
print("\n=== MIN-MAX SCALING ===")
data=np.array([2,4,6,8])
min_val=np.min(data)
max_val=np.max(data)
scaled=(data-min_val)/(max_val-min_val)
print("original data:", data)
print("scaled data:", scaled)

#3. COMPUTE MEAN OF EACH COLUMN OF 2D DATASET
print("\n=== COLUMN - WISE MEAN ===")
dataset=np.array([[1,2,3],[4,5,6],[7,8,9]])
col_mean=np.mean(dataset,axis=0)
print("DATASET:\n", dataset)
print("column-wise mean :", col_mean)

#4. REPLACE ALL NEGATIVE VALUES WITH 0
print("\n=== REPLACE NEGATIVE WITH 0 ===")
arr=np.array([-1,2,-3,4])
arr[arr<0]=0
print("modified array:",arr)

#5. COMPARE LOOP VS VECTORIZED TIMING
print("\n=== LOOP VS VECTORIZED TIMING ===")
size=1000000
vec=np.arange(size)

#using loop
start=time.time() #stores tsrting time before execution of large program
loop_sum=0
for x in vec:
  loop_sum+=x
end=time.time()# end time after execution of large data problem 
loop_time=end-start

#using numpy vectorized sum
start=time.time()
vec_sum=np.sum(vec)
end=time.time()
vec_time=end-start

print(f"loop time: {loop_time:.5f} seconds , Sum:{loop_sum}")
print(f"vectorized time: {vec_time:.5f} seconds , Sum:{vec_sum}")

# Vectorization avoids Python loops and runs faster using C-level operations
# ML models rely on vectorized math for speed on large datasets
