import numpy as np

#DOT PRODUCT
print("===DOT PRODUCT===")
v1=np.array([1,2,3])
v2=np.array([4,5,6])
dot=np.dot(v1,v2)
print("v1:",v1)
print("v2:",v2)
print("dot product:",dot)
#used in ML:weight*input multiplication in linear models 

#vector magnitude
print("\n===VECTOR MAGNITUDE ===")
magnitude=np.sqrt(np.sum(v1**2))
print("magnitude of v1:", magnitude)
#used in ML : normalization

#COSINE SIMILARITY 
print("\n===COSINE SIMILARITY ===")
cos_sim=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
print("COSINE SIMILARITY:",cos_sim)
#used in ML : similarity between embeddings, NLP

#EUCLIDEAN DISTANCE
print("\n===EUCLIDEAN DISTANCE===")
distance=np.sqrt(np.sum((v1-v2)**2))
print("DISTANCE BETWEEN V1 AND V2:", distance)
#used in ML : k-nn , clustering

