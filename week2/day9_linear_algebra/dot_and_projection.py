import numpy as np

print("=== DOT PRODUCT & PROJECTION ===")

# Two vectors
v1 = np.array([3, 4])
v2 = np.array([4, 0])

print("v1:", v1)
print("v2:", v2)

print("\n=== DOT PRODUCT ===")

dot = np.dot(v1, v2)
print("Dot product:", dot)

# Dot product measures:
# 1) how aligned two vectors are
# 2) how much one vector goes in the direction of another


print("\n=== MAGNITUDES ===")

mag_v1 = np.linalg.norm(v1)
mag_v2 = np.linalg.norm(v2)

print("||v1||:", mag_v1)
print("||v2||:", mag_v2)


print("\n=== COSINE SIMILARITY ===")

cos_sim = dot / (mag_v1 * mag_v2)
print("Cosine similarity:", cos_sim)

# Range:
# 1   → same direction
# 0   → orthogonal (unrelated)
# -1  → opposite direction


print("\n=== VECTOR PROJECTION ===")

# Project v1 onto v2
projection = (dot / np.dot(v2, v2)) * v2
print("Projection of v1 onto v2:", projection)

# Projection = how much of v1 lies along v2


print("\n=== ORTHOGONAL VECTORS ===")

v3 = np.array([0, 5])
v4 = np.array([5, 0])

print("v3:", v3)
print("v4:", v4)
print("Dot product:", np.dot(v3, v4))
print("Cosine similarity:", np.dot(v3, v4) /
      (np.linalg.norm(v3) * np.linalg.norm(v4)))

# Orthogonal vectors → dot = 0 → no relationship
