import numpy as np

print("=== UNIFORM DISTRIBUTION ===")
uniform = np.random.uniform(0, 10, 1000)
print("Mean:", np.mean(uniform))
print("Std Dev:", np.std(uniform))

print("\n=== NORMAL DISTRIBUTION ===")
normal = np.random.normal(0, 1, 1000)
print("Mean:", np.mean(normal))
print("Std Dev:", np.std(normal))

print("""
Why real-world looks Gaussian?
Because many small random effects combine together.
Height, marks, IQ, noise, errors → all formed from many tiny influences.
Central Limit Theorem makes this happen naturally.
""")
