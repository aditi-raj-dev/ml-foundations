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

#Probability stabilizes ML because models assume data follows a distribution.

#When we understand mean/varieance , we understand what "normal " data looks like.

#This prevents randomness from misleading the model.

#Randomness matters because real-world data is never perfect or deterministic.

#Noise , measurement errors, environment changes = randomness.

#ML learns patterns + handles randomness rather than breaking.

