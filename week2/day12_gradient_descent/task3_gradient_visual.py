import numpy as np

# Data
X = np.array([1,2,3,4,5])
Y = np.array([3,5,7,9,11])   # y = 2x + 1

m = 0
c = 0
lr = 0.01
epochs = 20

n = len(X)

for epoch in range(epochs):
    Y_pred = m*X + c
    
    loss = np.mean((Y - Y_pred)**2)

    dm = (-2/n) * np.sum(X * (Y - Y_pred))
    dc = (-2/n) * np.sum(Y - Y_pred)

    m = m - lr * dm
    c = c - lr * dc

    print(f"Epoch {epoch+1} | m={round(m,4)} | c={round(c,4)} | loss={round(loss,4)}")
