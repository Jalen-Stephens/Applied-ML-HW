import numpy as np
import matplotlib.pyplot as plt

r_vals = np.linspace(0.01, 0.99, 200)

d_eucl = 2 * r_vals  # ||u - v|| = 2r

# YOUR CODE HERE — compute d_hyp using the Poincaré formula
# Hint: arccosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
# What is ||u-v||^2 when u=(r,0) and v=(-r,0)?
# What is (1-||u||^2)(1-||v||^2)?

# Using Poincaré disk hyperbolic distance (MML-book § Hyperbolic Geometry)
d_hyp = np.arccosh(1 + 8*r_vals**2 / (1 - r_vals**2)**2)

plt.figure(figsize=(6, 4))
plt.plot(r_vals, d_eucl, label='Euclidean', color='#3d6b8e')
plt.plot(r_vals, d_hyp, label='Hyperbolic', color='#c45d3e')
plt.xlabel('r')
plt.ylabel('Distance')
plt.legend()
plt.title('Euclidean vs. Hyperbolic Distance')
plt.tight_layout()
plt.show()
