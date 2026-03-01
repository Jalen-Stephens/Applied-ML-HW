import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 14, 200)
f = x ** 2

pts = np.array([2, 4, 12])
avg_x = pts.mean()

# Jensen quantities
f_at_avg = avg_x ** 2
Ef = np.mean(pts ** 2)

plt.figure(figsize=(7, 5))
plt.plot(x, f, 'k-', linewidth=1.5, label=r'$f(x)=x^2$')

# Plot the three original points
plt.scatter(pts, pts**2, s=70, label='(2,4), (4,16), (12,144)')

# Plot f(E[X]) and E[f(X)]
plt.scatter([avg_x], [f_at_avg], s=100, label=r'$f(\mathbb{E}[X])$')
plt.scatter([avg_x], [Ef], s=100, label=r'$\mathbb{E}[f(X)]$')

# Draw chord between outer points
x_chord = np.array([pts.min(), pts.max()])
y_chord = x_chord ** 2
plt.plot(x_chord, y_chord, linestyle='--', linewidth=2, label='Chord (2 to 12)')

plt.xlim(0, 14)
plt.ylim(0, 200)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Which one is bigger?
comparison_text = f"E[f(X)] = {Ef:.2f} > f(E[X]) = {f_at_avg:.2f}"
plt.title("Jensen's Inequality - Geometric View\n" + comparison_text)

plt.tight_layout()
plt.show()

print("avg_x =", avg_x)
print("f(E[X]) =", f_at_avg)
print("E[f(X)] =", Ef)
print("Conclusion: E[f(X)] is bigger than f(E[X])")