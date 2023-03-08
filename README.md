# fft-odd-number-of-input-
import matplotlib.pyplot as plt
import numpy as np
import scipy

def bluestein_fft(x):
    n = len(x)
    m = int(np.ceil(np.log2(2*n-1)))
    p = 2**m
    x_padded = np.zeros(p, dtype=complex)
    x_padded[:n] = x
    t = np.arange(-n+1, n)
    t_squared = np.square(t)
    a = np.exp(-1j * np.pi * t_squared / n)
    b = np.exp(1j * np.pi * t / n)
    y_padded = np.multiply(x_padded, b)
    y_padded = np.multiply(y_padded, a)
    z_padded = scipy.fft(y_padded)
    c = np.exp(-1j * np.pi * t_squared / n)
    c = np.pad(c, (0, p - (2*n-1)), mode='constant')
    w_padded = np.multiply(c, z_padded)
    w = np.fft.ifft(w_padded)[:n]
    return w

n = 301
if n % 2 == 0:
    f = np.sin(np.arange(0, 2*np.pi, 2*np.pi/n))
else:
    threshold = 1000
    if n > threshold and (n+1) % threshold == 0:
        f = np.append(np.sin(np.arange(0, 2*np.pi, 2*np.pi/n)), [0])
        n += 1
    else:
        f = np.sin(np.arange(0, 2*np.pi, 2*np.pi/n))

if n % 2 == 0:
    F = scipy.fft(f)
else:
    F = bluestein_fft(f)

plt.subplot(2, 1, 1)
plt.scatter(np.arange(0, 2*np.pi, 2*np.pi/n), f, marker='o')
plt.subplot(2, 1, 2)
plt.scatter(range(0,len(F)), F.real, marker='o', color='green')
plt.scatter(range(0,len(F)), F.imag, marker='o', color='purple')
plt.show()
