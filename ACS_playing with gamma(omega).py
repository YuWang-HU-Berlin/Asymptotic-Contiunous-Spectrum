#!/usr/bin/env python3
"""
Created on Wed Jun  7 22:24:36 2023
playing with phi(omega)
@author: serhiyya  yuwang
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.font_manager import FontProperties

fnt = FontProperties(family='sans-serif',
                     size='large',
                     style='normal',
                     weight='normal',
                     stretch='normal')
n = 2
A = sp.Matrix([[0, 1],
               [-4,-1]])
B = sp.Matrix([[0, 0],
               [-1.5, 3]])
Id = sp.matrices.eye(n)

tau =20
h = 1e-9

lam, Y = sp.symbols('lam Y')
charmat = -Id * lam + A + B * Y
res = charmat.det()
quasipoly_sym = res.subs(Y, sp.exp(-lam * tau))
quasipoly = sp.lambdify(lam, quasipoly_sym)
quasipoly_der = lambda z: (quasipoly(z + h) - quasipoly(z)) / h
charpoly = sp.poly(res)
omegas = np.linspace(-10, 10, 1000)
gammas = np.zeros((omegas.shape[0], n))
phis = np.zeros(gammas.shape)
for ind in range(omegas.size):
    omega = omegas[ind]
    subs_charpoly = charpoly.subs(lam, 1j * omega)
    coefficients = np.array(subs_charpoly.coeffs())
    c = np.fromiter(coefficients, dtype=complex)
    Yroots = np.roots(c)
    gamma = -np.log(np.absolute(Yroots))
    phi = -np.angle(Yroots)
    gammas[ind, :] = gamma
    phis[ind, :] = phi


def mynewton(z0, tolerance=1e-11, max_iterations=200):
    z = z0
    iteration = 0

    while abs(quasipoly(z)) > tolerance and iteration < max_iterations:
        z -= quasipoly(z) / quasipoly_der(z)
        iteration += 1

    if abs(quasipoly(z)) <= tolerance:
        return z
    else:
        return None


zz = np.zeros(gammas.shape, dtype=complex)
for ind in range(omegas.size):
    omega = omegas[ind]
    zz[ind, 0] = mynewton(omega * 1j + gammas[ind, 0] / tau)
    zz[ind, 1] = mynewton(omega * 1j + gammas[ind, 1] / tau)

plt.figure(1)
num_columns = gammas.shape[1]
for col in range(num_columns):
    plt.plot(gammas[:, col], omegas,
             color='black',
             linestyle='--',
             linewidth=0.8,
             label=f'Column {col + 1}')

    real_parts = np.real(zz[:, col])
    imag_parts = np.imag(zz[:, col])

    # Plot points with real part > 0 in green
    plt.scatter(tau * real_parts[tau * real_parts > 0.001], imag_parts[tau * real_parts > 0.001],
                marker='o',
                s=40,
                c='red',
                label='Real > 0')

    # Plot points with real part = 0 in red
    plt.scatter(tau * real_parts[tau * real_parts < 0.001], imag_parts[tau * real_parts < 0.001],
                marker='o',
                s=40, c='black',
                label='Real = 0')

    # Plot points with real part < 0 in blue
    plt.scatter(tau * real_parts[tau * real_parts < -0.005], imag_parts[tau * real_parts < -0.005],
                marker='o',
                s=40,
                c='green',
                label='Real < 0')

    plt.axhline(0,
                color='blue',
                linestyle='dotted',
                linewidth=0.8,
                zorder=0)

    plt.axvline(0,
                color='blue',
                linestyle='dotted',
                linewidth=0.8,
                zorder=0)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.xlabel(r'$\tau\Re(\lambda)$',
           family=fnt.get_family(),
           fontdict={'family': 'Times New Roman', 'size': 25})
plt.ylabel(r'$\Im(\lambda)$',
           family=fnt.get_family(),
           fontdict={'family': 'Times New Roman', 'size': 25})

# Requires adjustment
plt.xlim(-1.2,1.2)
plt.ylim(-10.2,10.2)
plt.xticks([-1, -0.5, 0, 0.5, 1])
plt.yticks([-10, -5, 0, 5, 10])


plt.savefig('ActivePar_Fig4_2_.png', dpi=300, bbox_inches='tight')
# plt.legend()
plt.show()

