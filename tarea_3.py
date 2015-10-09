# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#P1: Integracion con el metodo de Runge-Kutta de orden 3

mu = 1.203 #RUT: 18.636.203 - 9
si = 0 #s inicial
sf = 20*np.pi #s final (aprox. 10 periodos)
h = 0.1 #paso
y0 = 0.1 #condicion inicial
n = (sf - si)/h
s = np.linspace(si, sf, n)

def F(mu, s, y, v):
    return -y - mu*(y**2 - 1)*v

def RK3(mu, h, n, s, y0):
    i = 0
    v = np.zeros(shape=(n))
    y = np.zeros(shape=(n))
    y[0] = y0
    while i < n-2:
        dv1 = h*F(mu, s[i], y[i], v[i])
        dy1 = h*v[i]
        dy2 = h*(v[i] + dv1/2)
        dv2 = h*F(mu, s[i] + h/2, y[i] + dy1/2, v[i] + dv1/2)
        dy3 = h*(v[i] + dv2/2)
        #dv3 = h*F(mu, s[i] + h/2, y[i] + dy2/2, v[i] +dv2/2)
        dy = (dy1 + 4*dy2 + dy3)/6
        y[i+1] = y[i] + dy
        i += 1
    return y
    
y = RK3(mu, h, n, s, y0)
plt.plot(s, y)        
    