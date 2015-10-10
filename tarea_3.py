# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#P1: Integracion con el metodo de Runge-Kutta de orden 3

mu = 1.203 #RUT: 18.636.203 - 9
si = 0 #s inicial
sf = 20*np.pi #s final (aprox. 10 periodos)
h = 0.1 #paso
y0 = 4.0 #condicion inicial
v0 = 0 #condicion inicial de dy/ds
n = (sf - si)/h
s = np.linspace(si, sf, n)
v = np.zeros(shape=(n)) #v = dy/ds

def F(mu, y, v):
    '''
    F: num num num -> float
    Calcula la funcion F(y,v)
    ejemplo: F(1.203, 0.1, 0.3) devuelve 0.25729...
    '''
    return -y - mu*(y**2 - 1)*v

def RK3(mu, h, n, v, y0):
    '''
    RK3: num num num array num -> array
    Integra la funcion F(y,v) con el metodo de Runge-Kutta de orden 3, y
    devuelve el valor de y.
    ejemplo: RK3(1.203, 0.1, 2, [0.1, 0.2], 0) devuelve [0.0, 0.0]
    '''
    i = 0
    y = np.zeros(shape=(n))
    y[0] = y0
    v[0] = v0
    while i < n-2:
        dv1 = h*F(mu, y[i], v[i])
        dy1 = h*v[i]
        dy2 = h*(v[i] + dv1/2)
        dv2 = h*F(mu, y[i] + dy1/2, v[i] + dv1/2)
        dy3 = h*(v[i] + dv2/2)
        dv3 = h*F(mu, y[i] + dy2/2, v[i] + dv2/2)
        dy = (dy1 + 4*dy2 + dy3)/6
        y[i+1] = y[i] + dy
        dv = (dv1 + 4*dv2 + dv3)/6
        v[i+1] = v[i] + dv
        i += 1
    return y
    
y = RK3(mu, h, n, v, y0)

#plt.plot(s, y, color='g')
#plt.plot(y, v, color='g')
#plt.title('Condiciones iniciales: dy/ds = 0    y = 4.0')
#plt.xlabel('y')
#plt.ylabel('dy/ds')
#plt.xlim([si, sf])
#plt.xlim([y[0], y[len(y)-1]])
#plt.savefig('grafico_dyds_2')


#P2: Resolver el sistema de Lorenz con el metodo de Runge-Kutta de orden 4


































    