# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:43:17 2019

@author: Agnibha Banerjee
"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.colors as mcolors
from scipy.integrate import simps
from scipy.special import legendre as lg

# Setting the orbital parameters
r1 = 1  # Planet
r2 = 100  # Star
R = 200  # Semi-Major Axis
I0 = 1  # Intensity at centre of limb-darkening
u = 0.5  # Limb-Darkening coefficient
L = I0*np.pi*r2**2*(1-u/3)  # Apparent luminosity of star

# Finding the umbra and penumbra terminators
termin1 = np.arccos((r1+r2)/R)
termin2 = np.arccos((r1-r2)/R)

n = 100  # Number of grid points

'''
Function to normalize by numeric integration
Input:  yar : The y data
        xar : The x data
Output: Normalized yar
'''


def normalizer(yar, *xar):
    y = yar
    for x in xar:
        area = simps(yar, x)
        yar = area
    return (y/area)


'''
Function to find incident light in the day region of planet
Input:  x : The angle from the substellar point
Output: The incident light on the given location
'''


def day_refl(x):
    leg = 0
    for i in range(1, 5):
        leg = leg + i*(r1/R)**(i-1)*lg(i)(x)
    return (L/(np.pi*R**2))*leg


'''
Function to find incident light in the twilight region of planet
Input:  x : The angle from the substellar point
Output: The incident light on the given location
'''


def twi_refl(x):
    r_ratio_pow = np.array([(r1/r2)**i for i in range(5)])
    new_var = np.array([(R*x/r2)**i for i in range(5)])

    c_u0 = np.sum(np.array([2/3, -np.pi/2, 1, 0, -1/12])*r_ratio_pow)/np.pi
    c_u1 = np.sum(np.array([np.pi/2, -2, 1/3, 0, 0])*r_ratio_pow)/np.pi
    c_u2 = np.sum(np.array([1, 0, -1/2, 0, 0])*r_ratio_pow)/np.pi
    c_u3 = np.sum(np.array([0, 1, 0, 0, 0])*r_ratio_pow)/np.pi
    c_u4 = np.sum(np.array([-1/12, 0, 0, 0, 0])*r_ratio_pow)/np.pi

    c_u = np.array([c_u0, c_u1, c_u2, c_u3, c_u4])

    c_d0 = np.sum(np.array([3/16, -1/2, 3/8, 0, -1/16])*r_ratio_pow)
    c_d1 = np.sum(np.array([1/2, -3/4, 0, 1/4, 0])*r_ratio_pow)
    c_d2 = np.sum(np.array([3/8, 0, -3/8, 0, 0])*r_ratio_pow)
    c_d3 = np.sum(np.array([0, 1/4, 0, 0, 0])*r_ratio_pow)
    c_d4 = np.sum(np.array([-1/16, 0, 0, 0, 0])*r_ratio_pow)

    c_d = np.array([c_d0, c_d1, c_d2, c_d3, c_d4])

    refl_u = (r2*L/(np.pi*R**3))*np.sum(c_u*new_var)

    refl_d = (r2*L/(np.pi*R**3))*np.sum(c_d*new_var)

    return (3*(1-u)/(3-u))*refl_u + (2*u/(3-u))*refl_d


'''
Function to find the incident light as a function of gamma
where gamma ranges from 0 to 2*pi
'''


def kopal_1d():
    gamma = np.linspace(0, np.pi, n//2)
    mu = np.cos(gamma)
    refl = np.zeros(n//2)

    for i in range(n//2):
        if gamma[i] < termin1:
            refl[i] = day_refl(mu[i])

        elif gamma[i] < termin2:
            refl[i] = twi_refl(mu[i])

        else:
            refl[i] = 0

    refl = np.concatenate((refl, np.flip(refl)))
    return refl


'''
Function to find the incident light on the planet as a function
of the latitude and longitude
'''


def kopal_2d():
    t = np.linspace(0, np.pi, n)
    p = np.linspace(0, 2*np.pi, n)
    theta, phi = np.meshgrid(t, p)
    mu = np.sin(phi)*np.cos(theta)
    gamma = np.arccos(mu)
    refl = np.zeros_like(gamma)

    for i in range(n):
        for j in range(n):
            if gamma[i, j] < termin1 or gamma[i, j] > 2*np.pi-termin1:
                refl[i, j] = day_refl(mu[i, j])

            elif gamma[i, j] < termin2 or gamma[i, j] > 2*np.pi-termin2:
                refl[i, j] = twi_refl(mu[i, j])

    return refl


'''
Function to plot the obtained Intensity Map as a function of gamma
'''


def plotter_1d():
    xdat = np.linspace(0, 2*np.pi, n)
    ydat = normalizer(kopal_1d(), xdat)

    plt.plot(xdat,
             normalizer(np.maximum(np.zeros(n), np.cos(xdat)), xdat),
             alpha=0.75, color='g', linewidth=2, label='cosine')

    plt.plot(xdat, ydat, label='Kopal', color='black', alpha=0.75)

    plt.xlabel(r'$\gamma_s$ (in radians)')
    plt.ylabel('Illumination')
    plt.axvline(termin1)
    plt.axvline(termin2)
    plt.axvline(2*np.pi-termin1)
    plt.axvline(2*np.pi-termin2)
    plt.legend()
    plt.savefig('compare.png')


'''
Function to plot the obtained Intensity Map as a function of the latitude
and longitude on the planet.
'''


def plotter_2d():
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')

    t = np.linspace(0, np.pi, n)
    p = np.linspace(0, 2*np.pi, n)
    tt, pp = np.meshgrid(t, p)

    fcolors = normalizer(kopal_2d(), t, p)
    ax.plot_surface(tt, pp, fcolors, cmap=cm.coolwarm)
    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Illumination')
    plt.show()


def main():
    plotter_1d()
    plotter_2d()


if __name__ == '__main__':
    main()
