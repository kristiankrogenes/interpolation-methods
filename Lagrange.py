import numpy as np
import math, random
import matplotlib.pyplot as plt
from dataset_generator import DatasetGenerator
from plotter import Plotter

def Lagrange(x, y, interval):
    P = []
    for i in range(len(x)):
        p_top, p_bottom = 1, 1
        for j in range(len(x)):
            if i != j:
                p_top *= interval-x[j]
                p_bottom *= x[i]-x[j]
        P.append(p_top/p_bottom)
    L = 0
    for i in range(len(y)):
        L += y[i]*P[i]
    return L

def LagrangeCoeffs(x, y):
    P = []
    for i in range(len(x)):
        pi = []
        for j in range(len(x)):
            if i != j:
                pi.append(np.poly1d([1/(x[i]-x[j]), -x[j]/(x[i]-x[j])]))
        p = 1
        for n in range(len(pi)):
            p *= pi[n]
        P.append(np.flip(np.array(p)))
    print("P: [1, x, x^2, ..., x^n]\n", np.array(P), "\n")
    L = [y[i]*P[i] for i in range(len(P))]
    print("L: [1, x, x^2, ..., x^n]\n", np.array(L), "\n")
    return np.sum(L, axis=0)

def interpolate(x_obs, y_obs):
    lagrange_coeffs = LagrangeCoeffs(x_obs, y_obs)
    print("LAGRANGE COEFFS: [1, x, x^2, ..., x^n]\n", lagrange_coeffs, "\n")
    x_dist = np.linspace(int(x_obs[0])-1, int(x_obs[-1])+1, (int(x_obs[-1])-int(x_obs[0])+2)*10)
    l_poly = np.sum(np.array([[lagrange_coeffs[index]*xi**index for index in range(len(lagrange_coeffs))] for xi in x_dist]), axis=1)
    Plotter().plot_interpolation(x_obs, y_obs, [x_dist], [l_poly])


if __name__ == '__main__':

    # // SET GENERATED PARAMETERS HERE // ===================
    # poly_deg = 2
    # interval = np.linspace(-10, 10, 101)
    # coeffs = DatasetGenerator().generate_random_polynomial(poly_deg)
    # x_obs, y_obs = DatasetGenerator().get_random_points_from_polynomial(coeffs, interval, len(coeffs))
    # // ==========================================

    x_obs = np.array([-2, 0, 2])
    y_obs = np.array([0, 4/6, 0])

    interpolate(x_obs, y_obs)
