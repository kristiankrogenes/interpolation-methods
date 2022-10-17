import numpy as np
import matplotlib.pyplot as plt
import math, random
from dataset_generator import DatasetGenerator
from plotter import Plotter

def LeastSquaresMethod(x_data, y_data, p):
    A = np.array([np.array(x_data)**i for i in p]).T
    f = np.vstack(y_data)
    coeffs = np.linalg.inv(A.T @ A) @ A.T @ f
    return coeffs.flatten()

def interpolate(x_obs, y_obs, monomials):
    lsm_coeffs = LeastSquaresMethod(x_obs, y_obs, monomials)
    print("MONOMIALS: ", monomials)
    print("INTERPOLATED COEFFICIENTS: (Monomial order)\n", lsm_coeffs)
    x_dist = np.linspace(int(x_obs[0])-1, int(x_obs[-1])+1, (int(x_obs[-1])-int(x_obs[0])+2)*10)
    interpolated_y = np.sum(np.array([[a*dx**p for p, a in zip(monomials, lsm_coeffs)] for dx in x_dist]), axis=1)
    Plotter().plot_interpolation(x_obs, y_obs, [x_dist], [interpolated_y])


if __name__ == '__main__':

    # // SET PARAMETERS HERE // ===================
    monomials=[i for i in range(4)]
    n=50
    # interval = np.linspace(-10, 10, 101)
    # coeffs = DatasetGenerator().generate_random_polynomial(len(monomials)-1)
    # x_obs, y_obs = DatasetGenerator().get_random_points_from_polynomial_with_noise(coeffs, interval, n)
    x_obs, y_obs = [1, 2, 4, 8, 13, 22], [2, 5, 8, 22, 27, 2]
    # // ==========================================
    interpolate(x_obs, y_obs, monomials)



    # x_synth = [i for i in range(1, 101)]
    # x_obs = [1, 13, 22, 37, 55, 73, 79, 86, 95, 100]
    # a, b, c, d, e = 2, -1, 0.01, 5, 3
    # f_synth = np.array([a + b*xs + c*xs**2 + d*math.sin(2*math.pi/25*xs) + e*math.cos(2*math.pi/25*xs) for xs in x_synth])
    # f_obs = np.array([f_synth[xo-1] for xo in x_obs])
    # interpolate(x_obs, f_obs, x_synth, f_synth, polynomial_degree)

    

