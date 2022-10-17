import numpy as np
import math, random
import matplotlib.pyplot as plt
from dataset_generator import DatasetGenerator
from plotter import Plotter

def hypotenuse(vec):
    return math.sqrt(np.sum([x**2 for x in vec]))
 
def proj(v1, v2):
    c1 = np.dot(v1, v2) / np.dot(v1, v1)
    c2 = v1
    return c1, c1*c2

def GramSchmidt(A):
    A = A.T
    A_GS, c = [], []
    for i in range(len(A)):
        ui = []
        ci = np.zeros(len(A))
        for j in range(i+1):
            if j==0:
                ui.append(A[i])
                ci[i]=1
            else:
                c1, prj = proj(A_GS[j-1], A[i])
                ui.append(-1 * prj)
                ci[j-1]=c1
        A_GS.append(np.sum(ui, axis=0))
        c.append(ci)
    # print("U: \n", np.array(u).T)
    # e = np.array([ui/hypotenuse(ui) for ui in u])
    # print("U-Normalized: \n", e.T)
    return np.array(A_GS).T, np.array(c).T

def LSM_GramSchmidt(x_data, y_data, p):
    A = np.array([np.array(x_data)**i for i in range(p+1)]).T
    b =  y_data[:, np.newaxis]
    P, S = GramSchmidt(A)
    D = np.vstack([np.dot(pi, y_data)/np.dot(pi, pi) for pi in P.T])

    c = np.dot(np.linalg.inv(S), D).flatten()

    print("A: \n", A, "\n")
    print("b: \n", b, "\n")
    print("P: \n", P, "\n")
    print("S: \n", S, "\n")
    print("D: \n", D, "\n")
    print("Coeffs: \n", c, "\n")

    return c

def interpolate(x_obs, y_obs, poly_deg):
    interpolation_coeffs = LSM_GramSchmidt(x_obs, y_obs, poly_deg)
    x_dist = np.linspace(int(x_obs[0])-1, int(x_obs[-1])+1, (int(x_obs[-1])-int(x_obs[0])+2)*10)
    interpolated_y = np.sum(np.array([[a*dx**index for index, a in enumerate(interpolation_coeffs)] for dx in x_dist]), axis=1)
    Plotter().plot_interpolation(x_obs, y_obs, [x_dist], [interpolated_y])

if __name__ == '__main__':

    # // SET PARAMETERS HERE // 
    poly_deg = 2
    # x_dist = np.linspace(-1, 4, 100)
    # original_coeffs = DatasetGenerator().generate_random_polynomial(2)
    # x_obs, y_obs = DatasetGenerator().get_random_points_from_polynomial_with_noise(original_coeffs, np.linspace(-10, 10, 101), 100)
    x_obs, y_obs = np.array([0, 1, 2, 3, 4]), np.array([0, 1, 4, 7, 0])
    # A = np.array([[1, 1, 1, 1], [0, 2, 2, 0], [1, 0, -1, 0]]).T
    # // ========================

    interpolate(x_obs, y_obs, poly_deg)

    # A_GS, S = GramSchmidt(A)
    # print("S: \n", S)
    # print("A(GS): \n", A_GS)
    # print("A: \n", np.dot(A_GS, S))
    # res = np.dot(A_GS, A_GS.T)
    # print("RESULT: \n", res)



