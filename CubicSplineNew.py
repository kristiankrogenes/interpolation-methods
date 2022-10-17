import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import DatasetGenerator
from plotter import Plotter

def get_spline_index(X, x):
    for i in range(len(X)-1):
        if x >= X[i] and x <= X[i+1]:
            return i
    return None

def f(coeffs, x):
    return  np.sum([c*x**i for i, c in enumerate(coeffs)])

def CubicSpline(x_obs, y_obs, N0, Nn, choice):
    n = len(x_obs)
    f = np.array(y_obs)
    h = np.array([x_obs[i+1]-x_obs[i] for i in range(n-1)])

    if choiceA:
        abc = np.array([[h[i+1], 2*(h[i]+h[i+1]), h[i]] for i in range(n-2)])
        d = np.array([3*(f[i+1]-f[i])*h[i-1]/h[i]+3*(f[i]-f[i-1])*h[i]/h[i-1] for i in range(1, n-1)])
    else:
        abc = np.array([[h[i], 2*(h[i]+h[i+1]), h[i+1]] for i in range(n-2)])
        d = np.array([6*(f[i+1]-f[i])/h[i]-6*(f[i]-f[i-1])/h[i-1] for i in range(1, n-1)])

    d[0] = d[0]-abc[0][0]*N0
    d[-1] = d[-1]-abc[-1][2]*Nn

    ABC = np.zeros((n-2, n-2))

    if n==3:
        ABC[0][0] = abc[0][1]
    else:
        for i in range(n-2):
            if i==0:
                ABC[i][0] = abc[0][1]
                ABC[i][1] = abc[0][2]
            elif i==n-3:
                ABC[i][i-1] = abc[i][0]
                ABC[i][i] = abc[i][1]
            else:
                ABC[i][i-1] = abc[i][0]
                ABC[i][i] = abc[i][1]
                ABC[i][i+1] = abc[i][2]

    if choiceA:
        N = list(np.dot(np.linalg.inv(ABC), d[:, np.newaxis]).flatten())
        N = np.array([N0] + N + [Nn])
        print("N with boundaries: \n", N)
    else:
        M = list(np.dot(np.linalg.inv(ABC), d[:, np.newaxis]).flatten())
        M = np.array([N0] + M + [Nn])
        print("M with boundaries: \n", M)

    if choiceA:
        S = [[f[i-1], N[i-1], 3*(f[i]-f[i-1])/h[i-1]**2-(2*N[i-1]+N[i])/h[i-1], -2*(f[i]-f[i-1])/h[i-1]**3+(N[i-1]+N[i])/h[i-1]**2] for i in range(1, n)]
    else:
        S = [[f[i-1], (f[i]-f[i-1])/h[i-1]-(h[i-1]*M[i-1])/3-(h[i-1]*M[i])/6, M[i-1]/2, (M[i]-M[i-1])/(6*h[i-1])] for i in range(1, n)]
    
    print("h: \n", h)
    print("abc: \n", abc)
    print("Updated d: \n", d)    
    print("ABC: \n", ABC)    
    print("S-COEFFICIENTS: \n", np.array(S))

    coeffs = [[S[i][0]-S[i][1]*x_obs[i]+S[i][2]*x_obs[i]**2-S[i][3]*x_obs[i]**3, S[i][1]-2*S[i][2]*x_obs[i]+3*S[i][3]*x_obs[i]**2, S[i][2]-3*S[i][3]*x_obs[i], S[i][3]] for i in range(len(x_obs)-1)]

    return np.array(coeffs)


if __name__ == '__main__':

    choiceA = False
    # x_obs, y_obs = DatasetGenerator().generate_random_xy_values(-10, 10, -10, 10, 100, 8)
    x_obs, y_obs = [-1, 0, 2, 5, 9], [1, 2, -1, 0, 6]
    boundary_conditions = [0, 0]

    coeffs = CubicSpline(x_obs, y_obs, boundary_conditions[0], boundary_conditions[1], choiceA)

    print("\nCUBIC SPLINE OPTION", "A" if choiceA else "B", "\n==================")
    print("COEFFICIENTS: \n", coeffs)


    # PLOT CUBIC SPLINE RESULT 
    x_dist, s1= [], []

    for i in range(len(x_obs)-1):
        dx = np.linspace(x_obs[i], x_obs[i+1], 100)
        y1 = np.sum([[coeffs[i][index]*xi**index for index in range(4)] for xi in dx], axis=1)
        x_dist.append(dx)
        s1.append(y1)
    
    # x_value_to_check = 3.5
    # si = get_spline_index(x_obs, x_value_to_check)
    # if si != None:
    #     S_x = f(coeffs[si], x_value_to_check)
    #     print("S" + str(si+1) + "(" + str(x_value_to_check) + ") =", S_x)
    # else:
    #     print("X value outside of interval")
    
    Plotter().plot_interpolation(x_obs, y_obs, x_dist, s1)



