import numpy as np
import matplotlib.pyplot as plt

from dataset_generator import DatasetGenerator

def CubicSpline(x_obs, y_obs, derivative_condition, boundary_conditions):
    if len(x_obs) != len(y_obs) or derivative_condition not in [1, 2] or len(boundary_conditions) != 2:
        print("Input parameters are wrong.")
        return None

    n=len(x_obs)
    unknown = 4*(n-1)
    equations = 2*(n-1) + 2*(n-2) + 2
    funcs = n-1

    X = [[xi**3, xi**2, xi, 1] for xi in x_obs]
    dX1 = [[3*xi**2, 2*xi, 1, 0] for xi in x_obs]
    dX2 = [[6*xi, 2, 0, 0] for xi in x_obs]

    b1 = []
    for i, y in enumerate(y_obs):
        if i==0:
            b1.append(y)
        elif i==len(y_obs)-1:
            b1.append(y)
        else:
            b1.append(y)
            b1.append(y)
    b = np.array(b1 + [0] * (2*(n-2)) + boundary_conditions)

    A1 = np.array([[np.zeros(4) for _ in range(n-1)] for _ in range(unknown)])

    for i, e in enumerate(range(0, 2*(n-1), 2)):
        A1[e][i] = X[i]
        A1[e+1][i] = X[i+1]
    
    for i, e in enumerate(range(2*(n-1), 2*(n-1)+2*(n-2), 2)):
        A1[e][i] = dX1[i+1]
        A1[e][i+1] = -1*np.array(dX1[i+1])
        A1[e+1][i] = dX2[i+1]
        A1[e+1][i+1] = -1*np.array(dX2[i+1])

    if derivative_condition == 1:
        A1[-2][0] = dX1[0]
        A1[-1][-1] = dX1[-1]
    elif derivative_condition == 2:
        A1[-2][0] = dX2[0]
        A1[-1][-1] = dX2[-1]

    A = []
    for e in A1:
        ei = []
        for s in e:
            ei += list(s)
        A.append(ei)
    print("b:\n", b, "\n")
    print("A:\n", np.array(A))
    coeffs = np.dot(np.linalg.inv(np.array(A)), b[:, np.newaxis])
    coeffs = np.array_split(coeffs.flatten(), len(coeffs)/4)
    return np.array([np.flip(c) for c in coeffs])

def get_spline_index(X, x):
    for i in range(len(X)-1):
        if x >= X[i] and x <= X[i+1]:
            return i
    return None

def f(coeffs, x):
    return  np.sum([c*x**i for i, c in enumerate(coeffs)])

def CubicSplineA(x_obs, y_obs, N0, Nn):
    n = len(x_obs)
    f = np.array(y_obs)
    h = np.array([x_obs[i+1]-x_obs[i] for i in range(n-1)])
    print("h: \n", h)
    abc = np.array([[h[i+1], 2*(h[i]+h[i+1]), h[i]] for i in range(n-2)])
    print("abc: \n", abc)
    d = np.array([3*(f[i+1]-f[i])*h[i-1]/h[i]+3*(f[i]-f[i-1])*h[i]/h[i-1] for i in range(1, n-1)])
    print("d: \n", d)
    d[0] = d[0]-abc[0][0]*N0
    d[-1] = d[-1]-abc[-1][2]*Nn
    print("Updated d: \n", d)
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
    print("ABC: \n", ABC)
    N = list(np.dot(np.linalg.inv(ABC), d[:, np.newaxis]).flatten())
    N = np.array([N0] + N + [Nn])
    print("N with boundaries: \n", N)
    # S = [[f[i-1], (f[i]-f[i-1])/h[i-1]-(h[i-1]*M[i-1])/3-(h[i-1]*M[i])/6, M[i-1]/2, (M[i]-M[i-1])/(6*h[i-1])] for i in range(1, n)]
    S = [[f[i-1], N[i-1], 3*(f[i]-f[i-1])/h[i-1]**2-(2*N[i-1]+N[i])/h[i-1], -2*(f[i]-f[i-1])/h[i-1]**3+(N[i-1]+N[i])/h[i-1]**2] for i in range(1, n)]
    print("S-COEFFICIENTS: \n", np.array(S))
    coeffs = [[S[i][0]-S[i][1]*x_obs[i]+S[i][2]*x_obs[i]**2-S[i][3]*x_obs[i]**3, S[i][1]-2*S[i][2]*x_obs[i]+3*S[i][3]*x_obs[i]**2, S[i][2]-3*S[i][3]*x_obs[i], S[i][3]] for i in range(len(x_obs)-1)]
    return np.array(coeffs)

def CubicSplineB(x_obs, y_obs, M0, Mn):
    n=len(x_obs)
    f = np.array(y_obs) 
    h = np.array([x_obs[i+1]-x_obs[i] for i in range(n-1)])
    print("h: \n", h)
    abc = np.array([[h[i], 2*(h[i]+h[i+1]), h[i+1]] for i in range(n-2)])
    print("abc: \n", abc)
    d = np.array([6*(f[i+1]-f[i])/h[i]-6*(f[i]-f[i-1])/h[i-1] for i in range(1, n-1)])
    print("d: \n", d)
    d[0] = d[0]-abc[0][0]*M0
    d[-1] = d[-1]-abc[-1][2]*Mn
    print("Updated d: \n", d)
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
    print("ABC: \n", ABC)
    M = list(np.dot(np.linalg.inv(ABC), d[:, np.newaxis]).flatten())
    M = np.array([M0] + M + [Mn])
    print("M with boundaries: \n", M)
    S = [[f[i-1], (f[i]-f[i-1])/h[i-1]-(h[i-1]*M[i-1])/3-(h[i-1]*M[i])/6, M[i-1]/2, (M[i]-M[i-1])/(6*h[i-1])] for i in range(1, n)]
    print("S-COEFFICIENTS: \n", S)
    coeffs = [[S[i][0]-S[i][1]*x_obs[i]+S[i][2]*x_obs[i]**2-S[i][3]*x_obs[i]**3, S[i][1]-2*S[i][2]*x_obs[i]+3*S[i][3]*x_obs[i]**2, S[i][2]-3*S[i][3]*x_obs[i], S[i][3]] for i in range(len(x_obs)-1)]
    return np.array(coeffs)

if __name__ == '__main__':
    # derivative_condition = 2
    # x_obs, y_obs = DatasetGenerator().generate_random_xy_values(-10, 10, -10, 10, 100, 8)
    # coeffs = CubicSpline(x_obs, y_obs, derivative_condition, boundary_conditions)

    choiceA = False
    x_obs, y_obs = [-1, 0, 2], [1, 2, -1]
    boundary_conditions = [0, 0]

    coeffs = CubicSplineA(x_obs, y_obs, boundary_conditions[0], boundary_conditions[1]) if choiceA else CubicSplineB(x_obs, y_obs, boundary_conditions[0], boundary_conditions[1])

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

    colors = ['r', 'g', 'y', 'b']
    plt.figure(figsize = (10, 5))
    plt.scatter(x_obs, y_obs, s=10)
    for i in range(len(s1)):
        plt.plot(x_dist[i], s1[i], colors[i%4]+'-', label="A: S" + str(i+1) + "")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



