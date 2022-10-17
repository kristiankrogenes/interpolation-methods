import matplotlib.pyplot as plt

class Plotter():

    def plot_interpolation(self, x_obs, y_obs, x_dist, y_inter):
        colors = ['r', 'g', 'y', 'b']

        plt.figure(figsize = (10, 5))
        plt.scatter(x_obs, y_obs, s=10)
        # plt.plot(x_dist, y_inter, 'r-', label='Interpolated polynomial')
        for i in range(len(y_inter)):
            plt.plot(x_dist[i], y_inter[i], colors[i%4]+'-', label="S" + str(i+1) + "")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()