import math, random
import numpy as np

class DatasetGenerator():

    def generate_random_polynomial(self, deg):
         return np.array([random.uniform(-10, 10) for _ in range(deg+1)])
    
    def get_random_points_from_polynomial(self, coeffs, interval, n):
        x = random.sample(list(interval), n)
        y = np.sum(np.array([[c*dx**i for i, c in enumerate(coeffs)] for dx in x]), axis=1)
        return x, y
    
    def get_random_points_from_polynomial_with_noise(self, coeffs, interval, n):
        x = np.sort(random.sample(list(interval), n))
        y = np.sum(np.array([[c*dx**i + random.uniform(-50, 50) for i, c in enumerate(coeffs)] for dx in x]), axis=1)
        return x, y

    def generate_random_xy_values(self, xmin, xmax, ymin, ymax, f, n):
        x_dist = np.linspace(xmin, xmax, f)
        y_dist = np.linspace(ymin, ymax, f)
        return np.sort(random.sample(list(x_dist), n)), np.array(random.sample(list(y_dist), n))

    def get_special_data(self):
        x_obs = [i for i in range(1, 100)]
        y_obs = [0, 83, 174, 276, 388, 512, 650, 801, 969, 1154, 1358, 1584, 1833, 2107, 2411, 2746, 3115, 3523, 3973, 4470, 5018, 5624, 6291, 7028, 7842, 8740, 9730, 10824, 12031, 13363, 14833, 16456, 18247, 20224, 22406, 24815, 27473, 30408, 33648, 37224, 41171, 45529, 50339, 55649, 61512, 67983, 75127, 83014, 91721, 101333, 111945, 123660, 136594, 150872, 166636, 184040, 203254, 224466, 247886, 273742, 302288, 333804, 368599, 407015, 449428, 496254, 547953, 605032, 668051, 737627, 814445, 899257, 992895, 1096278, 1210421, 1336443, 1475581, 1629200, 1798808, 1986068, 2192818, 2421087, 2673114, 2951373, 3258594, 3597792, 3972294, 4385776, 4842295, 5346332, 5902831, 6517253, 7195629, 7944614, 8771558, 9684577, 10692629, 11805606, 13034431]
        return x_obs, y_obs