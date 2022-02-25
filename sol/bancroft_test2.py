


from dataclasses import dataclass
import numpy as np
from random import randrange
import math

M = np.diag([1, 1, 1, -1])

@dataclass
class Vertexer:

    nodes: np.ndarray

    # Defaults
    v = 299792

    def __post_init__(self):
        # Calculate valid input range
        max = 0
        min = 1E+10
        centroid = np.average(self.nodes, axis = 0)
        for n in self.nodes:
            dist = np.linalg.norm(n - centroid)
            if dist < min:
                min = dist

            for p in self.nodes:
                dist = np.linalg.norm(n - p)

                if dist > max:
                    max = dist

        max /= self.v
        min /= self.v

        print(min, max)

    def errFunc(self, point, times):
        # Return RSS error
        error = 0

        for n, t in zip(self.nodes, times):
            error += ((np.linalg.norm(n - point) / self.v) - t)**2

        return error

    def find(self, times):
        def lorentzInner(v, w):
            # Return Lorentzian Inner-Product
            return np.sum(v * (w @ M), axis = -1)

        A = np.append(self.nodes, times * self.v, axis = 1)

        b = 0.5 * lorentzInner(A, A)
        oneA = np.linalg.solve(A, np.ones(4))
        invA = np.linalg.solve(A, b)

        solution = []
        for Lambda in np.roots([ lorentzInner(oneA, oneA),
                                (lorentzInner(oneA, invA) - 1) * 2,
                                 lorentzInner(invA, invA),
                                ]):
            X, Y, Z, T = M @ np.linalg.solve(A, Lambda * np.ones(4) + b)
            solution.append(np.array([X,Y,Z]))
    
        return min(solution, key = lambda err: self.errFunc(err, times))


# Simulate sources to test code
#

# Set velocity
c = 299792 # km/ns

# Pick nodes to be at random locations
x_1 = c; y_1 = 0; z_1 = 0
x_2 = 0; y_2 = c; z_2 = 0
x_3 = 0; y_3 = 0; z_3 = c
x_4 = -c; y_4 = 0; z_4 = 0
x_5 = 0; y_5 = c - 1000; z_5 = 0

# Pick source to be at random location
x = 0; y = 0; z = 0


# Generate simulated source
t_1 = math.sqrt( (x - x_1)**2 + (y - y_1)**2 + (z - z_1)**2 ) / c
t_2 = math.sqrt( (x - x_2)**2 + (y - y_2)**2 + (z - z_2)**2 ) / c
t_3 = math.sqrt( (x - x_3)**2 + (y - y_3)**2 + (z - z_3)**2 ) / c
t_4 = math.sqrt( (x - x_4)**2 + (y - y_4)**2 + (z - z_4)**2 ) / c
t_5 = math.sqrt( (x - x_5)**2 + (y - y_5)**2 + (z - z_5)**2 ) / c

print('Actual:', x, y, z)

myVertexer = Vertexer(np.array([[x_1, y_1, z_1],[x_2, y_2, z_2],[x_3, y_3, z_3],[x_4, y_4, z_4],[x_5, y_5, z_5]]))
print(myVertexer.find(np.array([[t_1], [t_2], [t_3], [t_4], [t_5]])))