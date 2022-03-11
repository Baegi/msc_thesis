from dataclasses import dataclass
import numpy as np
from random import randrange
import math

M = np.diag([1, 1, 1, -1])

@dataclass
class Vertexer:

    nodes: np.ndarray

    # Defaults
    v = 299792458

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
        print(A)
        At = np.transpose(A)
        print("At")
        print(At)
        AtA = np.matmul(At, A)
        print("AtA")
        print(AtA)
        invAtA = np.linalg.inv(AtA)
        print("invAtA")
        print(invAtA)
        A_plus = np.matmul(invAtA, At)
        print("A_plus")
        print(A_plus)


        b = 0.5 * lorentzInner(A, A)
        print("b")
        print(b)
        #oneA = np.linalg.solve(A_plus, np.ones(4))
        #invA = np.linalg.solve(A_plus, b)

        oneA_plus = np.matmul(A_plus, np.ones(len(self.nodes)))
        invA_plus = np.matmul(A_plus, b)

        print("oneA_plus", oneA_plus.shape, np.ones(len(self.nodes)).shape)
        print(oneA_plus)
        print("invA_plus")
        print(invA_plus)


        solution = []
        for Lambda in np.roots([ lorentzInner(oneA_plus, oneA_plus),
                                (lorentzInner(oneA_plus, invA_plus) - 1) * 2,
                                 lorentzInner(invA_plus, invA_plus),
                                ]):
            #X, Y, Z, T = M @ np.linalg.solve(, Lambda * np.ones(len(self.nodes)) + b)
            X, Y, Z, T = np.matmul(A_plus, (b + Lambda * np.ones(len(self.nodes))))
            solution.append(np.array([X,Y,Z]))
            print("Candidate:", X, Y, Z, math.sqrt(X**2 + Y**2 + Z**2))
    
        print()
        print()
        return min(solution, key = lambda err: self.errFunc(err, times))


# Simulate sources to test code
#

# Set velocity
c = 299792458 # m/s

# Pick nodes to be at random locations
x_1 = c; y_1 = 0; z_1 = 0
x_2 = 0; y_2 = c; z_2 = 0
x_3 = 0; y_3 = 0; z_3 = c
x_4 = -c; y_4 = 0; z_4 = 0
x_5 = c; y_5 = c; z_5 = c

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

'''
sensor_positions = [
    [ c, 0, 0 ],
    [ 0, c, 0 ],
    [ 0, 0, c ],
    [ 0, -c, 0 ],
    [ -c, 0, 0 ]
]

timestamps = [ 1,1,1,.99999, 1.000001 ]
'''

'''
1605.7264958657324 (3880282.877185033, 1094577.258990629, 4926169.276111776)
1605.7264959624852 (3880319.20724456, 1094385.0297938874, 4926154.921248427)
1605.7265816501094 (3959479.726311205, 1035065.1793950492, 4875876.687624674)
1605.726389895923 (3942857.0424463185, 1408130.0150678533, 4796307.437384814)
'''
sensor_positions = [
    [ 3880282.877185033, 1094577.258990629, 4926169.276111776 ],
    [ 3880319.20724456, 1094385.0297938874, 4926154.921248427 ],
    [ 3959479.726311205, 1035065.1793950492, 4875876.687624674 ],
    [ 3942857.0424463185, 1408130.0150678533, 4796307.437384814 ]
]
timestamps = [
    0,
    -0.0000000967527285,
    -0.00008578437,
    0.0001059698
]


myVertexer = Vertexer(np.array(sensor_positions))
print(myVertexer.find(np.array([[e] for e in timestamps])))
