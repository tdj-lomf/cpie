"""CPIE sub module providing Ellipsoid class
"""

import math
import numpy as np
from .helper import mean_and_covariance

class Ellipsoid:
    """An ellipsoid used by an enclosure to sample solutions
    """

    def __init__(self, solutions, alpha):
        """Ellipsoid constructor
        Arguments:
            solutions {list of Solution} -- Solutions to enclose
            alpha {float} --  Coefficient of updating an ellipsoid
        """
        self.dimension = solutions[0].x.size
        self.mu, variance = mean_and_covariance(solutions)
        self.B = np.linalg.cholesky(variance)
        self.BInv = np.linalg.inv(self.B)
        self.reenclose(solutions, alpha)

    def sample(self):
        """Sample a solution x on this ellipsoid uniformally
        Returns:
            numpy array -- Sampled vector on this ellipsoid
        """
        z = np.random.randn(self.dimension)
        z /= np.linalg.norm(z)
        x = np.dot(self.B, z) + self.mu
        return x

    def sample_near(self, base_x, phi_max):
        """Sample a solution x on this ellipsoid, especially near 'base_x'
        Arguments:
            base_x {numpy array} -- Base position of sampling
            phi_max {float} -- Max radians related to sampling direction
        Returns:
            numpy array -- Sampled vector on this ellipsoid
        """
        base_z = np.dot(self.BInv, base_x - self.mu)
        e = base_z / np.linalg.norm(base_z)
        y = np.random.randn(self.dimension)
        y /= np.linalg.norm(y)
        inner = np.dot(y, e)
        e_cross = (-inner * e + y) / math.sqrt(1.0 - inner * inner)
        tan_phi = math.tan(np.random.random() * 2.0 * phi_max - phi_max)
        z = e_cross * tan_phi + e
        z /= np.linalg.norm(z)
        x = np.dot(self.B, z) + self.mu
        return x

    def reenclose(self, solutions, alpha):
        """Re-enclose solutions by expanding this ellipsoid
        Arguments:
            solutions {list of Solution} -- Solutions to enclose
            alpha {float} --  Coefficient of updating an ellipsoid
        """
        for s in solutions:
            s.distance = self.mahalanomis_distance(s.x)
        farthest = max(solutions, key=lambda s: s.distance)
        while farthest.distance > 1.0:
            d = farthest.distance
            alpha_enc = (2.0*(d+3.0)*(d-1.0) + alpha*(d+1.0)*(d+1.0)) / (8.0*d*d)
            self.update(farthest.x, alpha_enc)
            for s in solutions:
                s.distance = self.mahalanomis_distance(s.x)
            farthest = max(solutions, key=lambda s: s.distance)

    def shrink(self, solutions, alpha):
        """Shrink this ellipsoid not to enclosre solutions
        Arguments:
            solutions {list of Solution} -- Solutions to enclose
            alpha {float} --  Coefficient of updating an ellipsoid
        """
        for s in solutions:
            s.distance = self.mahalanomis_distance(s.x)
        nearest = min(solutions, key=lambda s: s.distance)
        while nearest.distance <= 1.0:
            d = nearest.distance
            alpha_s = (2.0*(d+3.0)*(d-1.0) - alpha*(d+1.0)*(d+1.0)) / (8.0*d*d)
            self.update(nearest.x, alpha_s)
            for s in solutions:
                s.distance = self.mahalanomis_distance(s.x)
            nearest = min(solutions, key=lambda s: s.distance)

    def mahalanomis_distance(self, x):
        """Calculate mahalanobis distance between the center of this ellipsoid and x
        Arguments:
            x {numpy array} -- target
        Returns:
            float -- Mahalanobis distance between the center of this ellipsoid and x
        """
        z = np.dot(self.BInv, x - self.mu)
        return np.linalg.norm(z)

    def update(self, x, alpha):
        """Update this ellipsoid to/not to enclose x
        Arguments:
            x {numpy array} -- Target vector to/not to enclose
            alpha {float} --  Coefficient of updating an ellipsoid
        """
        diff = x - self.mu
        z = np.dot(self.BInv, diff)
        z_norm = np.linalg.norm(z)
        gamma = (math.sqrt(1.0 + alpha*z_norm*z_norm) - 1.0) / (z_norm*z_norm)

        self.B += np.outer(diff, z) * gamma
        self.mu += diff * gamma * z_norm

        gamma_inv = -gamma / (1.0 + gamma*z_norm*z_norm)
        self.BInv += np.outer(z, np.dot(z.T, self.BInv)) * gamma_inv
