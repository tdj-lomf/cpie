"""CPIE sub module providing Enclosure class.
"""

import copy
import math
import numpy as np
from .ellipsoid import Ellipsoid
from .solution import Solution
from .helper import mean_and_covariance
from .helper import k_means_mahalanobis


class Enclosure:
    """ Enclosure class that is used in CPie.
    An enclosure has an ellipsoid and promising solutions. The ellipsoid
    expands or shrinks to enclose the promising solutions at each step.
    """

    def __init__(self, alpha, solutions, ellipsoid=None):
        """Enclosure constructor.
        Arguments:
            alpha {float} --  Coefficient of updating an ellipsoid
            solutions {list of Solution} -- Solutions that would be enclosed by the ellipsoid
        Keyword Arguments:
            ellipsoid {Ellipsoid} -- Initial ellipsoid (default: {None})
        """
        self.dimension = solutions[0].x.size
        self.solutions = copy.deepcopy(solutions)
        self.improvement_rate = None
        # create an ellipsoid to enclose solutions
        self.ellipsoid = Ellipsoid(self.solutions, alpha) if not ellipsoid else ellipsoid

        # inner states
        self._sample = None
        self._t_count = 0
        self._q_count = 0
        self._o_count = 0
        self._average_history = []
        self._average = sum(s.f for s in self.solutions) / len(self.solutions)
        self._worst = max(s.f for s in self.solutions)
        self._normal_sample = True

    def get_best(self):
        """Get the best solution sampled by this enclosure
        Returns:
            Solution -- Best solution sampled by this enclosure
        """
        return min(self.solutions, key=lambda s: s.f)

    def sample(self, q, phi_max):
        """Sample a solution on the surface of the ellipsoid
        Arguments:
            q {int} -- Coefficient of deciding sampling method
            phi_max {float} -- Max radians related to sampling direction
        Returns:
            numpy array -- Solution x that should be evaluated next
        """
        self._normal_sample = self._q_count < q or self.dimension == 1
        if self._normal_sample:
            self._sample = self.ellipsoid.sample()
        else:
            base_solution = self._roulette_selection()
            self._sample = self.ellipsoid.sample_near(base_solution.x, phi_max)
        return self._sample

    def update(self, f_value, Ns, alpha, p, u):
        """Update this enclosure with evaluated function value (f_value)
        Arguments:
            f_value {float} -- Evaluated function value of the latest solution
            Ns {int} -- Max population size
            alpha {float} --  Coefficient of updating an ellipsoid
            p {int} -- Frequency of re-enclosing solutions
            u {int} -- Max record number of average history
        """
        self._update_counter(f_value, Ns)
        self._update_solutions(f_value, Ns, u)
        self._update_ellipsoid(f_value, alpha)
        if self._t_count % p == 0:
            self.ellipsoid.reenclose(self.solutions, alpha)

    def add_t_count(self):
        """Increment T(times of update).
        T is used in 'update' method and affect to frequency of re-enclosing.
        """
        self._t_count += 1

    def merge(self, enclosures, Ns, zeta, alpha):
        """Merge another enclosre into this one if the search spaces seems to be duplicated.
        Arguments:
            enclosures {list of Enclosure} -- Candidates to merge
            Ns {int} -- Max population size
            zeta {float} -- Threshold of mahalanobis distance between tow ellipsoids
            alpha {float} --  Coefficient of updating an ellipsoid
        Returns:
            bool -- True if merged
        """
        if len(self.solutions) < Ns:
            return False
        target_index, target = self._search_merge_target(enclosures, Ns, zeta)
        if not target:
            return False

        # merge solutions
        self.solutions += target.solutions
        self.solutions.sort(key=lambda s: s.f)
        del self.solutions[Ns:]  # delete worse(larger) solutions
        # reenclose new solutions
        if self._average < target._average:
            self.ellipsoid.reenclose(self.solutions, alpha)
        else:
            target.ellipsoid.reenclose(self.solutions, alpha)
            self.ellipsoid = target.ellipsoid
        self._worst = self.solutions[-1].f
        self._average = sum(s.f for s in self.solutions) / len(self.solutions)
        self._average_history.clear()
        self._q_count = 0
        self._o_count = 0
        del enclosures[target_index]
        return True

    def divide(self, enclosures, o, rho, alpha):
        """Divide this enclosure when good solutions are hardly sampled.
        The solutions are divided by k-means clustering with mahalanobis distance.
        Arguments:
            enclosures {list of Enclosure} -- List for newly generated enclosure
            o {int} -- Frequency of trying division
            rho {float} -- Threshold of mahalanobis distance
            alpha {float} --  Coefficient of updating an ellipsoid
        """
        if self._o_count < o:
            return
        self._o_count = 0
        class1, class2 = k_means_mahalanobis(self.solutions, k=2)
        if len(class1) <= self.dimension or len(class2) <= self.dimension:
            return
        # cancel dividing if the clustered solutions are mixed up
        ellipsoid1, ellipsoid2 = Ellipsoid(class1, alpha), Ellipsoid(class2, alpha)
        for e, c in zip((ellipsoid1, ellipsoid2), (class2, class1)):
            distances = [e.mahalanomis_distance(s.x) for s in c]
            if min(distances) <= rho:
                return
        # define worse set and better set
        average1, average2 = [sum(s.f for s in c)/len(c) for c in (class1, class2)]
        if average1 < average2:
            better = (class1, ellipsoid1, average1)
            worse = (class2, ellipsoid2, average2)
        else:
            worse = (class1, ellipsoid1, average1)
            better = (class2, ellipsoid2, average2)
        # add new enclosure with worse set
        worse_enclosure = Enclosure(alpha, solutions=worse[0], ellipsoid=worse[1])
        enclosures.append(worse_enclosure)
        # reinitialize this enclosure(self) with better set
        self.solutions = better[0]
        self.ellipsoid.shrink(worse[0], alpha)
        self._q_count = 0
        self._worst = max(s.f for s in self.solutions)
        self._average = better[2]
        self._average_history.clear()

    def next_sample_estimation(self):
        """Estimate evaluation value of a solution that would be sampled next.
        Returns:
            float -- Estimated evaluation value
        """
        best = min(s.f for s in self.solutions)
        variance = sum((s.f-best)*(s.f-best) for s in self.solutions) / (len(self.solutions) - 1)
        sigma = math.sqrt(variance)
        return np.random.randn() * sigma + best

    def calc_improvement_rate(self, best):
        """Calculate improvement rate of this enclosure
        Arguments:
            best {float} -- Estimated best value of the target problem
        """
        self.improvement_rate = (self._average_history[0] - best) / (self._average - best)

    def _update_solutions(self, f_value, Ns, u):
        if f_value >= self._worst:
            return
        if len(self.solutions) < Ns:
            solution = Solution(copy.deepcopy(self._sample), f_value)
            self.solutions.append(solution)
            sol_size = len(self.solutions)
            self._average = (self._average * (sol_size - 1) + f_value) / sol_size
        else:
            if self._normal_sample:
                replace = max(self.solutions, key=lambda s: s.f)
            else:
                # replace nearest worse solution
                worse_solutions = [s for s in self.solutions if s.f > f_value]
                for s in worse_solutions:
                    # mahalanobis distance between s and current sample
                    s.distance = np.linalg.norm(np.dot(self.ellipsoid.BInv, s.x - self._sample))
                replace = min(worse_solutions, key=lambda s: s.distance)
            self._average_history.append(self._average)
            if len(self._average_history) > u:
                del self._average_history[0]
            sol_size = len(self.solutions)
            self._average = (self._average * sol_size - replace.f + f_value) / sol_size
            replace.f = f_value
            replace.x = self._sample.copy()
            self._worst = max(s.f for s in self.solutions)

    def _update_ellipsoid(self, f_value, alpha):
        if f_value < self._worst:
            self.ellipsoid.update(self._sample, alpha)
        elif f_value > self._worst:
            self.ellipsoid.update(self._sample, -alpha)

    def _update_counter(self, f_value, Ns):
        if f_value < self._worst:
            self._q_count = 0
            if self._normal_sample:
                self._o_count = 0
        elif len(self.solutions) == Ns:
            self._q_count += 1
            if self._normal_sample:
                self._o_count += 1

    def _search_merge_target(self, enclosures, Ns, zeta):
        for i, e in enumerate(enclosures):
            if e is self or len(e.solutions) < Ns:
                continue
            d1 = self.ellipsoid.mahalanomis_distance(e.ellipsoid.mu)
            if d1 >= zeta:
                continue
            d2 = e.ellipsoid.mahalanomis_distance(self.ellipsoid.mu)
            if d2 >= zeta:
                continue
            return i, e
        return None, None

    def _roulette_selection(self):
        mu, covariance = mean_and_covariance(self.solutions)
        BInv = np.linalg.inv(np.linalg.cholesky(covariance))
        for s in self.solutions:
            s.distance = np.linalg.norm(np.dot(BInv, s.x - mu))
        self.solutions.sort(key=lambda s: -s.distance)  # descending order
        candidate_num = max(len(self.solutions) // 4, 1)
        candidates = self.solutions[:candidate_num]
        total = sum(s.distance for s in candidates)
        return np.random.choice(candidates, size=1, replace=False,
                                p=[s.distance/total for s in candidates])[0]
