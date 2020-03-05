"""Main module of the Clustering-based Promising Individual Enclosure
providing class 'CPie'
"""

import math
import numpy as np

from .enclosure import Enclosure
from .solution import Solution


class CPie:
    """Clustering-based Promising Individual Enclosure.
    Evolutionary computation algorithm which aims to search best
    parameters minimizing black-box function, especially UV-function.
    UV-function has U-valley, which occupies large search space but
    contains only local optima, and V-valley, which occupies small
    search space but contains global optima.
    e.g. Double-sphere: f(x) = min((x-2)^2 + 0.1, 10*(x+2)^2)

    Example Usage: Please see example_main.py
        def main():
            dimension = 2
            bounds_min = [-10.0] * dimension
            bounds_max = [ 10.0] * dimension
            cpie = CPie(bounds_min, bounds_max, Ns=7*dimension)
            for i in range(2000):
                solution = cpie.sample()
                f_value = objective_func(solution)
                cpie.update(f_value)
                cpie.print()
            print("global best x", cpie.best.x)
            print("global best f", cpie.best.f)
            bests = cpie.get_bests()
            for i, b in enumerate(bests):
                print("mode", i, " f", b.f)

    Returns:
        List of Solution class -- Optimized solutions [s1, s2, ...]
        Each solution has a parameter x and its evaluation value f.
    """

    def __init__(self, bounds_min, bounds_max, Ns=None, options=None):
        """CPie constructor
        Arguments:
            bounds_min {list or numpy array} -- Minimum search range of X
            bounds_max {list or numpy array} -- Maximum search range of X
        Keyword Arguments:
            Ns {int} -- Population size (default: {None})
            options {dictionary} -- Additional options such as "max_mode" (default: {None})
        """

        # problem settings
        self._dimension = len(bounds_min)
        self._bounds_min = np.array(bounds_min)
        self._bounds_max = np.array(bounds_max)

        # hyper parameters
        self._Ns = 7 * self._dimension if not Ns else Ns
        self._alpha = 1.0 / (1.0 + self._dimension)
        self._epsilon = 0.5
        self._rho = 1.6
        self._zeta = 0.6
        self._phi_max = math.pi / 18.0
        self._p = 2 * self._dimension
        self._q = 3
        self._o = 10
        self._u = 5

        # additional options
        self._options = {
            'max_mode': None,   # Max num of dividing enclosure. Final best solutions <= 'max_mode'
        }
        if options:
            for key, value in options.items():
                self._options[key] = value

        # inner states
        self._enclosures = []
        self._initial_solutions = []
        self._current_enclosure = None
        self._sample = None
        self._f_value = None

        self.best = None
        self.iteration = 0

    def sample(self):
        """Sample a solution that should be evaluated next
        Returns:
            numpy array -- Array of solution x
        """
        if not self._enclosures:
            bounds_range = self._bounds_max - self._bounds_min
            self._sample = np.random.random(
                self._dimension) * bounds_range + self._bounds_min
        else:
            self._current_enclosure = self._select_enclosure()
            self._sample = self._current_enclosure.sample(
                self._q, self._phi_max)
        return self._sample

    def update(self, f_value):
        """Update CPIE inner states (The ellipsoid and population)
        Arguments:
            f_value {float} -- Evaluation value of the latest solution x
        """
        self.iteration += 1
        self._f_value = f_value
        # update enclosure
        if not self._enclosures:    # initialization
            if f_value < float('inf'):
                self._initial_solutions.append(
                    Solution(self._sample.copy(), f_value))
                if len(self._initial_solutions) == self._Ns:
                    # generate first enclosure
                    enclosure = Enclosure(self._alpha, self._initial_solutions)
                    self._enclosures.append(enclosure)
        else:
            self._current_enclosure.update(
                f_value, self._Ns, self._alpha, self._p, self._u)
        if not self.best or f_value < self.best.f:
            self.best = Solution(self._sample.copy(), f_value)

        # merge/divide enclosure
        if not self._current_enclosure:
            return
        if self._current_enclosure.merge(self._enclosures, self._Ns, self._zeta, self._alpha):
            return
        if not self._options['max_mode'] or len(self._enclosures) < self._options['max_mode']:
            self._current_enclosure.divide(
                self._enclosures, self._o, self._rho, self._alpha)

    def print(self):
        """Print inner states (iteration, f, f_best, num mode)
        """
        print("iter:", self.iteration, "f:", self._f_value, "f_best:",
              self.best.f, "num mode:", len(self._enclosures))

    def get_bests(self):
        """Get best solutions of each mode
        Returns:
            list of Solution class -- Best solutions of each mode
        """
        return [e.get_best() for e in self._enclosures]

    def _select_enclosure(self):
        """Select enclosre which will sample next solution
        Returns:
            Enclosre class -- Selected enclosure which contains an ellipsoid and population
        """
        if len(self._enclosures) == 1:
            selected = self._enclosures[0]
        elif np.random.random() < self._epsilon:
            selected = np.random.choice(self._enclosures, size=1)[0]
        else:
            selected = self._select_promising_enclosure()
        selected.add_t_count()
        return selected

    def _select_promising_enclosure(self):
        """Select promising enclosure from perspective of improvement rate
        Returns:
            Enclosure class -- An enclosre which has the highest improvement rate
        """
        # if not enough data, then return the enclosure
        for e in self._enclosures:
            if len(e._average_history) < self._u:
                return e
        # select by improvement rate
        estimated_best = min(e.next_sample_estimation()
                             for e in self._enclosures)
        for e in self._enclosures:
            e.calc_improvement_rate(estimated_best)
        return max(self._enclosures, key=lambda e: e.improvement_rate)
