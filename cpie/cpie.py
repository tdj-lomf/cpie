import math
import numpy as np

from enclosure import Enclosure
from solution import Solution


class CPie:

    def __init__(self, bounds_min, bounds_max, Ns=None):
        # problem settings
        self._dimension = len(bounds_min)
        self._bounds_min = np.array([b_min for b_min in bounds_min])
        self._bounds_max = np.array([b_max for b_max in bounds_max])

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

        # inner states
        self._enclosures = []
        self._initial_solutions = []
        self._current_enclosure = None
        self._sample = None
        self._f_value = None

        self.best = None
        self.iteration = 0

    def sample(self):
        if not self._enclosures:
            bounds_range = self._bounds_max - self._bounds_min
            self._sample = np.random.random(self._dimension) * bounds_range + self._bounds_min
            return self._sample
        else:
            self._current_enclosure = self._select_enclosure()
            self._sample = self._current_enclosure.sample(self._q, self._phi_max)
        return self._sample

    def update(self, f_value):
        self.iteration += 1
        self._f_value = f_value
        # update enclosure
        if not self._enclosures:    # initialization
            if f_value < float('inf'):
                self._initial_solutions.append(Solution(self._sample.copy(), f_value))
                if len(self._initial_solutions) == self._Ns:
                    # generate first enclosure
                    enclosure = Enclosure(self._alpha, self._initial_solutions)
                    self._enclosures.append(enclosure)
        else:
            self._current_enclosure.update(f_value, self._Ns, self._alpha, self._p, self._u)

        if not self.best or f_value < self.best.f:
            self.best = Solution(self._sample.copy(), f_value)

        if not self._current_enclosure:
            return

        if self._current_enclosure.merge(self._enclosures, self._Ns, self._zeta, self._alpha):
            return

        self._current_enclosure.divide(self._enclosures, self._o, self._rho, self._alpha)

    def print(self):
        print("iter:", self.iteration, "f:", self._f_value, "f_best:", self.best.f, "num mode:", len(self._enclosures))
    
    def get_bests(self):
        return [e.get_best() for e in self._enclosures]

    def _select_enclosure(self):
        if len(self._enclosures) == 1:
            selected = self._enclosures[0]
        elif np.random.random() < self._epsilon:
            selected = np.random.choice(self._enclosures, size=1)[0]
        else:
            selected = self._select_promising_enclosure()
        selected.add_t_count()
        return selected
    
    def _select_promising_enclosure(self):
        # if not enough data, then return the enclosure
        for e in self._enclosures:
            if len(e._average_history) < self._u:
                return e
        # select by improvement rate
        estimated_best = min(e.next_sample_estimation() for e in self._enclosures)
        for e in self._enclosures:
            e.calc_improvement_rate(estimated_best)
        return max(self._enclosures, key=lambda e: e.improvement_rate)

