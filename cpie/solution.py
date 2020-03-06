"""CPIE sub module providing Solution class
"""

class Solution:
    """A solution that has parameter array x, evaluation value f, and distance
    """
    def __init__(self, x, f):
        """Solution constructor
        Arguments:
            x {numpy array} -- Parameters
            f {float} -- Evaluation value of x
        """
        self.x = x
        self.f = f
        self.distance = float('inf')
