from cpie import CPie


def sphere(x):
    return sum(xi*xi for xi in x)

def objective_func(x):
    return min(sphere(x-2)+0.1, 10*sphere(x+2))



if __name__ == "__main__":
    dimension = 2
    bounds_min = [-10.0] * dimension
    bounds_max = [ 10.0] * dimension
    cpie = CPie(bounds_min, bounds_max, Ns=7*dimension)
    for i in range(2000):
        solution = cpie.sample()
        f_value = objective_func(solution)
        cpie.update(f_value)
        cpie.print()
    print("best x", cpie.best.x)
    print("best f", cpie.best.f)
