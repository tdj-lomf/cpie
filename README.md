# cpie
Clustering-based Promising Individual Enclosre(CPIE) in Python.  

CPIE is an evolutionary computation algorithm which aims to search best parameters minimizing black-box function, especially UV-function.  
UV-function has U-valley, which occupies large search space but  contains only local optima, and V-valley, which occupies small search space but contains global optima.  
e.g. Double-sphere: f(x) = min((x-2)^2 + 0.1, 10*(x+2)^2)  

# Installation
```
$ pip install cpie
```
CPIE depends on numpy, so please install numpy if needed.

# Example of usage
First, you can import CPIE module as below.  
```python
    from cpie import CPie
```

Then, you need to prepare objective function to be minimized.  
```python
    def sphere(x):
        return sum(xi*xi for xi in x)

    def objective_func(x):
        return min(sphere(x-2)+0.1, 10*sphere(x+2))
```

You can minimize objective function like below.  
```python
    dimension = 2
    bounds_min = [-10.0] * dimension
    bounds_max = [ 10.0] * dimension
    cpie = CPie(bounds_min, bounds_max, Ns=7*dimension)
    for i in range(2000):
        solution = cpie.sample()
        f_value = objective_func(solution)
        cpie.update(f_value)
        cpie.print()
```
"bounds_min" and "bounds_max" means search space.  
CPIE starts optimization with Ns solutions sampled unimormally in the search space.  

After optimization loop, you can get optimized solution.
```python
    print("global best x", cpie.best.x)
    print("global best f", cpie.best.f)
```

CPIE is niching algorithm, so you can also get best solutions from each mode.  
```python
    bests = cpie.get_bests()
    for i, b in enumerate(bests):
        print("mode", i, " f", b.f)
```

example_main.py shows full example code.
