# Swarm-Algorithms
Python implemntation of swarm algorithms used for solveing non-convex optimization problems. 

<img src="images/pso_visualisation.png" alt="drawing" width="500px"/>

## Getting Started

In order to check how the implemented algorithms work just run [main.py](main.py). Originally it will use PSO for optimization of Auckley function. 
If you want to check either different function or different optimization method just uncomment the codeline. 
The default training loop for every i=30 steps show the positions of agents on a 2d visualisation. Furthermore it show a 3d visualisation after the end of optimization process. 

```python
NUMBER_OF_STEPS = 30
NUMBER_OF_AGENTS = 50

optimised_function = Auckley()
#optimised_function = Michalkiewicz()

if __name__ == '__main__':
    swarm = PSO(optimised_function=optimised_function, number_of_agents=NUMBER_OF_AGENTS)
    # swarm = SPP(optimised_function=optimised_function, number_of_agents=NUMBER_OF_AGENTS)
    # swarm = AAA(optimised_function=optimised_function, number_of_agents=NUMBER_OF_AGENTS)

    particles_in_step = []

    for i in trange(NUMBER_OF_STEPS):
        swarm.step()
        if i % 30 == 0:
            print(swarm.optimized_function(swarm.best_solution), swarm.best_solution)
            optimised_function.plot_2d(points=np.array([particle.position for particle in swarm.particles]), dirs=np.array([particle.velocity for particle in swarm.particles]))

    optimised_function.plot_3d(points=np.array([particle.position for particle in swarm.particles]))

```

## Implemented Swarm Algorithms
Three different optimisation algorithms have been implemented. Each algorithm for both functions is capable of finding the global solution in a finite number of steps.

### Paricle Swar

### Self Propelled Particles

### Artificial Algae Algorithm

Method inspired by [Artificial algae algorithm (AAA) for nonlinear global optimization, Uymaz et al.](https://www.sciencedirect.com/science/article/abs/pii/S1568494615001465?via%3Dihub)

## Testing functions

Two non-convex function functions - [Auckley](https://www.sfu.ca/~ssurjano/ackley.html) and [Gierwank](https://www.sfu.ca/~ssurjano/griewank.html) has been implemented. Both functions work for any number of dimensions (n).

### Auckley function

Auckley is as a classic non-convex function that is used for evaluation of swarm optimization algorithms. It is given by the following equation

<a href="https://www.codecogs.com/eqnedit.php?latex=f(\mathbf{x})&space;=&space;-20&space;exp(-0.2&space;\sqrt{\frac{1}{n}&space;\sum_{i=1}^n&space;x_i^2})&space;-&space;exp(\frac{1}{n}&space;\sum_{i=1}^n&space;cos(2\pi&space;x_i))&space;&plus;&space;20&space;&plus;&space;e" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(\mathbf{x})&space;=&space;-20&space;exp(-0.2&space;\sqrt{\frac{1}{n}&space;\sum_{i=1}^n&space;x_i^2})&space;-&space;exp(\frac{1}{n}&space;\sum_{i=1}^n&space;cos(2\pi&space;x_i))&space;&plus;&space;20&space;&plus;&space;e" title="f(\mathbf{x}) = -20 exp(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}) - exp(\frac{1}{n} \sum_{i=1}^n cos(2\pi x_i)) + 20 + e" /></a>

And has global minimim at:

<a href="https://www.codecogs.com/eqnedit.php?latex=f(0,&space;\cdots,&space;0)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(0,&space;\cdots,&space;0)&space;=&space;0" title="f(0, \cdots, 0) = 0" /></a>

<img src="images/ackley.png " alt="drawing" width="500px"/>

### Gierwank function
Gierwank is another non-convex functon that can be used for evaluation of swarm optimization algorithms. It is given by the following equation

<a href="https://www.codecogs.com/eqnedit.php?latex=f(\mathbf{x})&space;=&space;1&space;&plus;&space;\frac{1}{4000}&space;\sum_{i=1}^n&space;x^2_i&space;-&space;\prod_{i=1}^n&space;cos(\frac{x_i}{\sqrt{i}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(\mathbf{x})&space;=&space;1&space;&plus;&space;\frac{1}{4000}&space;\sum_{i=1}^n&space;x^2_i&space;-&space;\prod_{i=1}^n&space;cos(\frac{x_i}{\sqrt{i}})" title="f(\mathbf{x}) = 1 + \frac{1}{4000} \sum_{i=1}^n x^2_i - \prod_{i=1}^n cos(\frac{x_i}{\sqrt{i}})" /></a>

And has global minimim at:

<a href="https://www.codecogs.com/eqnedit.php?latex=f(0,&space;\cdots,&space;0)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(0,&space;\cdots,&space;0)&space;=&space;0" title="f(0, \cdots, 0) = 0" /></a>

<img src="images/gierwank.png " alt="drawing" width="500px"/>

## Visualisations
In order to better understand how the alogrithms works we propose two methods of visualzation. Boths methods work with 2D reaizations of optimised functions. 
First method scatter the function on 2D plane and uses color as a function value at a point. Moreover for PSO and SPP we show velocity vectors (AAA has no such thing). Second metod is using 3rd dimension to show the actual funtion value.
For both methods we show function values for every point from the domain as well as positions of every agent. 

Both visualisation methods can be found in [abstract_testing_function.py](abstract_testing_function.py)
## Authors

* [tugot17](https://github.com/tugot17)
* [snufkin12](https://github.com/snufkin12)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
