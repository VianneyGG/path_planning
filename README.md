# path_planing
Code for the INF421 Path Planing Course

# Setting up the environment

## Question 1

For this question, we decied to use a class ```Environment``` to capture the input and to represent an instance of a path planning problem. This acts as an interface for the different algorithms to interact with, facilitating rendering of different scenarios. 

## Question 2

We wrote the method ```render``` that shows a particular environment to the user. We used the libraries `matplotlib` and `shapely.geometry` to render the geometry.
To represent the path, we designed an abstract class `AbstractWaypoint` to represent a point in space and possibly other attributes specific to an algorithm. Similarly, the abstract class `AbstractPath` represents a collection of `Waypoints`.

# A first approach: Particle Swarm Optimization (PSO)

$S$: Number of particles

$k$: iteration step

Particle $i$ caracterized by postion $x_i^k$ and velocity $v_i^k$ at time $k$

$$
\begin{equation}
v_i^{k+1} = wv_i^k + c_1 r_1 (p_i^k - x_i^k + c_2 r_2 (g^k - x_i^k)) \\
\end{equation}
$$
$$
\begin{equation}
x_i^{k+1} = x_i^k + v_i^k+1
\end{equation}
$$
$c_1, c_2$: acceleration coefficients, which control the influence of the local and global best solutions

$w$: inertia weight for velocity

$r_1, r_2$: random variables uniformly sampled on $[0,1]$


## Question 3

--


# A second approach: Rapidly-exploring Random Tree (RRT)

## Question 13

We propose to use a recursive definition for the Tree:
```
TreeNode: 
    Position: Waypoint           # position of the node in space
    Children: [TreeNode] | None  # not fixed size 
    Parent:   TreeNode   | None  # Parent Node used for backtracking
```

## Question 14

Thanks to the Parent attribute to our Tree datastructure, we can easily backtrack by iteratively looking at the node's parent until reaching None Parent, meaning that we reached the root of the tree.

## Question 15

```
Pseudocode for RRT:
Initialise the Root to the StartPosition
While Stopping Criteria Not Met:
    While True
        Sample v_r randomly within the grid
        Find v_n the closest node to v_r within the existing nodes such that the path from v := v_n + delta * (v_r - v_n) / |v_r - v_n| and v_n does not cross an obstacle.
        If no such point exists, find a new v_r and repeat.
        Else, break out of the while loop.
    Add v to the Children of v_n and set v_n as the parent of v


```