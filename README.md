# inverted_pendulum_simulation_and_control
Investigating deep learning techniques for controlling a simulated inverted pendulum.

![Final Model Animation](https://github.com/pcummer/inverted_pendulum_simulation_and_control/blob/master/pendulum_rectified.gif)

## Overview
This is an investigation of machine learning for a dynamic control task with a focus on understanding how it works rather than optimizing the performance. The goal is to stabilize a simulated pendulum in an inverted state by applying either a positive or negative torque as decided by a neural network trained via reinforcement learning. 

This task has a number of attractive properties: 
* Extremely mathematically tractable, not only on the simulation side, but also for investigating the control system
* Continuous, bounded state space with finite and discrete action space
* Requires long term planning to successfully invert pendulum when starting from resting state
* Solvable nearly perfectly with two rules as either baseline or demonstration

## Setup
Our pendulum is simulated in C++ and controlled via a Tensorflow neural network hosted on a web API in Python. At every time step the state of the pendulum is updated according to its equation of motion under an applied torque in C++. At every tenth time step an http request is made to the neural net, sending the state and recieving a direction for the applied torque, clockwise or counterclockwise. Repeating the action for ten time steps improves computational efficiency, samples a more diverse range of states, and emphasizes the impact of the choice. Ideally it would all sit in C++, but the Tensorflow bindings are painful and the http requests are fast compared to even the forward pass of a moderately sized neural net. 

![Setup Diagram](https://github.com/pcummer/inverted_pendulum_simulation_and_control/blob/master/Setup%20diagram.PNG)

For interesting physics, we require that our physical parameters fall within certain ranges. The applied torque must be less than a critical value, 𝜏<sub>0</sub> = 𝑚∗𝑔∗𝑙, so that control system cannot lift the pendulum from rest to inversion against gravity; instead, under this constraint it must pump energy into the system to achieve inversion. We'd also benefit from small damping proportial to 𝜔, the angular velocity, to avoid our simulation breaking down under sufficiently high 𝜔, though we of course need to allow the system to accumulate sufficient kinetic energy. Roughly this mean we want 𝜏<sub>damping</sub> < 𝜏<sub>0</sub> when the kinetic energy is equal to the potential energy at inversion, 𝑚∗𝑔∗2𝑙=1/2 𝑚∗(𝜔∗𝑙)<sup>2</sup>. 

We're otherwise free to choose our parameters and retain interesting physics, but we should also think how these paramaters affect our control system and its learning. In general the problem will be much harder to learn for larger moments of inertia as this effectively lengthens the time scale over which our control system must plan. Put another way, for larger moments, each action has a smaller impact on the trajectory of the system and therefore many such actions must be coordinated to effect a significant change in trajectory. Similarly, smaller applied torques require longer horizons.

We have one more trick here to simplify the learning the process. The obvious state representation is -𝜋 < 𝜃 ≤ 𝜋 due to the cyclic nature of our problem. This, however, creates a significant, artificial non-linearity near -𝜋 and 𝜋 that our control system must learn. In practice we can learn this at the cost of training time and entropic capacity out of our model, but there's no need. We can instead recognize the mirror symmetry along the vertical axis and map the state into 0 < 𝜃 ≤ 𝜋 with an appropriate reflection of 𝜃 and 𝜔 whenever the pendulum would go below 0 or above 𝜋. Solving the problem in this regime is strictly equivalent to solving the problem in the wider regime.

## Rule-based Solution
A nearly optimal solution can be achieved with only two rules. If 𝜏<sub>0</sub> > 𝑚∗𝑔∗𝑙∗sin⁡𝜃 and 𝜃 ≤ 𝜋/2 then apply the torque counterclockwise (towards 𝜃 = 0). Otherwise, apply the torque along the direction of 𝜔. The first case corresponds to a partially inverted pendulum past the point where the gravitational torque fell below the applied torque so we can directly finish the inversion. The other case corresponds to dumping energy into the system by increasing 𝜔 so that the pendulum will swing up higher and higher until it reaches the first case. 

We implement this in C++ and find that it performs extremely well, both qualitatively and quantitatively. Given a pendulum starting near inversion, 𝜃 ≈ 0, the pendulum will remain within 𝜃 ≤ 0.4. Given a pendulum starting near rest, 𝜃 ≈ 𝜋, inversion will be achieved in the minimum possible number of time steps. That said, there is noticeable overshoot and for sufficiently high mass and low damping that overshoot would cause loss of inversion.  

