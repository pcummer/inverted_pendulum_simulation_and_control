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
Our pendulum is simulated in C++ and controlled via a Tensorflow neural network hosted on a web API in Python. At every time step the state of the pendulum is updated according to its equation of motion under an applied torque in C++. At every tenth time step an http request is made to the neural net sending the state and recieving a direction for the applied torque, clockwise or counterclockwise. Ideally it would all sit in C++, but the Tensorflow bindings are painful and the http requests are fast compared to even the forward pass of a moderately sized neural net. 

![Setup Diagram](marhttps://github.com/pcummer/inverted_pendulum_simulation_and_control/blob/master/Setup%20diagram.PNG)

For interesting physics, we require that our physical parameters fall within certain ranges. The applied torque must be less than a critical value, 𝜏_0 = 𝑚∗𝑔∗𝑙, so that control system cannot lift the pendulum from rest to inversion against gravity; instead, under this constraint it must pump energy into the system to achieve inversion. We'd also benefit from small damping proportial to 𝜔, the angular velocity, to avoid our simulation breaking down under sufficiently high 𝜔, though we of course need to allow the system to accumulate sufficient kinetic energy. Roughly this mean we want 𝜏_damping < 𝜏_0 when the kinetic energy is equal to the potential energy at inversion, 𝑚∗𝑔∗2𝑙=1/2 𝑚∗〖(𝜔∗𝑙)〗^2. 

We're otherwise free to choose our parameters and retain interesting physics, but we should also think how these paramaters affect our control system and its learning. In general the problem will be much harder to learn for larger moments of inertia as this effectively lengthens the time scale over which our control system must plan. Put another way, for larger moments, each action has a smaller impact on the trajectory of the system and therefore many such actions must be coordinated to effect a significant change in trajectory. Similarly, smaller applied torques require longer horizons.

