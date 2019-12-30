# inverted_pendulum_simulation_and_control
Investigating deep learning techniques for controlling a simulated inverted pendulum.

This is an investigation of machine learning for a dynamic control task with a focus on understanding how it works rather than optimizing the performance. The goal is to stabilize a simulated pendulum in an inverted state by applying either a positive or negative torque. 

This task has a number of attractive properties: 
* Extremely mathematically tractable, not only on the simulation side, but also for investigating the control system
* Continuous, bounded state space with finite and discrete action space
* Requires long term planning to successfully invert pendulum when starting from resting state
* Solvable nearly perfectly with two rules as either baseline or demonstration

