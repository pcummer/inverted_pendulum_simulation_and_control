# inverted_pendulum_simulation_and_control
## Overview
An animation of our final deep learning control system
![Final Model Animation](https://github.com/pcummer/inverted_pendulum_simulation_and_control/blob/master/pendulum_rectified.gif)

This is an investigation of machine learning for a dynamic control task with a focus on understanding how it works rather than optimizing the performance. The goal is to stabilize a simulated pendulum in an inverted state by applying either a positive or negative torque as decided by a neural network trained via reinforcement learning. 

This task has a number of attractive properties: 
* Extremely mathematically tractable, not only on the simulation side, but also for investigating the control system
* Continuous, bounded state space with finite and discrete action space
* Requires long term planning to successfully invert pendulum when starting from resting state
* Solvable nearly perfectly with two rules as either baseline or demonstration

## Background
### Setup
Our pendulum is simulated in C++ and controlled via a Tensorflow neural network hosted on a web API in Python. At every time step the state of the pendulum is updated according to its equation of motion under an applied torque in C++. At every tenth time step an http request is made to the neural net, sending the state and recieving a direction for the applied torque, clockwise or counterclockwise. Repeating the action for ten time steps improves computational efficiency, samples a more diverse range of states, and emphasizes the impact of the choice. Ideally it would all sit in C++, but the Tensorflow bindings are painful and the http requests are fast compared to even the forward pass of a moderately sized neural net. 

![Setup Diagram](https://github.com/pcummer/inverted_pendulum_simulation_and_control/blob/master/Setup%20diagram.PNG)

For interesting physics, we require that our physical parameters fall within certain ranges. The applied torque must be less than a critical value, 𝜏<sub>0</sub> = 𝑚∗𝑔∗𝑙, so that control system cannot lift the pendulum from rest to inversion against gravity; instead, under this constraint it must pump energy into the system to achieve inversion. We'd also benefit from small damping proportial to 𝜔, the angular velocity, to avoid our simulation breaking down under sufficiently high 𝜔, though we of course need to allow the system to accumulate sufficient kinetic energy. Roughly this mean we want 𝜏<sub>damping</sub> < 𝜏<sub>0</sub> when the kinetic energy is equal to the potential energy at inversion, 𝑚∗𝑔∗2𝑙=1/2 𝑚∗(𝜔∗𝑙)<sup>2</sup>. 

We're otherwise free to choose our parameters and retain interesting physics, but we should also think how these paramaters affect our control system and its learning. In general the problem will be much harder to learn for larger moments of inertia as this effectively lengthens the time scale over which our control system must plan. Put another way, for larger moments, each action has a smaller impact on the trajectory of the system and therefore many such actions must be coordinated to effect a significant change in trajectory. Similarly, smaller applied torques require longer horizons.

We have one more trick here to simplify the learning the process. The obvious state representation is -𝜋 < 𝜃 ≤ 𝜋 due to the cyclic nature of our problem. This, however, creates a significant, artificial non-linearity near -𝜋 and 𝜋 that our control system must learn. In practice we can learn this at the cost of training time and entropic capacity out of our model, but there's no need. We can instead recognize the mirror symmetry along the vertical axis and map the state into 0 < 𝜃 ≤ 𝜋 with an appropriate reflection of 𝜃 and 𝜔 whenever the pendulum would go below 0 or above 𝜋. Solving the problem in this regime is strictly equivalent to solving the problem in the wider regime.

### Rule-based Solution
A nearly optimal solution can be achieved with only two rules. If 𝜏<sub>0</sub> > 𝑚∗𝑔∗𝑙∗sin⁡𝜃 and 𝜃 ≤ 𝜋/2 then apply the torque counterclockwise (towards 𝜃 = 0). Otherwise, apply the torque along the direction of 𝜔. The first case corresponds to a partially inverted pendulum past the point where the gravitational torque fell below the applied torque so we can directly finish the inversion. The other case corresponds to dumping energy into the system by increasing 𝜔 so that the pendulum will swing up higher and higher until it reaches the first case. 

We implement this in C++ and find that it performs extremely well, both qualitatively and quantitatively. Given a pendulum starting near inversion, 𝜃 ≈ 0, the pendulum will remain within 𝜃 ≤ 0.4 as shown below (note that theta is not mirrored for clarity in the figure). Given a pendulum starting near rest, 𝜃 ≈ 𝜋, inversion will be achieved in the minimum possible number of time steps. That said, there is noticeable overshoot and for sufficiently high mass and low damping that overshoot would cause loss of inversion. 

![Rule-based Evolution](https://github.com/pcummer/inverted_pendulum_simulation_and_control/blob/master/rule_start_0_1.png)

## Reinforcement Learning
### Deep Q-learning
We will use deep Q-learning as a broadly effective and relatively explainable approach. For the unfamiliar, deep Q-learning uses a neural network to estimate the expected future rewards for each action in a given state, then picking the action with the best future rewards. This is learned iteratively via the relationship 𝑄(𝑆<sub>𝑛</sub>, 𝑎<sub>𝑛</sub>)=𝑟(𝑆<sub>𝑛</sub>, 𝑎<sub>𝑛</sub>)+𝑑∗𝑄(𝑆<sub>(𝑛+1)</sub>, 𝑎𝑟𝑔𝑚𝑎𝑥<sub>a</sub>(𝑄(𝑆<sub>𝑛+1</sub>,𝑎))) where 𝑆 is the state, 𝑎 is the action, 𝑟 is the reward function, and 𝑑 is the discount that devalues future rewards. Quite simply, the new Q-value for a state-action pair is the reward that it generates plus the highest Q-value possible in the resulting state times some discount factor. 

From our simulation, we record the initial state, action taken, reward garnered, and resulting state in order to calculate the above relationship, using our neural net to predict 𝑄(𝑆<sub>(𝑛+1)</sub>, 𝑎𝑟𝑔𝑚𝑎𝑥<sub>a</sub>(𝑄(𝑆<sub>𝑛+1</sub>,𝑎))). We then perform an update on our network weights towards predicting this calculated Q-value for the initial state and action taken. In practice we actually randomly sample out of memory buffer of such state-action-Q-value records since records close in time have insufficient variation leading to biased learning. 

One might be rightly suspicious at how we use our networks estimates in order to update our network. There is an established tendency for this approach to optimistically over-estimate Q-values; however, as long as the learning rate is kept sufficiently low this only rarely diverges. It is an unfortunate case where one must simply be watchful and adjust hyperparameters as necessary for effective learning.

Another major difficulty is getting the network established to the point where it consistently receives a reward signal. Even with a dense reward signal, naively using the network's best guesses to explore the environment tends to perform poorly: the network will get trapped in a relatively poor, but not terrible, behavior and never receive a signal to incentivize other actions. A common technique is to use an 𝜀-greedy policy that generally follows the network's recommendations, but takes a random action with probability 𝜀, usually exponentially decaying 𝜀 over the course of learning. 

Here we extend this concept by instead applying zero-centered additive gaussian noise with standard deviation 𝜀 to the network predictions. This similarly accomplishes the goal of encouraging exploration, but biases the exploration to moments of uncertainty or equanamity and avoids exploring avenues that are known to be catastrophically poor. We initialize 𝜀 to be an order of magnitude below the theoretical maximum Q-value and decay it to zero over the course of learning. With our limited compute power, this targeted reduction in the search space is hugely beneficial. We also initialize our network weights with a round of training on data generated by the rule-based control system to speed up the process.

### Down the garden path
In keeping with the goal of learning an effective policy within the constraints of our limited resources (a single laptop CPU), we begin with a dense reward function: 𝑟 = cos⁡𝜃. This is going to turn out to be a very bad idea, but the way in which it fails is quite interesting. At first blush, this reward function appears as if it will encourage full inversion via teaching the network to lift the pendulum higher and higher. Arguably, the shallow gradient of cos⁡𝜃 near 0 will make the control system a bit lazy in terms of achieving perfect inversion, and we should therefore use cos⁡<sup>n</sup>𝜃 for some value of n between 1 and 10, but that's not our major issue.

Upon implementing this setup, we find that network sits the pendulum at sin⁡𝜃 = 𝜏<sub>0</sub> / 𝑚∗𝑔∗𝑙 (the solution near 𝜋). This is the result of continually forcing the pendulum in one direction until it balances against the gravitation torque. Upon further interrogation, we can see that this is the optimal action under our reward function in the short term since it avoids the highly negative rewards near 𝜋. The obvious answer is to just raise 𝑑 towards 1 so that the model optimizes on a longer time scale. Why is this going to fail?

