// cpp_console.cpp : This file contains the 'main' function. Program execution begins and ends there.
// The simulation is handled here in C++ with a tensorflow model hosted by a concurrent python script and queried with http requests

#include "pch.h"
#include <iostream>
#include <cpr/cpr.h>

int choose_action_rule_based(double theta, double omega, double max_applied_torque, double gravity, double mass, double length) {
	// very simple, yet effective rules for choosing the action
	// if theta is near 0 then apply a torque to return it to 0
	// if theta is far from 0 then apply a torque to increase the energy in the system so the pendulum will ultimately swing upward like pumping a swing
	// the critical point for near vs far is the applied torque being equal in strength to the gravitational torque and in the theta less than pi/2 regime

	int action;
	if (omega < 0) {
		action = 0;
	}
	else
	{
		action = 1;
	}

	if ((max_applied_torque > abs(gravity * mass * length * sin(theta))) and (abs(sin(theta / 2)) < 0.707)) {
		if (sin(theta) / abs(sin(theta)) > 0) {
			action = 0;
		}
		else
		{
			action = 1;
		}
	}
	return action;
}

double calculate_reward(double theta) {
	// reward is 1 for an inverted pendulum with theta less than 0.4, 0 otherwise. The theta/2 trick is left over from mapping theta into 0 to 2 pi
	double reward = 0;
	if (abs(sin(theta / 2)) < 0.2) {
		reward = 1;
	}
	return reward;
}

double calculate_reward_cos(double theta) {
	// dense reward function that is flat near 0 and pi
	double reward = 0;
	reward = cos(theta);
	return reward;
}

int choose_action_neural_net(double theta, double omega) {
	// Pass the state to the neural net hosted on the predict endpoint in python and recieve an action

	int action = 0;
	auto r = cpr::Post(cpr::Url("http://127.0.0.1:5000/predict"), cpr::Multipart{ {"theta", std::to_string(theta)}, {"omega", std::to_string(omega)} });
	action = std::stoi(r.text);
	return action;
}

int choose_action_neural_net_alternate_controls(double theta, double omega) {
	// This control system chooses to add or remove energy from the system i.e. apply a torque with or against the angular momentum
	// action = 1 corresponds to increasing energy, action = 0 corresponds to decreasing energy
	// this action is then translated into the other coordinates where action = 1 corresponds to a positive torque, and action = 0 to a negative torque
	// as above this passes the state to a predict endpoint and recieves an action

	int action = 0;
	auto r = cpr::Post(cpr::Url("http://127.0.0.1:5000/predict"), cpr::Multipart{ {"theta", std::to_string(theta)}, {"omega", std::to_string(omega)} });
	action = std::stoi(r.text);

	if (omega < 0) {
		if (action == 0) {
			action = 1;
		}
		else {
			action = 0;
		}
	}
	return action;
}


void save_history(double theta, double omega, double action, double theta_2, double omega_2, double reward) {
	// passes the last state, action, next state, and reward to model for training
	auto r = cpr::Post(cpr::Url("http://127.0.0.1:5000/save"), cpr::Multipart{ {"theta", std::to_string(theta)}, {"omega", std::to_string(omega)}, {"action", std::to_string(action)}, {"theta_2", std::to_string(theta_2)}, {"omega_2", std::to_string(omega_2)}, {"reward", std::to_string(reward)} });
}

void train_model(int epochs) {
	// posts to python endpoint telling model to train for a number of epochs, normally one
	auto r = cpr::Post(cpr::Url("http://127.0.0.1:5000/train"), cpr::Multipart{ {"epochs", std::to_string(epochs)}});
}

void debug_python() {
	// hits an extra endpoint for debugging the python
	auto r = cpr::Post(cpr::Url("http://127.0.0.1:5000/debug"), cpr::Multipart{ {"key", "test"} });
}

double theta_derivative(double omega, double time_step) {
	return omega * time_step;
}

double omega_derivative(double theta, double omega, double gravity, double length, double mass, double moment, double applied_torque, double drag, double time_step) {
	return time_step * (gravity * mass * length * sin(theta) + applied_torque) / moment - drag * omega;
}

void runge_kutta(double theta, double omega, double gravity, double length, double mass, double moment, double applied_torque, double drag, double time_step, double *theta_address, double *omega_address) {
	// fourth order runge kutta for coupled differential equations. just a nicer numerical approximation than simply multiplying the derivative by the timestep
	double k_omega_1 = omega_derivative(theta, omega, gravity, length, mass, moment, applied_torque, drag, time_step);
	double k_theta_1 = theta_derivative(omega, time_step);
	double k_omega_2 = omega_derivative(theta + 0.5 * k_theta_1, omega + 0.5 * k_omega_1, gravity, length, mass, moment, applied_torque, drag, time_step);
	double k_theta_2 = theta_derivative(omega + 0.5 * k_omega_1, time_step);
	double k_omega_3 = omega_derivative(theta + 0.5 * k_theta_2, omega + 0.5 * k_omega_2, gravity, length, mass, moment, applied_torque, drag, time_step);
	double k_theta_3 = theta_derivative(omega + 0.5 * k_omega_2, time_step);
	double k_omega_4 = omega_derivative(theta + 1.0 * k_theta_3, omega + 1.0 * k_omega_3, gravity, length, mass, moment, applied_torque, drag, time_step);
	double k_theta_4 = theta_derivative(omega + 1.0 * k_omega_3, time_step);
	*omega_address = omega + (k_omega_1 + 2 * k_omega_2 + 2 * k_omega_3 + k_omega_4) / 6;
	*theta_address = theta + (k_theta_1 + 2 * k_theta_2 + 2 * k_theta_3 + k_theta_4) / 6;
}


int main()
{
	// define variables
	double theta = 3.14 / 2;
	double omega = 0;
	double alpha = 0;
	double torque = 0;
	double applied_torque = 0;
	int action = 0;

	// define physical parameters, note that changing these will require retraining of the neural network
	double mass = 0.8;
	double length = 1;
	double drag = 0.01;
	double gravity = 5;
	double max_applied_torque = 3;
	double moment = mass * length * length;

	// define simulation parameters
	double time_step = 0.01;
	bool use_neural_net = false;
	bool train = false;
	int runs = 1;
	int iterations_per_run = 200;
	double state_action_state_reward_history[6][5000];

	// begin simulation
	std::cout << "Release pendulum!\n";

	// repeat simulation for defined number of runs
	for (int run = 0; run < runs; run++) {

		// begin each run at a random starting angle with no angular velocity
		// theta = (rand() % 100) * 3.14 / 100;
		omega = 0;

		// update simulation state for defined number of iterations
		for (int i = 0; i < iterations_per_run; i++) {

			// record initial state of system
			state_action_state_reward_history[0][i] = theta;
			state_action_state_reward_history[1][i] = omega;

			// determine direction of applied torque either via querying neural net or defined rules
			// action = 1 corresponds to a positive torque, action = 0 is a negative torque
			if (use_neural_net) {
				action = choose_action_neural_net_alternate_controls(theta, omega);
			}
			else
			{
				action = choose_action_rule_based(theta, omega, max_applied_torque, gravity, mass, length);
			}
			
			if (action == 1) {
				applied_torque = max_applied_torque;
			}
			else
			{
				applied_torque = -max_applied_torque;
			}

			// record the action taken as either increasing or decreasing energy of the system rather than direction of torque
			// see discussion for the effect of these controls
			if (omega < 0) {
				if (action == 0) {
					action = 1;
				}
				else
				{
					action = 0;
				}
			}
			state_action_state_reward_history[2][i] = action;

			// Advance time for 10 steps between decisions. We expect that the correct action should only change after many time steps so this gives us a 
			// large performance boost by avoiding the call to the NN every step. It should also encourage divergence between q values for the actions
			for (int j = 0; j < 10; j++) {
				// torque = gravity * mass * length * sin(theta) + applied_torque;
				// alpha = torque / moment;

				// omega = omega + alpha * time_step - drag * omega;
				// theta = theta + omega * time_step;
				runge_kutta(theta, omega, gravity, length, mass, moment, applied_torque, drag, time_step, &theta, &omega);
			}


			// We always map theta to between 0 and pi for convenience. Mapping between 0 and 2pi is obviously trivial, but with the mirror symmetry
			// along the y-axis we can reduce our space to 0 to pi with appropriate negative signs. Has no effect on the physics, but eases the burden
			// on our neural net by eliminating the significant non-linearity around 0 and 2pi
			if (theta > 3.14) {
				theta = theta - 2 * 3.14;
			}
			else
			{
				if (theta < -3.14) {
					theta = theta + 2 * 3.14;
				}
			}

			if (theta < 0) {
				theta = -theta;
				omega = -omega;
			}

			// record the state after our action has been taken along with the resulting reward
			state_action_state_reward_history[3][i] = theta;
			state_action_state_reward_history[4][i] = omega;
			state_action_state_reward_history[5][i] = calculate_reward(theta);

			// pass the record over to python for training/plotting/etc.
			save_history(state_action_state_reward_history[0][i], state_action_state_reward_history[1][i], state_action_state_reward_history[2][i], state_action_state_reward_history[3][i], state_action_state_reward_history[4][i], state_action_state_reward_history[5][i]);
			
			// perform a single update on the neural net
			if (train) {
				train_model(1);
			}

			// print state
			auto output = theta;
			std::cout << output;
			std::cout << "\n";
			std::cout << action;
			std::cout << "\n";
		}

		std::cout << "done";
		std::cout << "\n";
		// debug_python();
	}
}
