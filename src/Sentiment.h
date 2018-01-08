#pragma once
#include "Header.h"
#include "Other.h"


int random_buffer_size = 100000;
double * random_buffer;
int * random_buffer_index;

// Simulate Vasicek process
double * simulation_vasicek_full(double alpha, double beta, double sigma, int N, double x0, double delta, int step)
{
	double dt = delta / step;
	double * process = (double*)malloc(N * sizeof(double));
	process[0] = x0;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);

	for (int i = 1; i != N; ++i)
	{
		process[i] = process[i - 1];
		for (int j = 0; j != step; ++j)
		{
			process[i] += beta * (alpha - process[i]) * dt + sigma * pow(dt, 0.5) * distribution(generator);
		}
	}

	return process;
}

// Simulate CIR process
double * simulation_cir_full(double alpha, double beta, double sigma, int N, double x0, double delta, int step)
{
	double dt = delta / step;
	double * process = (double*)malloc(N * sizeof(double));
	process[0] = x0;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);

	for (int i = 1; i != N; ++i)
	{
		process[i] = process[i - 1];
		for (int j = 0; j != step; ++j)
		{
			process[i] += beta * (alpha - process[i]) * dt + sigma * pow(dt, 0.5) * sqrt(abs(process[i])) * distribution(generator);
		}
	}

	return process;
}

// Analytical LL for Vasicek process for optimization
double analytical_ll_vasicek_optim(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	double analytical_ll = 0.0;

	double alpha = x[0];
	double beta = x[1];
	double sigma = x[2];
	WrapperAnalytical * object = static_cast<WrapperAnalytical*>(data);
	int N_obs = object->N_obs;
	double delta = object->delta;
	double * process = object->process;

	double denominator = sigma * exp(-beta * delta) * pow(0.5 * (exp(2 * beta * delta) - 1) / beta, 0.5);
	double input = (process[0] - object->x0 * exp(-beta * delta) - alpha * (1 - exp(-beta * delta))) / denominator;
	double normal_pdf = exp(-0.5 * input * input) / pow(M_PI * 2.0, 0.5);

	analytical_ll += log(normal_pdf / denominator);

	for (int i = 1; i != N_obs; ++i)
	{
		denominator = sigma * exp(-beta * delta) * pow(0.5 * (exp(2 * beta * delta) - 1) / beta, 0.5);
		input = (process[i] - process[i - 1] * exp(-beta * delta) - alpha * (1 - exp(-beta * delta))) / denominator;
		normal_pdf = exp(-0.5 * input * input) / pow(M_PI * 2.0, 0.5);

		analytical_ll += log(normal_pdf / denominator);
	}

	return -analytical_ll;
}

// Simulated LL for Vasicek process for optimization
double simulated_ll_vasicek_optim(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	double alpha = x[0];
	double beta = x[1];
	double sigma = x[2];
	WrapperSimulated * obj = static_cast<WrapperSimulated*>(data);
	int L = obj->N_obs;
	int N = obj->N_sim;
	int M = obj->step;
	double delta = obj->delta;
	double * y = obj->process;
	double * simulated_y = obj->simulated_process;

	double ll = 0.0;
	double kernel_sum = 0.0;

	int random_index = 0; // index for indexing random numbers from global buffer

	// PRE-COMPUTATIONS
	static double sqrt_pi = pow(2.0 * M_PI, 0.5);
	double dt = delta / M;
	double sigma_sqrt_dt = sigma * pow(dt, 0.5);
	int dimy = 1;
	double undersmooth = 0.5;
	double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N, -(1.0 + undersmooth) / (dimy + 4.0)); // only 1D
	double h;

	for (int i = 1; i != L; ++i)
	{
		for (int j = 0; j != N; ++j)
		{
			simulated_y[j] = y[i - 1];
			for (int k = 0; k != M; ++k)
			{
				simulated_y[j] += beta * (alpha - simulated_y[j]) * dt + sigma_sqrt_dt * random_buffer[random_buffer_index[random_index]]; // Process structure
				random_index = (random_index + 1) % random_buffer_size;
			}
		}

		h = h_frac * st_dev(simulated_y, N); // Optimal kernel bandwidth computation

		for (int j = 0; j != N; ++j)
		{
			kernel_sum += exp((-(simulated_y[j] - y[i]) * (simulated_y[j] - y[i])) / (2.0 * h * h)) / (h * sqrt_pi);
		}

		ll += log(kernel_sum / N);

		kernel_sum = 0.0;
	}

	return -ll;
}

// Simulated LL for CIR process for optimization
double simulated_ll_cir_optim(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	WrapperSimulated * obj = static_cast<WrapperSimulated*>(data);
	int L = obj->N_obs;
	int N = obj->N_sim;
	int M = obj->step;
	double delta = obj->delta;
	double * y = obj->process;
	double * simulated_y = obj->simulated_process;
	double alpha = x[0];
	double beta = x[1];
	double sigma = x[2];

	double ll = 0.0;
	double kernel_sum = 0.0;

	int random_index = 0; // index for indexing random numbers from global buffer

	// PRE-COMPUTATIONS
	static double sqrt_pi = pow(2.0 * M_PI, 0.5);
	double dt = delta / M;
	double sigma_sqrt_dt = sigma * pow(dt, 0.5);
	double alpha_beta_dt = alpha * beta * dt;
	double a, b, c;
	int dimy = 1;
	double undersmooth = 0.5;
	double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N, -(1.0 + undersmooth) / (dimy + 4.0));
	double h;

	for (int i = 1; i != L; ++i)
	{
		for (int j = 0; j != N; ++j)
		{
			simulated_y[j] = y[i - 1];
			for (int k = 0; k != M; ++k)
			{
				a = -beta * simulated_y[j] * dt;
				b = sqrt(abs(simulated_y[j]));
				c = random_buffer[random_buffer_index[random_index]];
				random_index = (random_index + 1) % random_buffer_size;
				simulated_y[j] += alpha_beta_dt + a + sigma_sqrt_dt * b * c; // Process structure
			}
		}

		h = h_frac * st_dev(simulated_y, N); // Optimal kernel bandwidth computation

		for (int j = 0; j != N; ++j)
		{
			kernel_sum += exp((-(simulated_y[j] - y[i]) * (simulated_y[j] - y[i])) / (2.0 * h * h)) / (h * sqrt_pi);
		}

		ll += log(kernel_sum / N);

		kernel_sum = 0.0;
	}

	return -ll;
}
