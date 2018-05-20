#pragma once
#include "Header.h"
#include "Other.h"

namespace NPSMLE
{
	// Simulate Vasicek process
	template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
	double * simulation_vasicek(double *process, double alpha, double beta, double sigma, double delta, int N, int step, double x0)
	{
		double dt = delta / step;
		process[0] = x0;

		GeneratorType generator;
		generator.seed(GeneratorSeed()());
		std::normal_distribution<double> distribution(0.0, 1.0);

		for (int i = 1; i != N; ++i)
		{
			process[i] = process[i - 1];
			for (int j = 0; j != step; ++j)
			{
				process[i] += beta * (alpha - process[i]) * dt + sigma * sqrt(dt) * distribution(generator);
			}
		}

		return process;
	}

	// Analytical LL for Vasicek process for optimization
	double analytical_ll_vasicek_optim(const std::vector<double>& x, std::vector<double>& grad, void* data)
	{
		double analytical_ll = 0.0;

		double beta = x[0];
		double alpha = x[1];
		double sigma = x[2];
		WrapperAnalytical * object = static_cast<WrapperAnalytical*>(data);
		int N_obs = object->N_obs;
		double delta = object->delta;
		double * process = object->process;
		
		double denominator; // = sigma * exp(-beta * delta) * sqrt(0.5 * (exp(2 * beta * delta) - 1) / beta);
		double input; // = (process[0] - object->x0 * exp(-beta * delta) - alpha * (1 - exp(-beta * delta))) / denominator;
		double normal_pdf; // = exp(-0.5 * input * input) / sqrt(M_PI * 2.0);

		// analytical_ll += ::log(normal_pdf / denominator);

		for (int i = 1; i != N_obs; ++i)
		{
			denominator = sigma * exp(-beta * delta) * sqrt(0.5 * (exp(2 * beta * delta) - 1) / beta);
			input = (process[i] - process[i - 1] * exp(-beta * delta) - alpha * (1 - exp(-beta * delta))) / denominator;
			normal_pdf = exp(-0.5 * input * input) / sqrt(M_PI * 2.0);

			analytical_ll += ::log(normal_pdf / denominator);
		}

#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (analytical_ll == -INFINITY || !std::isnormal(analytical_ll))
		{
			return max_double;
		}
#endif

		return -analytical_ll;
	}

	// Simulated LL for Vasicek process for optimization
	template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
	double simulated_ll_vasicek_optim(const std::vector<double>& x, std::vector<double>& grad, void* data)
	{
		double beta = x[0];
		double alpha = x[1];
		double sigma = x[2];
		WrapperSimulated<GeneratorType, GeneratorSeed> * obj = static_cast<WrapperSimulated<GeneratorType, GeneratorSeed>*>(data);
		int L = obj->N_obs;
		int N = obj->N_sim;
		int M = obj->step;
		double delta = obj->delta;
		double * y = obj->process;
		double * simulated_y = obj->simulated_process;
		double * wiener = obj->random_buffer;

		double ll = 0.0;
		double kernel_sum = 0.0;

		// PRE-COMPUTATIONS
		static double sqrt_pi = sqrt(2.0 * M_PI);
		double dt = delta / M;
		double sigma_sqrt_dt = sigma * sqrt(dt);
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
					simulated_y[j] += beta * (alpha - simulated_y[j]) * dt + sigma_sqrt_dt * wiener[j * M + k]; // Process structure
				}
			}

			h = h_frac * st_dev(simulated_y, N); // Optimal kernel bandwidth computation

			for (int j = 0; j != N; ++j)
			{
				kernel_sum += exp((-(simulated_y[j] - y[i]) * (simulated_y[j] - y[i])) / (2.0 * h * h)) / (h * sqrt_pi);
			}

			ll += ::log(kernel_sum / N);

			kernel_sum = 0.0;

#ifdef INFINITY_CHECK
			// Speed up in cases of infinity
			if (ll == -INFINITY || !std::isnormal(ll))
			{
				return max_double;
			}
#endif
		}

		return -ll;
	}
}
