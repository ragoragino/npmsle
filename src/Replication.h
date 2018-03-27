#pragma once
#include "Header.h"

template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
class WrapperSimulatedReplication
{
public:
	WrapperSimulatedReplication(double * price, double * volatility, double dt,
		int N_obs, int N_sim, int M_sim, double v0) : N_obs(N_obs), N_sim(N_sim),
		M_sim(M_sim), dt(dt), price(price), volatility(volatility), v0(v0)
	{
		int random_buffer_size = N_sim * M_sim;

		// Allocate buffers to hold simulated values for individual processses
		simulated_price = (double*)malloc(N_sim * sizeof(double));
		simulated_volatility = (double*)malloc(N_sim * sizeof(double));

		wiener_buffer_price = (double*)malloc(random_buffer_size * sizeof(double));
		wiener_buffer_volatility = (double*)malloc(random_buffer_size * sizeof(double));

		// Allocate and initialize buffers to hold random values
		GeneratorType generator;
		generator.seed(GeneratorSeed()());
		std::normal_distribution<double> distribution(0.0, 1.0);
	
		random_buffer_price = (double*)malloc(random_buffer_size * sizeof(double));
		random_buffer_volatility = (double*)malloc(random_buffer_size * sizeof(double));
		for (int i = 0; i != random_buffer_size; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
		}
	}

	void update_random_buffer()
	{
		GeneratorType generator;
		generator.seed(GeneratorSeed()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		for (int i = 0; i != N_sim * M_sim; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
		}
	}

	~WrapperSimulatedReplication()
	{
		free(wiener_buffer_price);
		free(wiener_buffer_volatility);
		free(simulated_price);
		free(simulated_volatility);
		free(random_buffer_price);
		free(random_buffer_volatility);
	}

	int N_obs, N_sim, M_sim;
	double dt, v0;
	double *price, *volatility; // original processes
	double * wiener_buffer_volatility, *wiener_buffer_price;
	double *simulated_price, *simulated_volatility; // buffers to hold simulted processes 
	double *random_buffer_price, *random_buffer_volatility; // buffers to hold random draws
};


template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
void simulate_replication(double * price, double * volatility, const std::vector<double>&
	parameters, double dt, int N_obs, int M_obs, double p0, double v0)
{
	// Unpack simulation parameters
	double mu = parameters[0];
	double alpha_0 = parameters[1];
	double alpha_1 = parameters[2];
	double alpha_2 = parameters[3];
	double rho = parameters[4];

	// Pre-allocate variables
	static double mp, sp, mv, sv;
	double delta = dt / (double)M_obs;

	// Allocate space for correlated Wiener processes
	double W_p, W_v;

	// Initialize random engines for jumps
	GeneratorType generator;
	generator.seed(GeneratorSeed()());
	std::normal_distribution<double> distribution_wiener(0.0, 1.0);

	// Initialize first values and jumps
	volatility[0] = v0;
	price[0] = p0;

	for (int i = 1; i != N_obs; ++i)
	{
		volatility[i] = volatility[i - 1];
		price[i] = price[i - 1];

		for (int j = 0; j != M_obs; ++j)
		{
			W_v = distribution_wiener(generator);
			W_p = sqrt(1.0 - rho * rho) * distribution_wiener(generator) + rho * W_v;

			price[i] += (mu - exp(volatility[i]) * 0.5) * delta + exp(volatility[i] * 0.5) * W_p * sqrt(delta);
			volatility[i] += (alpha_0 - alpha_1 * volatility[i]) * delta + alpha_2 * W_v * sqrt(delta);

		}
	}
}


template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
double simulated_replication(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	// Unwraping parameter
	double mu = x[0];
	double alpha_0 = x[1];
	double alpha_1 = x[2];
	double alpha_2 = x[3];
	double rho = x[4];

	// Unwraping data
	WrapperSimulatedReplication<GeneratorType, GeneratorSeed> *wrapper =
		static_cast<WrapperSimulatedReplication<GeneratorType, GeneratorSeed>*>(data);
	double *price = wrapper->price;
	double *volatility = wrapper->volatility;
	double *simulated_price = wrapper->simulated_price;
	double *simulated_volatility = wrapper->simulated_volatility;
	double *random_buffer_price = wrapper->random_buffer_price;
	double *random_buffer_volatility = wrapper->random_buffer_volatility;
	int N_obs = wrapper->N_obs;
	int N_sim = wrapper->N_sim;
	int M_sim = wrapper->M_sim;
	double dt = wrapper->dt;
	double v0 = wrapper->v0;

	// Pre-allocating variables
	double ll = 0.0;
	double kernel_sum_price = 0.0, kernel_sum_volatility = 0.0, kernel_sum = 0.0;
	const double sqrt_pi = sqrt(2.0 * M_PI);
	const int dimy = 1;
	const double undersmooth = 0.5;
	const double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N_obs, -(1.0 + undersmooth) / (dimy + 4.0));
	// const double h_frac = 0.00058;
	double h_price, h_volatility;
	const double delta = dt / M_sim;
	const double sqrt_delta = sqrt(delta);

	// Fill in correlated random buffers
	int random_buffer_length = N_sim * M_sim;
	double *W_v = wrapper->random_buffer_volatility;
	double *W_p = wrapper->wiener_buffer_price;
	for (int i = 0; i != random_buffer_length; ++i)
	{
		W_p[i] = sqrt(1.0 - rho * rho) * random_buffer_price[i] + rho * W_v[i];
	}

	// Main log-likelihood computation
	for (int i = 1; i != N_obs; ++i)
	{
		for (int j = 0; j != N_sim; ++j)
		{
			simulated_price[j] = price[i - 1];
			simulated_volatility[j] = volatility[i - 1];

			for (int k = 0; k != M_sim; ++k)
			{
				simulated_price[j] += (mu - exp(simulated_volatility[j]) * 0.5) * delta + exp(simulated_volatility[j] * 0.5) * W_p[j * M_sim + k] * sqrt_delta;
				simulated_volatility[j] += (alpha_0 - alpha_1 * simulated_volatility[j]) * delta + alpha_2 * W_v[j * M_sim + k] * sqrt_delta;
			}
		}

		// Optimal kernel bandwidth computation
		h_price = h_frac * st_dev(simulated_price, N_sim);
		h_volatility = h_frac * st_dev(simulated_volatility, N_sim);

		for (int j = 0; j != N_sim; ++j)
		{
			kernel_sum_price = exp((-(simulated_price[j] - price[i]) * (simulated_price[j] - price[i])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			kernel_sum_volatility = exp((-(simulated_volatility[j] - volatility[i]) * (simulated_volatility[j] - volatility[i])) / (2.0 * h_volatility * h_volatility)) / (h_volatility * sqrt_pi);
			kernel_sum += kernel_sum_volatility * kernel_sum_price;
		}
		
		ll += log(kernel_sum / N_sim);

		kernel_sum = 0.0;

#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (ll == -INFINITY || isnan(ll))
		{
			return INFINITY;
		}
#endif
	}

	return -ll;
}


