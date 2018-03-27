#include "Header.h"


template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
class WrapperSimulatedJointLatentVolatility
{
public:
	WrapperSimulatedJointLatentVolatility(double * price, double * volatility, double dt, 
		int N_obs, int N_sim, int M_sim, double v0) : N_obs(N_obs), N_sim(N_sim), 
		M_sim(M_sim), dt(dt), price(price), volatility(volatility), v0(v0)
	{
		int random_buffer_size = N_sim * M_sim;

		// Allocate buffers to hold values for Wiener processes
		wiener_price = (double*)malloc(random_buffer_size * sizeof(double));
		wiener_volatility = (double*)malloc(random_buffer_size * sizeof(double));

		// Allocate buffers to hold simulated values for individual processses
		simulated_price = (double*)malloc(N_sim * sizeof(double));
		simulated_volatility = (double*)malloc(N_sim * sizeof(double));

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

	~WrapperSimulatedJointLatentVolatility()
	{
		free(wiener_price);
		free(wiener_volatility);
		free(simulated_price);
		free(simulated_volatility);
		free(random_buffer_price);
		free(random_buffer_volatility);
	}

	int N_obs, N_sim, M_sim;
	double dt, v0;
	double *price, *volatility; // original processes
	double *simulated_price, *simulated_volatility; // buffers to hold simulted processes 
	double *wiener_price, *wiener_volatility; // buffers to hold Wiener processes
	double *random_buffer_price, *random_buffer_volatility; // buffers to hold random draws
};


template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
void simulate_stoch_vol_process(double * price, double * volatility, const std::vector<double>& 
	parameters, double dt, int N_obs, int M_obs, double p0, double v0)
{
	// Unpack simulation parameters
	double mu_p = parameters[0];
	double delta_p = parameters[1];
	double mu_v = parameters[2];
	double delta_v = parameters[3];
	double rho_pv = parameters[4];

	// Pre-allocate variables
	static double mp, sp, mv, sv, ms;
	double delta = dt / M_obs;
	double sqrt_delta = sqrt(delta);

	// Allocate space for correlated Wiener processes
	static double W_p, W_v, W_s;

	// Initialize random engines
	GeneratorType generator;
	generator.seed(GeneratorSeed()());
	std::normal_distribution<double> distribution(0.0, 1.0);

	// Initialize first values
	volatility[0] = v0;
	price[0] = p0;

	// Process generation
	for (int i = 1; i != N_obs; ++i)
	{
		volatility[i] = volatility[i - 1];
		price[i] = price[i - 1];

		for (int j = 0; j != M_obs; ++j)
		{
			W_s = distribution(generator);
			W_v = distribution(generator);
			W_p = sqrt(1.0 - rho_pv * rho_pv) * distribution(generator) + rho_pv * W_v;

			mp = (mu_p - price[i]);
			sp = delta_p * sqrt(abs(volatility[i]));
			price[i] += mp * delta + W_p * sp * sqrt_delta;

			mv = (mu_v - volatility[i]);
			sv = delta_v * sqrt(abs(volatility[i]));
			volatility[i] += mv * delta + W_v * sv * sqrt_delta;
		}
	}
}


template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
double simulated_ll_joint_latent(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	// Unwraping parameter
	double mu_p = x[0];
	double delta_p = x[1];
	double mu_v = x[2];
	double delta_v = x[3];
	double rho_pv = x[4];

	// Unwraping data
	WrapperSimulatedJointLatentVolatility<GeneratorType, GeneratorSeed> *wrapper =
		static_cast<WrapperSimulatedJointLatentVolatility<GeneratorType, GeneratorSeed>*>(data);
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
	const double sqrt_pi = sqrt(2.0 * M_PI);
	const int dimy = 1;
	const double undersmooth = 0.5;
	const double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N_sim, -(1.0 + undersmooth) / (dimy + 4.0));
	double h_price, h_volatility;
	double mp, mv, sv, sp;
	double ll = 0.0;
	double kernel_sum_price = 0.0, kernel_sum_volatility = 0.0, kernel_sum = 0.0;
	double kernel_sum_price_lag = 1.0, kernel_sum_numerator = 0.0, kernel_sum_denominator = 0.0;
	const double delta = dt / M_sim;
	const double sqrt_delta = sqrt(delta);

	// Fill correlated Wiener process buffers
	double * W_v = random_buffer_volatility;
	double * W_p = wrapper->wiener_price;

	// Initialize random buffers
	int random_buffer_size = N_sim * M_sim;
	for(int i = 0; i != random_buffer_size; ++i)
	{
		W_p[i] = sqrt(1.0 - rho_pv * rho_pv) * random_buffer_price[i] + rho_pv * W_v[i];
	}
	/*
	// Main log-likelihood computation
	for (int i = 1; i != N_obs; ++i)
	{
		for (int j = 0; j != N_sim; ++j)
		{
			simulated_price[j] = price[i - 1];
			simulated_volatility[j] = volatility[i - 1];

			for (int k = 0; k != M_sim; ++k)
			{
				mp = (mu_p - simulated_price[j]);
				sp = sqrt(abs(simulated_volatility[j]));
				simulated_price[j] += mp * delta + W_p[j * M_sim + k] * sp * delta_p * sqrt_delta;

				mv = (mu_v - simulated_volatility[j]);
				sv = sqrt(abs(simulated_volatility[j]));
				simulated_volatility[j] += mv * delta + W_v[j * M_sim + k] * sv * delta_v * sqrt_delta;
			}
		}

		// Optimal kernel bandwidth computation
		h_price = h_frac * st_dev(simulated_price, N_sim);
		h_volatility = h_frac * st_dev(simulated_volatility, N_sim);

		for (int j = 0; j != N_sim; ++j)
		{
			kernel_sum_price = exp((-(simulated_price[j] - price[i]) * (simulated_price[j] - price[i])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			kernel_sum_volatility = exp((-(simulated_volatility[j] - volatility[i]) * (simulated_volatility[j] - volatility[i])) / (2.0 * h_volatility * h_volatility)) / (h_volatility * sqrt_pi);
			kernel_sum += kernel_sum_price * kernel_sum_volatility;
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
	*/
	
	// Main log-likelihood computation
	simulated_price[0] = price[0];
	simulated_volatility[0] = v0;

	for (int i = 1; i != N_sim; ++i)
	{
		simulated_price[i] = simulated_price[i - 1];
		simulated_volatility[i] = simulated_volatility[i - 1];

		for (int j = 0; j != M_sim; ++j)
		{
			mp = (mu_p - simulated_price[i]);
			sp = sqrt(abs(simulated_volatility[i]));
			simulated_price[i] += mp * delta + W_p[i * M_sim + j] * sp * delta_p * sqrt_delta;

			mv = (mu_v - simulated_volatility[i]);
			simulated_volatility[i] += mv * delta + W_v[i * M_sim + j] * sp * delta_v * sqrt_delta;
		}
	}

	// Optimal kernel bandwidth computation
	h_price = h_frac * st_dev(simulated_price, N_sim);

	// LIL estimator
	const int limit = 1;
	for (int i = limit; i != N_obs; ++i)
	{
		for (int j = limit; j != N_sim; ++j)
		{
			kernel_sum_price = exp((-(simulated_price[j] - price[i]) * (simulated_price[j] - price[i])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			for (int k = 1; k != limit + 1; ++k)
			{
				kernel_sum_price_lag *= exp((-(simulated_price[j - k] - price[i - k]) * (simulated_price[j - k] - price[i - k])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			}
			kernel_sum_denominator += kernel_sum_price_lag;
			kernel_sum_numerator += kernel_sum_price * kernel_sum_price_lag;

			kernel_sum_price_lag = 1.0;
		}

		ll += log(kernel_sum_numerator / kernel_sum_denominator);

#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (ll == -INFINITY || isnan(ll))
		{			
			return INFINITY;
		}
#endif
		kernel_sum_numerator = 0.0;
		kernel_sum_denominator = 0.0;
	}
	
	return -ll;
}
