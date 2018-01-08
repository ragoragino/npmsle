#pragma once
#include "Header.h"

double st_dev(double * process, int length)
{
	double mean = 0.0;
	double standard_deviation = 0.0;
	for (int i = 0; i != length; ++i)
	{
		mean += process[i] / length;
	}

	for (int i = 0; i != length; ++i)
	{
		standard_deviation += (process[i] - mean) * (process[i] - mean) / (length - 1.0);
	}

	return pow(standard_deviation, 0.5);
}

template<typename GeneratorType = std::mt19937_64>
std::vector<double> perturb(const std::vector<double>& values, double scale = 0.1)
{
	std::vector<double> result{ values };
	GeneratorType generator;
	generator.seed(std::random_device()());
	std::normal_distribution<double> distribution(0.0, 1.0);

	for (int i = 0; i != result.size(); ++i)
	{
		result[i] += distribution(generator) * sqrt(abs(scale * result[i]));
	}	

	return result;
}


// Classes to wrap all the necessary data for loglikelihood routines (analytical or simulated) passing through NLOPT functions
class WrapperAnalytical
{
public:
	WrapperAnalytical(int N_obs, double delta, double x0, double * process) :
		N_obs(N_obs), delta(delta), x0(x0), process(process)
	{}

	int N_obs;
	double delta, x0;
	double * process;
};

template<typename GeneratorType = std::mt19937_64>
class WrapperSimulated
{
public:
	WrapperSimulated(int N_obs, int N_sim, int step, double delta,
		double * process) :
		N_obs(N_obs), N_sim(N_sim), step(step), delta(delta),
		process(process), simulated_process(simulated_process)
	{
		simulated_process = (double*)malloc(N_sim * sizeof(double));

		GeneratorType generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		int random_buffer_size = N_sim * step;
		random_buffer = (double*)malloc(random_buffer_size * sizeof(double));
		for (int i = 0; i != random_buffer_size; ++i)
		{
			random_buffer[i] = distribution(generator);
		}	
	}

	~WrapperSimulated()
	{
		free(random_buffer);
		free(simulated_process);
	}

	int N_obs, N_sim, step;
	double delta;
	double * process; // original proces
	double * simulated_process; // simulated process
	double * random_buffer; // buffer of random draws
};

template<typename GeneratorType = std::mt19937_64>
class WrapperSimulatedJoint
{
public:
	WrapperSimulatedJoint(double * price, double * volatility, double * sentiment, double dt, int N_obs, int N_sim,
		int M_sim) :
		N_obs(N_obs), N_sim(N_sim), M_sim(M_sim), dt(dt), price(price), volatility(volatility), sentiment(sentiment)
	{
		// Allocate buffers to hold values for Wiener processes
		wiener_price = (double*)malloc(N_sim * M_sim * sizeof(double));
		wiener_volatility = (double*)malloc(N_sim * M_sim * sizeof(double));
		wiener_sentiment = (double*)malloc(N_sim * M_sim * sizeof(double));

		// Allocate buffers to hold simulated values for individual processses
		simulated_price = (double*)malloc(N_sim * sizeof(double));
		simulated_volatility = (double*)malloc(N_sim * sizeof(double));
		simulated_sentiment = (double*)malloc(N_sim * sizeof(double));

		// Allocate and initialize buffers to hold random values
		GeneratorType generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		int random_buffer_size = N_sim * M_sim;
		random_buffer_price = (double*)malloc(random_buffer_size * sizeof(double));
		random_buffer_volatility = (double*)malloc(random_buffer_size * sizeof(double));
		random_buffer_sentiment = (double*)malloc(random_buffer_size * sizeof(double));
		for (int i = 0; i != random_buffer_size; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	void update_random_buffer()
	{
		GeneratorType generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		for (int i = 0; i != N_sim * M_sim; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	~WrapperSimulatedJoint()
	{
		free(wiener_price);
		free(wiener_volatility);
		free(wiener_sentiment);
		free(simulated_price);
		free(simulated_volatility);
		free(simulated_sentiment);
		free(random_buffer_price);
		free(random_buffer_volatility);
		free(random_buffer_sentiment);
	}

	int N_obs, N_sim, M_sim;
	double dt;
	double *price, *volatility, *sentiment; // original processes
	double *simulated_price, *simulated_volatility, *simulated_sentiment; // buffers to hold simulted processes 
	double *wiener_price, *wiener_volatility, *wiener_sentiment; // buffers to hold Wiener processes
	double *random_buffer_price, *random_buffer_volatility, *random_buffer_sentiment; // buffers to hold random draws
};


template<typename GeneratorType = std::mt19937_64>
class WrapperSimulatedJointExpectation
{
public:
	WrapperSimulatedJointExpectation(double * price, double * volatility, double * sentiment, double dt, int N_obs, int N_sim,
		int M_sim, std::vector<double> vas_params) :
		N_obs(N_obs), N_sim(N_sim), M_sim(M_sim), dt(dt), price(price), volatility(volatility), sentiment(sentiment),
		lambda_s(vas_params[0]), mu_s(vas_params[1]), sigma_s(vas_params[2])
	{
		// Allocate buffers to hold values for Wiener processes
		wiener_price = (double*)malloc(N_sim * M_sim * sizeof(double));
		wiener_volatility = (double*)malloc(N_sim * M_sim * sizeof(double));
		wiener_sentiment = (double*)malloc(N_sim * M_sim * sizeof(double));

		// Allocate buffers to hold simulated values for individual processses
		simulated_price = (double*)malloc(N_sim * sizeof(double));
		simulated_volatility = (double*)malloc(N_sim * sizeof(double));
		simulated_sentiment = (double*)malloc(N_sim * sizeof(double));

		// Allocate and initialize buffers to hold random values
		GeneratorType generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		int random_buffer_size = N_sim * M_sim;
		random_buffer_price = (double*)malloc(random_buffer_size * sizeof(double));
		random_buffer_volatility = (double*)malloc(random_buffer_size * sizeof(double));
		random_buffer_sentiment = (double*)malloc(random_buffer_size * sizeof(double));
		for (int i = 0; i != random_buffer_size; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	void update_random_buffer()
	{
		std::mt19937_64 generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		for (int i = 0; i != N_sim * M_sim; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	~WrapperSimulatedJointExpectation()
	{
		free(wiener_price);
		free(wiener_volatility);
		free(wiener_sentiment);
		free(simulated_price);
		free(simulated_volatility);
		free(simulated_sentiment);
		free(random_buffer_price);
		free(random_buffer_volatility);
		free(random_buffer_sentiment);
	}

	int N_obs, N_sim, M_sim;
	double dt, lambda_s, mu_s, sigma_s;
	double *price, *volatility, *sentiment; // original processes
	double *simulated_price, *simulated_volatility, *simulated_sentiment; // buffers to hold simulted processes 
	double *wiener_price, *wiener_volatility, *wiener_sentiment; // buffers to hold Wiener processes
	double *random_buffer_price, *random_buffer_volatility, *random_buffer_sentiment; // buffers to hold random draws
};


template<typename GeneratorType = std::mt19937_64>
class WrapperSimulatedJointExpectationLatent
{
public:
	WrapperSimulatedJointExpectationLatent(double * price, double * sentiment, double dt, int N_obs, int N_sim,
		int M_sim, double v0, std::vector<double> vas_params) :
		N_obs(N_obs), N_sim(N_sim), M_sim(M_sim), dt(dt), v0(v0), price(price), sentiment(sentiment),
		lambda_s(vas_params[0]), mu_s(vas_params[1]), sigma_s(vas_params[2])
	{
		// Allocate buffers to hold values for Wiener processes
		wiener_price = (double*)malloc(N_sim * M_sim * sizeof(double));
		wiener_volatility = (double*)malloc(N_sim * M_sim * sizeof(double));
		wiener_sentiment = (double*)malloc(N_sim * M_sim * sizeof(double));

		// Allocate buffers to hold simulated values for individual processses
		simulated_price = (double*)malloc(N_sim * sizeof(double));
		simulated_volatility = (double*)malloc(N_sim * sizeof(double));
		simulated_sentiment = (double*)malloc(N_sim * sizeof(double));

		// Allocate and initialize buffers to hold random values
		GeneratorType generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		int random_buffer_size = N_sim * M_sim;
		random_buffer_price = (double*)malloc(random_buffer_size * sizeof(double));
		random_buffer_volatility = (double*)malloc(random_buffer_size * sizeof(double));
		random_buffer_sentiment = (double*)malloc(random_buffer_size * sizeof(double));
		for (int i = 0; i != random_buffer_size; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	void update_random_buffer()
	{
		std::mt19937_64 generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		for (int i = 0; i != N_sim * M_sim; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	~WrapperSimulatedJointExpectationLatent()
	{
		free(wiener_price);
		free(wiener_volatility);
		free(wiener_sentiment);
		free(simulated_price);
		free(simulated_volatility);
		free(simulated_sentiment);
		free(random_buffer_price);
		free(random_buffer_volatility);
		free(random_buffer_sentiment);
	}

	int N_obs, N_sim, M_sim;
	double dt, lambda_s, mu_s, sigma_s, v0;
	double *price, *sentiment; // original processes
	double *simulated_price, *simulated_volatility, *simulated_sentiment; // buffers to hold simulted processes 
	double *wiener_price, *wiener_volatility, *wiener_sentiment; // buffers to hold Wiener processes
	double *random_buffer_price, *random_buffer_volatility, *random_buffer_sentiment; // buffers to hold random draws
};


class JointParameters
{
public:
	JointParameters(double mu_p, double gamma_p, double lambda_v, double mu_v, double beta_v, double sigma_v,
		double rho_pv, double lambda_s, double mu_s, double sigma_s, double rho_vs) :
		mu_p(mu_p), gamma_p(gamma_p), lambda_v(lambda_v), mu_v(mu_v), beta_v(beta_v), sigma_v(sigma_v), 
		rho_pv(rho_pv), lambda_s(lambda_s), mu_s(mu_s), sigma_s(sigma_s), rho_vs(rho_vs) {}

	double mu_p, gamma_p, lambda_v, mu_v, beta_v, sigma_v, rho_pv, lambda_s, mu_s, sigma_s, rho_vs;
};


template<typename GeneratorType = std::mt19937_64>
class WrapperSimulatedJointSimple
{
public:
	WrapperSimulatedJointSimple(double * sentiment, double dt, int N_obs, int N_sim,
		int M_sim) :
		N_obs(N_obs), N_sim(N_sim), M_sim(M_sim), dt(dt), sentiment(sentiment)
	{
		// Allocate buffers to hold values for Wiener processes
		wiener_sentiment = (double*)malloc(N_sim * M_sim * sizeof(double));

		// Allocate buffers to hold simulated values for individual processses
		simulated_sentiment = (double*)malloc(N_sim * sizeof(double));

		// Allocate and initialize buffers to hold random values
		GeneratorType generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		int random_buffer_size = N_sim * M_sim;
		random_buffer_sentiment = (double*)malloc(random_buffer_size * sizeof(double));
		for (int i = 0; i != random_buffer_size; ++i)
		{
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	void update_random_buffer()
	{
		GeneratorType generator;
		generator.seed(std::random_device()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		int random_buffer_size = N_sim * M_sim;
		for (int i = 0; i != random_buffer_size; ++i)
		{
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	~WrapperSimulatedJointSimple()
	{
		free(wiener_sentiment);
		free(simulated_sentiment);
		free(random_buffer_sentiment);
	}

	int N_obs, N_sim, M_sim;
	double dt;
	double *sentiment; // original processes
	double *simulated_sentiment; // buffers to hold simulted processes 
	double *wiener_sentiment; // buffers to hold Wiener processes
	double *random_buffer_sentiment; // buffers to hold random draws
};