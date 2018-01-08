#pragma once
#include "Header.h"
#include "Other.h"

template<typename GeneratorType = std::mt19937_64>
void simulate_joint_process(double * price, double * volatility, double * sentiment, JointParameters *
	parameters, double dt, int N_obs, int M_obs, double p0, double v0, double s0)
{
	// Unpack simulation parameters
	double mu_p = parameters->mu_p;
	double gamma_p = parameters->gamma_p;
	double lambda_v = parameters->lambda_v;
	double mu_v = parameters->mu_v;
	double beta_v = parameters->beta_v;
	double sigma_v = parameters->sigma_v;
	double rho_pv = parameters->rho_pv;
	double lambda_s = parameters->lambda_s;
	double mu_s = parameters->mu_s;
	double sigma_s = parameters->sigma_s;
	double rho_vs = parameters->rho_vs;

	// Pre-allocate variables
	double ms, mv, sv, mp, sp;
	double delta = dt / M_obs;
	double sqrt_delta = sqrt(delta);

	// Allocate space for correlated Wiener processes
	double W_p, W_v, W_s;
	
	// Initialize random engines
	GeneratorType generator;
	generator.seed(std::random_device()());
	std::normal_distribution<double> distribution(0.0, 1.0);

	// Fill correlated Wiener process buffers
	// QUESTION : why also this one %dt = Delta / M_obs / 255;
	// QUESTION : why this double dt1 = 250 * dt;			

	sentiment[0] = s0;
	volatility[0] = v0;
	price[0] = p0;

	// Process generation
	for (int i = 1; i != N_obs; ++i)
	{
		sentiment[i] = sentiment[i - 1];
		volatility[i] = volatility[i - 1];
		price[i] = price[i - 1];

		for (int j = 0; j != M_obs; ++j)
		{
			W_s = distribution(generator);
			ms = lambda_s * (mu_s - sentiment[i]);
			sentiment[i] += ms * delta + W_s * sigma_s * sqrt_delta;

			W_v = sqrt(1.0 - rho_vs * rho_vs) * distribution(generator) + rho_vs * W_s;
			mv = lambda_v * (mu_v + beta_v * abs(sentiment[i]) - volatility[i]);
			sv = sigma_v * sqrt(abs(volatility[i]));
			volatility[i] += mv * delta + W_v * sv * sqrt_delta;

			W_p = sqrt(1.0 - rho_pv * rho_pv) * distribution(generator) + rho_pv * W_v;
			mp = mu_p + gamma_p * sentiment[i];
			sp = sqrt(abs(volatility[i]));
			price[i] += mp * delta + W_p * sp * sqrt_delta;
		}
	}
}

template<typename GeneratorType = std::mt19937_64>
double simulated_ll_joint(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	// Unwraping parameter
	double mu_p = x[0];
	double gamma_p = x[1];
	double lambda_v = x[2];
	double mu_v = x[3];
	double beta_v = 0.0;
	double sigma_v = x[4];
	double rho_pv = 0.0;
	double lambda_s = x[5];
	double mu_s = x[6];
	double sigma_s = x[7];
	double rho_vs = 0.0;

	// Unwraping data
	WrapperSimulatedJoint<GeneratorType> *wrapper = static_cast<WrapperSimulatedJoint<GeneratorType>*>(data);
	double *price = wrapper->price;
	double *volatility = wrapper->volatility;
	double *sentiment = wrapper->sentiment;
	double *simulated_price = wrapper->simulated_price;
	double *simulated_volatility = wrapper->simulated_volatility;
	double *simulated_sentiment = wrapper->simulated_sentiment;
	double *random_buffer_price = wrapper->random_buffer_price;
	double *random_buffer_volatility = wrapper->random_buffer_volatility;
	double *random_buffer_sentiment = wrapper->random_buffer_sentiment;
	int N_obs = wrapper->N_obs;
	int N_sim = wrapper->N_sim;
	int M_sim = wrapper->M_sim;
	double dt = wrapper->dt;

	// Pre-allocating variables
	double ll = 0.0;
	double kernel_sum_price = 0.0, kernel_sum_volatility = 0.0, kernel_sum_sentiment = 0.0;
	static double sqrt_pi = pow(2.0 * M_PI, 0.5);
	double ms, mv, mp, sqrt_vol;
	int dimy = 1;
	double undersmooth = 0.5;
	double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N_sim, -(1.0 + undersmooth) / (dimy + 4.0));
	double h_price, h_volatility, h_sentiment;

	double delta = dt / M_sim;
	double sqrt_delta = sqrt(delta);
	double sentiment_sigma_dt = sigma_s * sqrt_delta;

	// Fill correlated Wiener process buffers
	double * W_s = wrapper->wiener_sentiment;
	double * W_v = wrapper->wiener_volatility;
	double * W_p = wrapper->wiener_price;

	for (int i = 0; i != N_sim * M_sim; ++i)
	{
		W_s[i] = random_buffer_sentiment[i];
		W_v[i] = sqrt(1.0 - rho_vs * rho_vs) * random_buffer_volatility[i] + rho_vs * W_s[i];
		W_p[i] = sqrt(1.0 - rho_pv * rho_pv) * random_buffer_price[i] + rho_pv * W_v[i];
		
		//  QUESTION : why also this one %dt = Delta / M_obs / 255;
		// QUESTION : why this double dt1 = 250 * dt;	
	}

	// Main log-likelihood computation
	// QUESTION: why only one-dimensional volatility in the original code
	for (int i = 1; i != N_obs; ++i)
	{
		for (int j = 0; j != N_sim; ++j)
		{
			simulated_price[j] = price[i - 1];
			simulated_volatility[j] = volatility[i - 1];
			simulated_sentiment[j] = sentiment[i - 1];
			for (int k = 0; k != M_sim; ++k)
			{
				ms = lambda_s * (mu_s - simulated_sentiment[j]);
				simulated_sentiment[j] += ms * delta + W_s[j * M_sim + k] * sentiment_sigma_dt;

				mv = lambda_v * (mu_v + beta_v * abs(simulated_sentiment[j]) - simulated_volatility[j]);
				sqrt_vol = sqrt(abs(simulated_volatility[j]));
				simulated_volatility[j] += mv * delta + W_v[j * M_sim + k] * sigma_v * sqrt_vol * sqrt_delta;

				mp = mu_p + gamma_p * simulated_sentiment[j];
				simulated_price[j] += mp * delta + W_p[j * M_sim + k] * sqrt_vol * sqrt_delta;
			}
		}

		h_price = h_frac * st_dev(simulated_price, N_sim); // Optimal kernel bandwidth computation
		h_volatility = h_frac * st_dev(simulated_volatility, N_sim); // Optimal kernel bandwidth computation
		h_sentiment = h_frac * st_dev(simulated_sentiment, N_sim); // Optimal kernel bandwidth computation

		for (int j = 0; j != N_sim; ++j)
		{
			kernel_sum_price += exp((-(simulated_price[j] - price[i]) * (simulated_price[j] - price[i])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			kernel_sum_volatility += exp((-(simulated_volatility[j] - volatility[i]) * (simulated_volatility[j] - volatility[i])) / (2.0 * h_volatility * h_volatility)) / (h_volatility * sqrt_pi);
			kernel_sum_sentiment += exp((-(simulated_sentiment[j] - sentiment[i]) * (simulated_sentiment[j] - sentiment[i])) / (2.0 * h_sentiment * h_sentiment)) / (h_sentiment * sqrt_pi);
		}

		ll += log(kernel_sum_price * kernel_sum_volatility * kernel_sum_sentiment / (N_sim * N_sim * N_sim));

		kernel_sum_price = 0.0, kernel_sum_volatility = 0.0, kernel_sum_sentiment = 0.0;


#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (ll == -INFINITY)
		{
			return -ll;
		}
#endif
	}

	// printf("ll: %f \n", -ll);

	return -ll;
}

template<typename GeneratorType = std::mt19937_64>
double simulated_ll_joint_exp(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	// Unwraping parameter
	double mu_p = x[0];
	double gamma_p = x[1];
	double lambda_v = x[2];
	double mu_v = x[3];
	double beta_v = x[4];
	double sigma_v = x[5];
	double rho_vs = 0.0;
	double rho_pv = 0.0;

	// Unwraping data
	WrapperSimulatedJointExpectation<GeneratorType> *wrapper = static_cast<WrapperSimulatedJointExpectation<GeneratorType>*>(data);
	double *price = wrapper->price;
	double *volatility = wrapper->volatility;
	double *sentiment = wrapper->sentiment;
	double *simulated_price = wrapper->simulated_price;
	double *simulated_volatility = wrapper->simulated_volatility;
	double *simulated_sentiment = wrapper->simulated_sentiment;
	double *random_buffer_price = wrapper->random_buffer_price;
	double *random_buffer_volatility = wrapper->random_buffer_volatility;
	double *random_buffer_sentiment = wrapper->random_buffer_sentiment;
	int N_obs = wrapper->N_obs;
	int N_sim = wrapper->N_sim;
	int M_sim = wrapper->M_sim;
	double dt = wrapper->dt;
	double lambda_s = wrapper->lambda_s;
	double mu_s = wrapper->mu_s;
	double sigma_s = wrapper->sigma_s;

	// Pre-allocating variables
	double ll = 0.0;
	double kernel_sum_price = 0.0, kernel_sum_volatility = 0.0, kernel_sum_sentiment = 0.0;
	static double sqrt_pi = pow(2.0 * M_PI, 0.5);
	double ms, mv, mp, sqrt_vol;
	int dimy = 1;
	double undersmooth = 0.5;
	double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N_sim, -(1.0 + undersmooth) / (dimy + 4.0));
	double h_price, h_volatility, h_sentiment;

	double delta = dt / M_sim;
	double sqrt_delta = sqrt(delta);
	double sentiment_sigma_dt = sigma_s * sqrt_delta;

	// Fill correlated Wiener process buffers
	double * W_s = wrapper->wiener_sentiment;
	double * W_v = wrapper->wiener_volatility;
	double * W_p = wrapper->wiener_price;

	for (int i = 0; i != N_sim * M_sim; ++i)
	{
		W_s[i] = random_buffer_sentiment[i];
		W_v[i] = sqrt(1.0 - rho_vs * rho_vs) * random_buffer_volatility[i] + rho_vs * W_s[i];
		W_p[i] = sqrt(1.0 - rho_pv * rho_pv) * random_buffer_price[i] + rho_pv * W_v[i];

		// QUESTION : why also this one %dt = Delta / M_obs / 255;
		// QUESTION : why this double dt1 = 250 * dt;	
	}

	// Main log-likelihood computation
	// QUESTION: why only one-dimensional volatility in the original code
	for (int i = 1; i != N_obs; ++i)
	{
		for (int j = 0; j != N_sim; ++j)
		{
			simulated_price[j] = price[i - 1];
			simulated_volatility[j] = volatility[i - 1];
			simulated_sentiment[j] = sentiment[i - 1];
			for (int k = 0; k != M_sim; ++k)
			{
				ms = lambda_s * (mu_s - simulated_sentiment[j]);
				simulated_sentiment[j] += ms * delta + W_s[j * M_sim + k] * sentiment_sigma_dt;

				mv = lambda_v * (mu_v + beta_v * abs(simulated_sentiment[j]) - simulated_volatility[j]);
				sqrt_vol = sqrt(abs(simulated_volatility[j]));
				simulated_volatility[j] += mv * delta + W_v[j * M_sim + k] * sigma_v * sqrt_vol * sqrt_delta;

				mp = mu_p + gamma_p * simulated_sentiment[j];
				simulated_price[j] += mp * delta + W_p[j * M_sim + k] * sqrt_vol * sqrt_delta;
			}
		}

		h_price = h_frac * st_dev(simulated_price, N_sim); // Optimal kernel bandwidth computation
		h_volatility = h_frac * st_dev(simulated_volatility, N_sim); // Optimal kernel bandwidth computation
		h_sentiment = h_frac * st_dev(simulated_sentiment, N_sim); // Optimal kernel bandwidth computation

		for (int j = 0; j != N_sim; ++j)
		{
			kernel_sum_price += exp((-(simulated_price[j] - price[i]) * (simulated_price[j] - price[i])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			kernel_sum_volatility += exp((-(simulated_volatility[j] - volatility[i]) * (simulated_volatility[j] - volatility[i])) / (2.0 * h_volatility * h_volatility)) / (h_volatility * sqrt_pi);
			kernel_sum_sentiment += exp((-(simulated_sentiment[j] - sentiment[i]) * (simulated_sentiment[j] - sentiment[i])) / (2.0 * h_sentiment * h_sentiment)) / (h_sentiment * sqrt_pi);
		}

		ll += log(kernel_sum_price * kernel_sum_volatility * kernel_sum_sentiment / (N_sim * N_sim * N_sim));

		kernel_sum_price = 0.0, kernel_sum_volatility = 0.0, kernel_sum_sentiment = 0.0;


#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (ll == -INFINITY)
		{
			return -ll;
		}
#endif
	}

	// printf("ll: %f \n", -ll);

	return -ll;
}

template<typename GeneratorType = std::mt19937_64>
double simulated_ll_joint_exp_latent(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	// Unwraping parameter
	double mu_p = x[0];
	double gamma_p = x[1];
	double lambda_v = x[2];
	double mu_v = x[3];
	double beta_v = x[4];
	double sigma_v = x[5];
	double rho_vs = 0.0;
	double rho_pv = 0.0;

	// Unwraping data
	WrapperSimulatedJointExpectationLatent<GeneratorType> *wrapper = static_cast<WrapperSimulatedJointExpectationLatent<GeneratorType>*>(data);
	double *price = wrapper->price;
	double *sentiment = wrapper->sentiment;
	double *simulated_price = wrapper->simulated_price;
	double *simulated_volatility = wrapper->simulated_volatility;
	double *simulated_sentiment = wrapper->simulated_sentiment;
	double *random_buffer_price = wrapper->random_buffer_price;
	double *random_buffer_volatility = wrapper->random_buffer_volatility;
	double *random_buffer_sentiment = wrapper->random_buffer_sentiment;
	int N_obs = wrapper->N_obs;
	int N_sim = wrapper->N_sim;
	int M_sim = wrapper->M_sim;
	double dt = wrapper->dt;
	double lambda_s = wrapper->lambda_s;
	double mu_s = wrapper->mu_s;
	double sigma_s = wrapper->sigma_s;
	double v0 = wrapper->v0;

	// Pre-allocating variables
	double ll = 0.0;
	double kernel_sum_price = 0.0, kernel_sum_sentiment = 0.0;
	static double sqrt_pi = pow(2.0 * M_PI, 0.5);
	double ms, mv, mp, sqrt_vol;
	int dimy = 1;
	double undersmooth = 0.5;
	double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N_sim, -(1.0 + undersmooth) / (dimy + 4.0));
	double h_price,  h_sentiment;

	double delta = dt / M_sim;
	double sqrt_delta = sqrt(delta);
	double sentiment_sigma_dt = sigma_s * sqrt_delta;

	// Fill correlated Wiener process buffers
	double * W_s = wrapper->wiener_sentiment;
	double * W_v = wrapper->wiener_volatility;
	double * W_p = wrapper->wiener_price;

	for (int i = 0; i != N_sim * M_sim; ++i)
	{
		W_s[i] = random_buffer_sentiment[i];
		W_v[i] = sqrt(1.0 - rho_vs * rho_vs) * random_buffer_volatility[i] + rho_vs * W_s[i];
		W_p[i] = sqrt(1.0 - rho_pv * rho_pv) * random_buffer_price[i] + rho_pv * W_v[i];

		// QUESTION : why also this one %dt = Delta / M_obs / 255;
		// QUESTION : why this double dt1 = 250 * dt;	
	}

	for (int i = 0; i != N_sim; ++i)
	{
		simulated_volatility[i] = v0;
	}

	// Main log-likelihood computation
	// QUESTION: why only one-dimensional volatility in the original code
	for (int i = 1; i != N_obs; ++i)
	{
		for (int j = 0; j != N_sim; ++j)
		{
			simulated_price[j] = price[i - 1];
			simulated_sentiment[j] = sentiment[i - 1];
			for (int k = 0; k != M_sim; ++k)
			{
				ms = lambda_s * (mu_s - simulated_sentiment[j]);
				simulated_sentiment[j] += ms * delta + W_s[j * M_sim + k] * sentiment_sigma_dt;

				mv = lambda_v * (mu_v + beta_v * abs(simulated_sentiment[j]) - simulated_volatility[j]);
				sqrt_vol = sqrt(abs(simulated_volatility[j]));
				simulated_volatility[j] += mv * delta + W_v[j * M_sim + k] * sigma_v * sqrt_vol * sqrt_delta;

				mp = mu_p + gamma_p * simulated_sentiment[j];
				simulated_price[j] += mp * delta + W_p[j * M_sim + k] * sqrt_vol * sqrt_delta;
			}
		}

		h_price = h_frac * st_dev(simulated_price, N_sim); // Optimal kernel bandwidth computation
		h_sentiment = h_frac * st_dev(simulated_sentiment, N_sim); // Optimal kernel bandwidth computation

		for (int j = 0; j != N_sim; ++j)
		{
			kernel_sum_price += exp((-(simulated_price[j] - price[i]) * (simulated_price[j] - price[i])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			kernel_sum_sentiment += exp((-(simulated_sentiment[j] - sentiment[i]) * (simulated_sentiment[j] - sentiment[i])) / (2.0 * h_sentiment * h_sentiment)) / (h_sentiment * sqrt_pi);
		}

		ll += log(kernel_sum_price * kernel_sum_sentiment / (N_sim * N_sim));

		kernel_sum_price = 0.0, kernel_sum_sentiment = 0.0;


#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (ll == -INFINITY)
		{
			return -ll;
		}
#endif
	}

	return -ll;
}


template<typename GeneratorType = std::mt19937_64>
double simulated_ll_joint_simple(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	// Unwraping parameters
	double lambda_s = x[0];
	double mu_s = x[1];
	double sigma_s = x[2];
	double lambda_v = x[3];
	
	// Unwraping data
	WrapperSimulatedJointSimple<GeneratorType> *wrapper = static_cast<WrapperSimulatedJointSimple<GeneratorType>*>(data);
	double *sentiment = wrapper->sentiment;
	double *simulated_sentiment = wrapper->simulated_sentiment;
	double *random_buffer_sentiment = wrapper->random_buffer_sentiment;
	int N_obs = wrapper->N_obs;
	int N_sim = wrapper->N_sim;
	int M_sim = wrapper->M_sim;
	double dt = wrapper->dt;

	// Pre-allocating variables
	double ll = 0.0;
	double kernel_sum_sentiment = 0.0;
	static double sqrt_pi = pow(2.0 * M_PI, 0.5);
	double ms;
	int dimy = 1;
	double undersmooth = 0.5;
	double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N_sim, -(1.0 + undersmooth) / (dimy + 4.0));
	double h_sentiment;
	double delta = dt / M_sim;
	double sqrt_delta = sqrt(delta);
	double sentiment_sigma_dt = sigma_s * sqrt(delta);

	// Fill correlated Wiener process buffers
	double * W_s = wrapper->wiener_sentiment;

	for (int i = 0; i != N_sim * M_sim; ++i)
	{
		W_s[i] = random_buffer_sentiment[i];
	}

	// Main log-likelihood computation
	// QUESTION: why only one-dimensional volatility in the original code
	for (int i = 1; i != N_obs; ++i)
	{
		for (int j = 0; j != N_sim; ++j)
		{
			simulated_sentiment[j] = sentiment[i - 1];
			for (int k = 0; k != M_sim; ++k)
			{
				ms = lambda_s * (mu_s - simulated_sentiment[j]);
				simulated_sentiment[j] += ms * delta + W_s[j * M_sim + k] * sentiment_sigma_dt;
			}
		}

		h_sentiment = h_frac * st_dev(simulated_sentiment, N_sim); // Optimal kernel bandwidth computation

		for (int j = 0; j != N_sim; ++j)
		{
			kernel_sum_sentiment += exp((-(simulated_sentiment[j] - sentiment[i]) * (simulated_sentiment[j] - sentiment[i])) / (2.0 * h_sentiment * h_sentiment)) / (h_sentiment * sqrt_pi);
		}

		ll += log(kernel_sum_sentiment / N_sim);

		kernel_sum_sentiment = 0.0;


#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (ll == -INFINITY)
		{
			return -ll;
		}
#endif
	}

	return -ll;
}