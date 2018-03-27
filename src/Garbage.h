

template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
class WrapperSimulatedJointLatent
{
public:
	WrapperSimulatedJointLatent(double * price, double * sentiment, double dt, int N_obs, int N_sim, int M_sim, double v0) :
		N_obs(N_obs), N_sim(N_sim), M_sim(M_sim), dt(dt), price(price), sentiment(sentiment), v0(v0)
	{
		int random_buffer_size = N_sim * M_sim;

		// Allocate buffers to hold values for Wiener processes
		wiener_price = (double*)malloc(random_buffer_size * sizeof(double));
		wiener_volatility = (double*)malloc(random_buffer_size * sizeof(double));
		wiener_sentiment = (double*)malloc(random_buffer_size * sizeof(double));

		// Allocate buffers to hold simulated values for individual processses
		simulated_price = (double*)malloc(N_sim * sizeof(double));
		simulated_volatility = (double*)malloc(N_sim * sizeof(double));
		simulated_sentiment = (double*)malloc(N_sim * sizeof(double));

		// Allocate and initialize buffers to hold random values
		GeneratorType generator;
		generator.seed(GeneratorSeed()());
		std::normal_distribution<double> distribution(0.0, 1.0);
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
		generator.seed(GeneratorSeed()());
		std::normal_distribution<double> distribution(0.0, 1.0);
		for (int i = 0; i != N_sim * M_sim; ++i)
		{
			random_buffer_price[i] = distribution(generator);
			random_buffer_volatility[i] = distribution(generator);
			random_buffer_sentiment[i] = distribution(generator);
		}
	}

	~WrapperSimulatedJointLatent()
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
	double dt, v0;
	double *price, *sentiment; // original processes
	double *simulated_price, *simulated_volatility, *simulated_sentiment; // buffers to hold simulted processes 
	double *wiener_price, *wiener_volatility, *wiener_sentiment; // buffers to hold Wiener processes
	double *random_buffer_price, *random_buffer_volatility, *random_buffer_sentiment; // buffers to hold random draws
};


template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
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
		generator.seed(GeneratorSeed()());
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
		generator.seed(GeneratorSeed()());
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


template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
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
		generator.seed(GeneratorSeed()());
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
		generator.seed(GeneratorSeed()());
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



template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
double simulated_ll_joint_latent(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	// Unwraping parameter
	double mu_p = 0.0;
	double gamma_p = x[0];
	double lambda_v = x[1];
	double mu_v = x[2];
	double beta_v = x[3];
	double sigma_v = x[4];
	double rho_pv = x[5];
	double lambda_s = x[6];
	double mu_s = x[7];
	double sigma_s = x[8];
	double rho_vs = 0.0;

	// Unwraping data
	WrapperSimulatedJointLatent<GeneratorType, GeneratorSeed> *wrapper = static_cast<WrapperSimulatedJointLatent<GeneratorType, GeneratorSeed>*>(data);
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
	double v0 = wrapper->v0;

	// Pre-allocating variables
	double ll = 0.0;
	double kernel_sum_price = 0.0, kernel_sum_sentiment = 0.0, kernel_sum_price_lag = 0.0, kernel_sum_sentiment_lag = 0.0;
	double kernel_sum_numerator = 0.0, kernel_sum_denominator = 0.0;
	static double sqrt_pi = sqrt(2.0 * M_PI);
	double ms, mv, mp, sqrt_vol;
	int dimy = 1;
	double undersmooth = 0.5;
	double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N_obs, -(1.0 + undersmooth) / (dimy + 4.0));
	double h_price, h_sentiment;
	double delta = dt / M_sim;
	double sqrt_delta = sqrt(delta);
	double sentiment_sigma_dt = sigma_s * sqrt_delta;
	double volatility_sigma_dt = sigma_v * sqrt_delta;

	// Fill correlated Wiener process buffers
	double * W_s = wrapper->wiener_sentiment;
	double * W_v = wrapper->wiener_volatility;
	double * W_p = wrapper->wiener_price;

	for (int i = 0; i != N_sim * M_sim; ++i)
	{
		W_s[i] = random_buffer_sentiment[i];
		W_v[i] = sqrt(1.0 - rho_vs * rho_vs) * random_buffer_volatility[i] + rho_vs * W_s[i];
		W_p[i] = sqrt(1.0 - rho_pv * rho_pv) * random_buffer_price[i] + rho_pv * W_v[i];
	}

	// Main log-likelihood computation
	simulated_price[0] = price[0];
	simulated_volatility[0] = v0;
	simulated_sentiment[0] = sentiment[0];

	for (int i = 1; i != N_sim; ++i)
	{
		simulated_price[i] = simulated_price[i - 1];
		simulated_volatility[i] = simulated_volatility[i - 1];
		simulated_sentiment[i] = simulated_sentiment[i - 1];

		for (int j = 0; j != M_sim; ++j)
		{
			mp = gamma_p * (mu_s - simulated_sentiment[i]);
			sqrt_vol = sqrt(abs(simulated_volatility[i]));
			simulated_price[i] += mp * delta + W_p[i * M_sim + j] * sqrt_vol * sqrt_delta;

			mv = lambda_v * (mu_v + beta_v * abs(simulated_sentiment[i]) - simulated_volatility[i]);
			simulated_volatility[i] += mv * delta + W_v[i * M_sim + j] * sqrt_vol * volatility_sigma_dt;

			ms = lambda_s * (mu_s - simulated_sentiment[i]);
			simulated_sentiment[i] += ms * delta + W_s[i * M_sim + j] * sentiment_sigma_dt;
		}
	}

	// Optimal kernel bandwidth computation
	h_price = h_frac * st_dev(simulated_price, N_sim);
	h_sentiment = h_frac * st_dev(simulated_sentiment, N_sim);

	// LIL estimator
	for (int i = 1; i != N_obs; ++i)
	{
		for (int j = 1; j != N_sim; ++j)
		{
			kernel_sum_price = exp((-(simulated_price[j] - price[i]) * (simulated_price[j] - price[i])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			kernel_sum_price_lag = exp((-(simulated_price[j - 1] - price[i - 1]) * (simulated_price[j - 1] - price[i - 1])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			kernel_sum_sentiment = exp((-(simulated_sentiment[j] - sentiment[i]) * (simulated_sentiment[j] - sentiment[i])) / (2.0 * h_sentiment * h_sentiment)) / (h_sentiment * sqrt_pi);
			kernel_sum_sentiment_lag = exp((-(simulated_sentiment[j - 1] - sentiment[i - 1]) * (simulated_sentiment[j - 1] - sentiment[i - 1])) / (2.0 * h_sentiment * h_sentiment)) / (h_sentiment * sqrt_pi);
			kernel_sum_denominator += kernel_sum_price_lag * kernel_sum_sentiment_lag;
			kernel_sum_numerator += kernel_sum_price * kernel_sum_sentiment * kernel_sum_price_lag * kernel_sum_sentiment_lag;
		}

		ll += log(kernel_sum_numerator / (kernel_sum_denominator * h_price * h_sentiment));

#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (ll == -INFINITY)
		{
			return -ll;
		}
#endif

		kernel_sum_numerator = 0.0;
		kernel_sum_denominator = 0.0;
	}

	return -ll;
}


template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
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
	double rho_pv = x[6];

	// Unwraping data
	WrapperSimulatedJointExpectation<GeneratorType, GeneratorSeed> *wrapper = static_cast<WrapperSimulatedJointExpectation<GeneratorType, GeneratorSeed>*>(data);
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
	double kernel_sum = 0.0;
	static double sqrt_pi = sqrt(2.0 * M_PI);
	double ms, mv, mp, sqrt_vol;
	int dimy = 1;
	double undersmooth = 0.5;
	double h_frac = pow(4.0 / dimy + 2.0, 1.0 / (dimy + 4.0)) * pow(N_sim, -(1.0 + undersmooth) / (dimy + 4.0));
	double h_price, h_volatility, h_sentiment;
	double delta = dt / M_sim;
	double sqrt_delta = sqrt(delta);
	double sentiment_sigma_dt = sigma_s * sqrt_delta;
	double volatility_sigma_dt = sigma_v * sqrt_delta;

	// Fill correlated Wiener process buffers
	double * W_s = wrapper->wiener_sentiment;
	double * W_v = wrapper->wiener_volatility;
	double * W_p = wrapper->wiener_price;

	for (int i = 0; i != N_sim * M_sim; ++i)
	{
		W_s[i] = random_buffer_sentiment[i];
		W_v[i] = sqrt(1.0 - rho_vs * rho_vs) * random_buffer_volatility[i] + rho_vs * W_s[i];
		W_p[i] = sqrt(1.0 - rho_pv * rho_pv) * random_buffer_price[i] + rho_pv * W_v[i];
	}

	// Main log-likelihood computation
	for (int i = 1; i != N_obs; ++i)
	{
		for (int j = 0; j < N_sim; ++j)
		{
			simulated_price[j] = price[i - 1];
			simulated_volatility[j] = volatility[i - 1];
			simulated_sentiment[j] = sentiment[i - 1];

			for (int k = 0; k != M_sim; ++k)
			{
				mp = mu_p + gamma_p * simulated_sentiment[j];
				sqrt_vol = sqrt(abs(simulated_volatility[j]));
				simulated_price[j] += mp * delta + W_p[j * M_sim + k] * sqrt_vol * sqrt_delta;

				mv = lambda_v * (mu_v + beta_v * abs(simulated_sentiment[j]) - simulated_volatility[j]);
				simulated_volatility[j] += mv * delta + W_v[j * M_sim + k] * sqrt_vol * volatility_sigma_dt;

				ms = lambda_s * (mu_s - simulated_sentiment[j]);
				simulated_sentiment[j] += ms * delta + W_s[j * M_sim + k] * sentiment_sigma_dt;
			}
		}

		// Optimal kernel bandwidth computation
		h_price = h_frac * st_dev(simulated_price, N_sim);
		h_volatility = h_frac * st_dev(simulated_volatility, N_sim);
		h_sentiment = h_frac * st_dev(simulated_sentiment, N_sim);

		for (int j = 0; j != N_sim; ++j)
		{
			kernel_sum_price = exp((-(simulated_price[j] - price[i]) * (simulated_price[j] - price[i])) / (2.0 * h_price * h_price)) / (h_price * sqrt_pi);
			kernel_sum_volatility = exp((-(simulated_volatility[j] - volatility[i]) * (simulated_volatility[j] - volatility[i])) / (2.0 * h_volatility * h_volatility)) / (h_volatility * sqrt_pi);
			kernel_sum_sentiment = exp((-(simulated_sentiment[j] - sentiment[i]) * (simulated_sentiment[j] - sentiment[i])) / (2.0 * h_sentiment * h_sentiment)) / (h_sentiment * sqrt_pi);
			kernel_sum += kernel_sum_price * kernel_sum_sentiment * kernel_sum_volatility;
		}

		ll += log(kernel_sum / N_sim);

		kernel_sum = 0.0;

#ifdef INFINITY_CHECK
		// Speed up in cases of infinity
		if (ll == -INFINITY || ll == NAN)
		{
			return -ll;
		}
#endif
	}

	return -ll;
}

template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
double simulated_ll_joint_simple(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
	// Unwraping parameters
	double lambda_s = x[0];
	double mu_s = x[1];
	double sigma_s = x[2];
	double lambda_v = x[3];

	// Unwraping data
	WrapperSimulatedJointSimple<GeneratorType, GeneratorSeed> *wrapper = static_cast<WrapperSimulatedJointSimple<GeneratorType, GeneratorSeed>*>(data);
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


#ifdef JOINT_EXP
int main()
{
	/*
	mu_p = theta(1);
	gamma = theta(2);
	lambda_v = theta(3);
	mu_v = theta(4);
	beta = theta(5);
	sigma_v = theta(6);
	rhopv = theta(7);
	lambda_s = theta(8);
	mu_s = theta(9);
	sigma_s = theta(10);
	rhovs = theta(11);
	*/

	// Specify pseudo-random generator to be used and set random_seed if deterministic start
	typedef std::default_random_engine RandomEngine;
	typedef RandomStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 456;

	// Initialize logger
	FileWriter logger("logging\\log05.txt");

	// JointParameters parameters = JointParameters(0.05, 0.1, 5.0, 0.05, 0.9, -0.5, 0.1, 0.27, 30.0, 0.5, 0.0);
	JointParameters parameters = JointParameters(0.05, 0.1, 1.5, 0.2, 0.6, 0.9, -0.5, 1.0, 0.27, 0.5, 0.0);

	// Observations and time
	int N_obs = 1500;
	int NN = 26;
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = NN;

	// Generate stochastic processes from the model P1
	double * price = (double*)malloc(N_obs * sizeof(double));
	double * volatility = (double*)malloc(N_obs * sizeof(double));
	double * sentiment = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = 7;
	double v_0 = 0.02;
	double s_0 = 0.27;

	// Simulate the process
	simulate_joint_process<RandomEngine, StartType>(price, volatility, sentiment, &parameters, delta, N_obs, M_obs, p_0, v_0, s_0);

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 300;
	int M_sim = NN;

	// OPTIMIZING VASICEK PROCESS
	WrapperAnalytical wrap_vas(N_obs, delta, s_0, sentiment);
	void *data_vas = (void*)(&wrap_vas);

	unsigned int n_params_vas = 3;
	std::vector<double> params_vas(n_params_vas);
	std::vector<double> best_params_vas(n_params_vas);
	double obj_fun_value_vas = 0.0;
	double best_obj_fun_value_vas = (double)INFINITY;

	nlopt::opt optimizer_vas = nlopt::opt(nlopt::LN_NELDERMEAD, n_params_vas);
	optimizer_vas.set_min_objective(analytical_ll_vasicek_optim, data_vas);
	optimizer_vas.set_ftol_rel(1e-4);

	int opt_length_vas = 100;
	for (int i = 0; i != opt_length_vas; ++i)
	{
		params_vas = {
			(0.1 + rand()) / (double)RAND_MAX,
			(0.1 + rand()) / (double)RAND_MAX,
			(0.1 + rand()) / (double)RAND_MAX
		};

		nlopt::result res = optimizer_vas.optimize(params_vas, obj_fun_value_vas);

		if (obj_fun_value_vas < best_obj_fun_value_vas)
		{
			best_obj_fun_value_vas = obj_fun_value_vas;
			best_params_vas = params_vas;
		}
	}

	logger.write(0.0, best_params_vas, best_obj_fun_value_vas);

	// OPTIMIZING THE REST
	logger.write("OPTIMIZATION OF THE REST, 1st STEP \n");

	WrapperSimulatedJointExpectation<RandomEngine, StartType> wrapper(price, volatility, sentiment, delta, N_obs, N_sim, M_sim, best_params_vas);
	void *data = (void*)(&wrapper);

	int n_params = 7;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double obj_fun_value = 0.0;
	double best_obj_fun_value = (double)INFINITY;

	clock_t begin = clock();
	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_joint_exp<RandomEngine, StartType>, data);
	optimizer.set_ftol_rel(1e-4);
	// optimizer.set_lower_bounds(0.01);
	// optimizer.set_upper_bounds(1.0);

	int time_frame = 20;
	double avg_time = 0.0;
	int avg_counter = 0;
	for (int i = 0; i != time_frame; ++i)
	{
		// Initialize parameters
		for (int j = 0; j != n_params; ++j)
		{
			params[j] = (0.1 + rand()) / (double)RAND_MAX;
		}

		// Optimize
		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, obj_fun_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(obj_fun_value) != INFINITY)
		{
			++avg_counter;
		}

		// Write results to a file
		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, obj_fun_value);

		if (obj_fun_value < best_obj_fun_value)
		{
			best_obj_fun_value = obj_fun_value;
			best_params = params;
		}
	}

	logger.write("OPTIMIZATION OF THE REST, 2nd STEP \n");

	for (int i = 0; i != time_frame; ++i)
	{
		// Initialize parameters
		params = perturb(best_params, 0.1);

		// Optimize
		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, obj_fun_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(obj_fun_value) != INFINITY)
		{
			++avg_counter;
		}

		// Write results to a file
		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, obj_fun_value);
	}

	logger.write("AVG TIME: %f \n", avg_time / avg_counter);

	free(price);
	free(volatility);
	free(sentiment);

	return 0;

}
#endif