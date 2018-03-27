#define _CRT_SECURE_NO_WARNINGS
#define INFINITY_CHECK
// #define JOINT
// #define SINGLE
// #define JOINT_LATENT
// #define REPLICATION
// #define SINGLE_ESTIMATION
// #define JOINT_ESTIMATION
// #define JOINT_ESTIMATION_FULL
// #define SINGLE_ESTIMATION_MP
#define JOINT_ESTIMATION_FULL_MP

#include "D:\Materials\Programming\Projekty\cpplot\src\Figure.h"
#include "Header.h" 
#include "Other.h"
#include "Single.h"
#include "Joint.h"
#include "Latent.h"
#include "Replication.h"
#include "Joint2D.h"

int DeterministicStart::ran_seed;

#ifdef SINGLE
int main()
{
	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef DeterministicStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 123;

	// Initialize logger
	// FileWriter logger("logging\\log19.txt");
	ConsoleWriter logger{};

	// Set the parameters
	int N_obs = 1000;
	int N_sim = 1000;
	int sim_step = 50;
	int optim_step = 50;
	double alpha = 0.01;
	double beta = 1.0;
	double sigma = 0.1;
	double delta = 1.0; // Beware of delta != 1 as it is harder to estimate!!!
	double start = alpha;

	double * process = simulation_vasicek<RandomEngine, StartType>(alpha, beta, sigma, N_obs, start, delta, sim_step);
	// double * process = simulation_cir<RandomEngine, StartType>(alpha, beta, sigma, N_obs, start, delta, sim_step);

	//WrapperAnalytical wrap_analytical(N_obs, delta, start, process);
	//void * data = static_cast<void*>(&wrap_analytical);

	WrapperSimulated<RandomEngine, StartType> wrap_sim(N_obs, N_sim, optim_step, delta, process);
	void* data = static_cast<void*>(&wrap_sim);

	int n_params = 3;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double objective_function_value = 0.0;
	double best_objective_function_value = INFINITY;

	clock_t begin = clock();
	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_vasicek_optim<RandomEngine, StartType>, data);
	// optimizer.set_min_objective(analytical_ll_vasicek_optim, data);
	// optimizer.set_min_objective(simulated_ll_cir_optim<RandomEngine, StartType>, data);
	optimizer.set_ftol_rel(1e-4);

	// MEASURING TIMING
	logger.write("OPTIMIZATION, 1st STEP \n");

	int time_frame = 10;
	double avg_time = 0.0;
	int avg_counter = 0;
	for (int i = 0; i != time_frame; ++i)
	{
		for (int j = 0; j != n_params; ++j)
		{
			params[j] = (0.1 + rand()) / (double)RAND_MAX;
		}

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}
	}

	logger.write("OPTIMIZATION, 2nd STEP \n");

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.1);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);
	}

	logger.write("AVG TIME: %f \n", avg_time / avg_counter);

	free(process);

	throw;

	return 0;
}
#endif


#ifdef JOINT
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

	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef RandomStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 123;

	// Initialize logger
	// FileWriter logger("logging\\log01.txt");
	ConsoleWriter logger{};

	// JointParameters parameters = JointParameters(0.05, 0.1, 5.0, 0.05, 0.9, -0.5, 0.1, 0.27, 30.0, 0.5, 0.0);
	JointParameters parameters = JointParameters(2.0, 0.5, 0.1, 1.5, 0.2, 0.6, 0.15, -0.5, 1.0, 0.27, 0.2, 0.0);
	// JointParameters parameters = JointParameters(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.06, 0.15, 0.0);

	// Observations and time - distance between obs.
	int N_obs = 1000;
	int NN = 30;
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = NN;

	// Generate buffers to hold stochastic price process from the model
	double * price = (double*)malloc(N_obs * sizeof(double));
	double * volatility = (double*)malloc(N_obs * sizeof(double));
	double * sentiment = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = 0.5;
	double v_0 = 0.02;
	double s_0 = 0.27;

	// Simulate the process
	simulate_joint_process<RandomEngine, StartType>(price, volatility, sentiment, &parameters, delta, N_obs, M_obs, p_0, v_0, s_0);

	// Create plot
	std::vector<double> x(N_obs), y(N_obs), z(N_obs);
	for (int i = 0; i != N_obs; ++i)
	{
		x[i] = volatility[i];
		y[i] = price[i];
		z[i] = sentiment[i];
	}
	cpplot::Figure plt(1000, 1000);
	plt.plot(x, "volatility", "line", 1);
	plt.plot(y, "price", "line", 1);
	plt.plot(z, "sentiment", "line", 1);
	plt.legend();
	plt.xlabel("Price");
	plt.ylabel("Volatility");
	plt.show();

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 1000;
	int M_sim = NN;

	WrapperSimulatedJoint<RandomEngine, StartType> wrapper(price, volatility, sentiment, delta, N_obs, N_sim, M_sim);
	void *data = static_cast<void*>(&wrapper);

	int n_params = 11;
	std::vector<double> test_params{ 2.0, 0.5, 0.1, 1.5, 0.2, 0.6, 0.15, -0.5, 1.0, 0.27, 0.2 };
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double objective_function_value = 0.0;
	double best_objective_function_value = INFINITY;

	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_joint<RandomEngine, StartType>, data);
	optimizer.set_ftol_rel(1e-2);

	logger.write("OPTIMIZATION, 1st STEP \n");

	// MEASURING TIMING
	int time_frame = 10;
	double avg_time = 0.0;
	int avg_counter = 0;
	for (int i = 0; i != time_frame; ++i)
	{
		for (int j = 0; j != n_params; ++j)
		{
			params[j] = (0.1 + rand()) / (double)RAND_MAX;
		}
		
		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}
	}

	logger.write("OPTIMIZATION, 2nd STEP \n");

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.05);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);
	}	

	logger.write("AVG TIME: %f \n", avg_time / avg_counter);

	free(price);
	free(volatility);
	free(sentiment);

return 0;
}

#endif




#ifdef JOINT_LATENT
int main()
{
	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef DeterministicStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 123;

	// Initialize logger
	// FileWriter logger("logging\\log01.txt");
	ConsoleWriter logger{};

	std::vector<double> parameters = {0.25, 0.1, 0.25, 0.1, -0.5 };

	// Observations and time - distance between obs.
	int N_obs = 2000;
	int NN = 50;
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = NN;

	// Generate buffers to hold stochastic price process from the model
	double *price = (double*)malloc(N_obs * sizeof(double));
	double *volatility = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = 0.25;
	double v_0 = 0.25;

	// Simulate the process
	simulate_stoch_vol_process<RandomEngine, StartType>(price, volatility, parameters, delta, N_obs, M_obs, p_0, v_0);

	// Create plot
	std::vector<double> x(volatility, volatility + N_obs);
	std::vector<double> y(price, price + N_obs);
	cpplot::Figure plt(1000, 1000);
	plt.plot(x, "volatility", "line", 1, BLUE);
	plt.plot(y, "price", "line", 1, RED);
	plt.legend();
	plt.xlabel("Price");
	plt.ylabel("Volatility");
	plt.show();

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 2000;
	int M_sim = NN;

	WrapperSimulatedJointLatentVolatility<RandomEngine, StartType> wrapper(price, volatility, delta, N_obs, N_sim, M_sim, v_0);
	void *data = static_cast<void*>(&wrapper);

	logger.write("OPTIMIZATION, 1st STEP \n");

	int n_params = 5;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double objective_function_value = 0.0;
	double best_objective_function_value = INFINITY;

	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_joint_latent<RandomEngine, StartType>, data);
	optimizer.set_ftol_rel(1e-4);

	// MEASURING TIMING
	int time_frame = 10;
	double avg_time = 0.0;
	int avg_counter = 0;

	for (int i = 0; i < time_frame; ++i)
	{
		for (int j = 0; j != n_params; ++j)
		{
			params[j] = (0.1 + rand()) / (double)RAND_MAX;
		}

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}
	}

	logger.write("OPTIMIZATION, 2nd STEP \n");

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.05);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);
	}

	logger.write("AVG TIME: %f \n", avg_time / avg_counter);

	free(price);
	free(volatility);

	throw;

	return 0;
}
#endif




#ifdef REPLICATION
int main()
{
	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef DeterministicStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 123;

	// Initialize logger
	// FileWriter logger("logging\\log01.txt");
	ConsoleWriter logger{};

	double alpha_0 = 0.01;
	double alpha_1 = 1.0;
	double alpha_2 = 0.1;
	double rho = -0.5;
	double mu = exp(alpha_0 / alpha_1) * 0.5;
	std::vector<double> parameters = { mu, alpha_0, alpha_1, alpha_2,  rho };

	// Observations and time - distance between obs.
	int N_obs = 1000;
	int NN = 30;
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = NN;

	// Generate buffers to hold stochastic price process from the model
	double *price = (double*)malloc(N_obs * sizeof(double));
	double *volatility = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = 0.0;
	double v_0 = alpha_0 / alpha_1;

	// Simulate the process
	simulate_replication<RandomEngine, StartType>(price, volatility, parameters, delta, N_obs, M_obs, p_0, v_0);

	// Create plot
	std::vector<double> x(N_obs), y(N_obs), z(N_obs);
	for (int i = 0; i != N_obs; ++i)
	{
		x[i] = volatility[i];
		y[i] = price[i];
		z[i] = alpha_0 / alpha_1;
	}
	cpplot::Figure plt(1000, 1000);
	plt.plot(x, "volatility", "line", 1, BLUE);
	plt.plot(y, "price", "line", 1, RED);
	plt.plot(z, "-1", "line", 1, BLACK);
	plt.legend();
	plt.xlabel("Price");
	plt.ylabel("Volatility");
	plt.show();

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 1000;
	int M_sim = NN;
	
	WrapperSimulatedReplication<RandomEngine, StartType> wrapper(price, volatility, delta, N_obs, N_sim, M_sim, v_0);
	void *data = static_cast<void*>(&wrapper);
	
	int n_params = 5;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double objective_function_value = 0.0;
	double best_objective_function_value = INFINITY;

	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_replication<RandomEngine, StartType>, data);
	optimizer.set_ftol_rel(1e-4);
	// optimizer.set_maxeval(500);

	logger.write("OPTIMIZATION, 1st STEP \n");

	// MEASURING TIMING
	int time_frame = 10;
	double avg_time = 0.0;
	int avg_counter = 0;

	for (int i = 0; i < time_frame; ++i)
	{
		for (int j = 0; j != n_params; ++j)
		{
			params[j] = (0.1 + rand()) / (double)RAND_MAX;
		}

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}
	}

	logger.write("OPTIMIZATION, 2nd STEP \n");

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.1);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);
	}

	logger.write("AVG TIME: %f \n", avg_time / avg_counter);

	free(price);
	free(volatility);
	
	throw;

	return 0;
}
#endif


#ifdef SINGLE_ESTIMATION
int main()
{
	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef DeterministicStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 123;

	// Initialize logger
	// FileWriter logger("logging\\log02.txt");
	ConsoleWriter logger{};

	// Load the data
	const std::string filename = R"(D:/Materials/Programming/Projekty/npsmle/data.csv)";
	const int position = 3;
	const bool header = true;
	int N_obs;
	double *process = loader(filename, position, header, N_obs);
	double *sentiment = &process[2 * N_obs];

	// Create plot
	std::vector<double> x(sentiment, sentiment + N_obs);
	cpplot::Figure plt(1000, 1000);
	plt.plot(x, "volatility", "line", 1, BLUE);
	plt.legend();
	plt.xlabel("Index");
	plt.ylabel("Sentiment");
	plt.show();

	// Set the parameters
	int N_sim = 1000;
	int optim_step = 50;
	double delta = 1.0; // Beware of delta != 1 as it is harder to estimate!!!

	WrapperSimulated<RandomEngine, StartType> wrap_sim(N_obs, N_sim, optim_step, delta, sentiment);
	void* data = static_cast<void*>(&wrap_sim);

	int n_params = 3;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double objective_function_value = 0.0;
	double best_objective_function_value = INFINITY;

	clock_t begin = clock();
	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_vasicek_optim<RandomEngine, StartType>, data);
	optimizer.set_ftol_rel(1e-4);

	// MEASURING TIMING
	logger.write("OPTIMIZATION, 1st STEP \n");

	int time_frame = 10;
	double avg_time = 0.0;
	int avg_counter = 0;
	for (int i = 0; i != time_frame; ++i)
	{
		for (int j = 0; j != n_params; ++j)
		{
			params[j] = (0.1 + rand()) / (double)RAND_MAX;
		}

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}
	}

	logger.write("OPTIMIZATION, 2nd STEP \n");

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.1);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);
	}

	logger.write("AVG TIME: %f \n", avg_time / avg_counter);

	free(process);

	return 0;
}
#endif


#ifdef JOINT_ESTIMATION
int main()
{
	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef RandomStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 123;

	// Initialize logger
	// FileWriter logger("logging\\log03.txt");
	ConsoleWriter logger{};

	JointParameters2D parameters = JointParameters2D(2.0, 0.2, 2.0, 0.2, 0.1, 0.15, -0.5);
	
	// Load the data
	const std::string filename = R"(D:/Materials/Programming/Projekty/npsmle/data.csv)";
	const int position = 3;
	const bool header = true;
	int N_obs;
	double *process = loader(filename, position, header, N_obs);
	double *sentiment = &process[2 * N_obs];

	// Time distance between obs.
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = 50;

	// Generate buffers to hold stochastic price process from the model
	double *price = (double*)malloc(N_obs * sizeof(double));
	double *volatility = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = 0.2;
	double v_0 = 0.2;

	// Simulate the process
	simulate_joint2D_process<RandomEngine, StartType>(price, volatility, sentiment, &parameters, delta, N_obs, M_obs, p_0, v_0);

	// Create plot
	std::vector<double> x(volatility, volatility + N_obs);
	std::vector<double> y(price, price + N_obs);
	std::vector<double> z(sentiment, sentiment + N_obs);
	cpplot::Figure plt(1000, 1000);
	plt.plot(y, "price", "line", 1);
	plt.plot(z, "sentiment", "line", 1);
	plt.plot(x, "volatility", "line", 1);
	plt.legend();
	plt.xlabel("Time");
	plt.ylabel("Value");
	plt.show();

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 1000;
	int M_sim = M_obs;

	WrapperSimulatedJoint2D<RandomEngine, StartType> wrapper(price, volatility, sentiment, delta, N_obs, N_sim, M_sim);
	void *data = static_cast<void*>(&wrapper);

	int n_params = 7;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double objective_function_value = 0.0;
	double best_objective_function_value = INFINITY;

	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_joint2D<RandomEngine, StartType>, data);
	optimizer.set_ftol_rel(1e-4);

	logger.write("OPTIMIZATION, 1st STEP \n");

	// MEASURING TIMING
	int time_frame = 10;
	double avg_time = 0.0;
	int avg_counter = 0;
	for (int i = 0; i != time_frame; ++i)
	{
		for (int j = 0; j != n_params; ++j)
		{
			params[j] = (0.1 + rand()) / (double)RAND_MAX;
		}

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}
	}

	logger.write("OPTIMIZATION, 2nd STEP \n");

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.1);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);
	}

	logger.write("AVG TIME: %f \n", avg_time / avg_counter);

	free(price);
	free(volatility);
	free(process);

	return 0;
}
#endif

#ifdef JOINT_ESTIMATION_FULL
int main()
{
	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef RandomStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 123;

	// Initialize logger
	// FileWriter logger("logging\\log03.txt");
	ConsoleWriter logger{};

	// Load the data
	const std::string filename = R"(D:/Materials/Programming/Projekty/npsmle/data.csv)";
	const int position = 3;
	const bool header = true;
	int N_obs;
	double *process = loader(filename, position, header, N_obs);

	// Time distance between obs.
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = 50;

	// Generate buffers to hold stochastic price process from the model
	double *price = process;
	double *volatility = &process[N_obs];
	double *sentiment = &process[2 * N_obs];

	// Scale the price and volatility
	for (int i = 1; i != N_obs; i++)
	{
		price[i] /= price[0];
		volatility[i] /= volatility[0];
	}
	price[0] = 1.0;
	volatility[0] = 1.0;

	// Initial values of the processes
	double p_0 = 0.2;
	double v_0 = 0.2;

	// Create plot
	std::vector<double> x(volatility, volatility + N_obs);
	std::vector<double> y(price, price + N_obs);
	std::vector<double> z(sentiment, sentiment + N_obs);
	cpplot::Figure plt(1000, 1000);
	plt.plot(y, "price", "line", 1);
	plt.plot(z, "sentiment", "line", 1);
	plt.plot(x, "volatility", "line", 1);
	plt.legend();
	plt.xlabel("Time");
	plt.ylabel("Value");
	plt.show();

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	N_obs = 1000;

	// Parameters
	int N_sim = 1000;
	int M_sim = M_obs;

	WrapperSimulatedJoint2D<RandomEngine, StartType> wrapper(price, volatility, sentiment, delta, N_obs, N_sim, M_sim);
	void *data = static_cast<void*>(&wrapper);

	int n_params = 7;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double objective_function_value = 0.0;
	double best_objective_function_value = INFINITY;

	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_joint2D<RandomEngine, StartType>, data);
	optimizer.set_ftol_rel(1e-4);

	logger.write("OPTIMIZATION, 1st STEP \n");

	// MEASURING TIMING
	int time_frame = 10;
	double avg_time = 0.0;
	int avg_counter = 0;
	for (int i = 0; i != time_frame; ++i)
	{
		for (int j = 0; j != n_params; ++j)
		{
			params[j] = (0.1 + rand()) / (double)RAND_MAX;
		}

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}
	}

	logger.write("OPTIMIZATION, 2nd STEP \n");

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.1);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, params, objective_function_value);
	}

	logger.write("AVG TIME: %f \n", avg_time / avg_counter);

	free(process);

	return 0;
}
#endif

#ifdef SINGLE_ESTIMATION_MP
int main()
{
	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef DeterministicStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 456;

	// Initialize logger
	// FileWriter logger("logging\\log02.txt");
	ConsoleWriter logger{};

	// Load the data
	const std::string filename = R"(D:/Materials/Programming/Projekty/npsmle/data.csv)";
	const int position = 3;
	const bool header = true;
	int N_obs;
	double *process = loader(filename, position, header, N_obs);
	double *sentiment = &process[2 * N_obs];

	// Create plot
	std::vector<double> x(sentiment, sentiment + N_obs);
	cpplot::Figure plt(1000, 1000);
	plt.plot(x, "volatility", "line", 1, BLUE);
	plt.legend();
	plt.xlabel("Index");
	plt.ylabel("Sentiment");
	plt.show();

	// Set the parameters
	int N_sim = 1000;
	int optim_step = 50;
	double delta = 1.0; // Beware of delta != 1 as it is harder to estimate!!!

	// Allocate and set optimization parameters
	int time_frame = 10;
	constexpr int n_params = 3;
	std::vector<double> best_params(n_params);
	std::vector<double> params(n_params);
	double best_objective_function_value = INFINITY;
	
	// Set number of threads
	omp_set_num_threads(omp_get_max_threads());

#pragma omp parallel default(none) firstprivate(N_obs, N_sim, optim_step, delta, params) \
	shared(sentiment, logger)
	{
		int thread_num = omp_get_thread_num();
		srand(thread_num);

		double objective_function_value = 0.0;

		// Create wrapper for the data
		double *local_sentiment = (double*)malloc(N_obs * sizeof(double));
		memcpy(local_sentiment, sentiment, N_obs * sizeof(double));
		WrapperSimulated<RandomEngine, StartType> 
			wrap_sim(N_obs, N_sim, optim_step, delta, local_sentiment);
		void* data = static_cast<void*>(&wrap_sim);

		// Define the optimizer and its properties
		nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
		optimizer.set_min_objective(simulated_ll_vasicek_optim<RandomEngine, StartType>, 
			data);
		optimizer.set_ftol_rel(1e-4);

#pragma omp single
		{
			logger.write("OPTIMIZATION, 1st STEP \n");
		}

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
			for (int j = 0; j != n_params; ++j)
			{
				params[j] = (0.1 + rand()) / (double)RAND_MAX;
			}

			clock_t begin = clock();
			nlopt::result res = optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				logger.write("THREAD NUM: %d\n", thread_num);
				logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, 
					params, objective_function_value);

				if (objective_function_value < best_objective_function_value)
				{
					best_objective_function_value = objective_function_value;
					for (int i = 0; i != n_params; i++)
					{
						best_params[i] = params[i];
					}
				}
			}
		}

#pragma omp single
		{
			logger.write("OPTIMIZATION, 2nd STEP \n");
		}

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
#pragma omp critical
			{
				params = perturb(best_params, 0.1);
			}

			clock_t begin = clock();
			nlopt::result res = optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				logger.write("THREAD NUM: %d\n", thread_num);
				logger.write((clock() - begin) / (double)CLOCKS_PER_SEC, 
					params, objective_function_value);
			}
		}

		free(local_sentiment);
	}

	free(sentiment);

	return 0;
}
#endif


#ifdef JOINT_ESTIMATION_FULL_MP
int main()
{
	// Specify random engine and whether random or deterministic seed
	typedef std::default_random_engine RandomEngine;
	typedef RandomStart StartType; // specify RandomStart or DeterministicStart
	DeterministicStart::ran_seed = 123;

	// Initialize logger
	// FileWriter logger("logging\\log03.txt");
	ConsoleWriter logger{};

	// Load the data
	const std::string filename = R"(D:/Materials/Programming/Projekty/npsmle/data.csv)";
	const int position = 3;
	const bool header = true;
	int N_obs;
	double *process = loader(filename, position, header, N_obs);

	// Time distance between obs.
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = 50;

	// Generate buffers to hold stochastic price process from the model
	double *price = process;
	double *volatility = &process[N_obs];
	double *sentiment = &process[2 * N_obs];

	// Scale the price and volatility
	for (int i = 1; i != N_obs; i++)
	{
		price[i] /= price[0];
		volatility[i] /= volatility[0];
	}
	price[0] = 1.0;
	volatility[0] = 1.0;

	// Initial values of the processes
	double p_0 = 0.2;
	double v_0 = 0.2;

	// Create plot
	std::vector<double> x(volatility, volatility + N_obs);
	std::vector<double> y(price, price + N_obs);
	std::vector<double> z(sentiment, sentiment + N_obs);
	cpplot::Figure plt(1000, 1000);
	plt.plot(y, "price", "line", 1);
	plt.plot(z, "sentiment", "line", 1);
	plt.plot(x, "volatility", "line", 1);
	plt.legend();
	plt.xlabel("Time");
	plt.ylabel("Value");
	plt.show();

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 1000;
	int M_sim = M_obs;
	N_obs = 1000;

	constexpr int time_frame = 10;
	constexpr int n_params = 6;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double best_objective_function_value = INFINITY;
	
	// Set number of threads
	omp_set_num_threads(3);

#pragma omp parallel default(none) firstprivate(N_obs, N_sim, M_sim, delta, params) \
	shared(price, volatility, sentiment, logger)
	{
		int thread_num = omp_get_thread_num();
		srand(thread_num);

		double objective_function_value = 0.0;

		// Create wrapper for the data
		double *local_price = (double*)malloc(N_obs * sizeof(double));
		double *local_volatility = (double*)malloc(N_obs * sizeof(double));
		double *local_sentiment = (double*)malloc(N_obs * sizeof(double));
		memcpy(local_price, price, N_obs * sizeof(double));
		memcpy(local_volatility, volatility, N_obs * sizeof(double));
		memcpy(local_sentiment, sentiment, N_obs * sizeof(double));
		WrapperSimulatedJoint2D<RandomEngine, StartType>
			wrap_sim(local_price, local_volatility, local_sentiment, delta, N_obs, N_sim, M_sim);
		void* data = static_cast<void*>(&wrap_sim);

		// Define the optimizer and its properties
		nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
		optimizer.set_min_objective(simulated_ll_joint2D<RandomEngine, StartType>,
			data);
		optimizer.set_ftol_rel(1e-4);

#pragma omp single
		{
			logger.write("OPTIMIZATION, 1st STEP \n");
		}

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
			for (int j = 0; j != n_params; ++j)
			{
				params[j] = (0.1 + rand()) / (double)RAND_MAX;
			}

			clock_t begin = clock();
			nlopt::result res = optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				logger.write("THREAD NUM: %d\n", thread_num);
				logger.write((clock() - begin) / (double)CLOCKS_PER_SEC,
					params, objective_function_value);

				if (objective_function_value < best_objective_function_value)
				{
					best_objective_function_value = objective_function_value;
					for (int i = 0; i != n_params; i++)
					{
						best_params[i] = params[i];
					}
				}
			}
		}

#pragma omp single
		{
			logger.write("OPTIMIZATION, 2nd STEP \n");
		}

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
#pragma omp critical
			{
				params = perturb(best_params, 0.1);
			}

			clock_t begin = clock();
			nlopt::result res = optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				logger.write("THREAD NUM: %d\n", thread_num);
				logger.write((clock() - begin) / (double)CLOCKS_PER_SEC,
					params, objective_function_value);
			}
		}

		free(local_price);
		free(local_volatility);
		free(local_sentiment);
	}

	free(process);

	return 0;
}
#endif