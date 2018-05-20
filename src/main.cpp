#define _CRT_SECURE_NO_WARNINGS
#define INFINITY_CHECK

#include "Header.h" 
#include "Other.h"
#include "Single.h"
#include "Joint.h" 
#include "Globals.h"
#include "Replication.h"
#include "test.h"

void main_ssn();
void main_ssn_par();
void main_ssa();
void main_sea();
void main_jsn();
void main_jsn_old();
void main_jen();
void main_rep();
void internal_vasicek_estimation(double *process, double delta, int N_obs);
void internal_joint2D_estimation(double *price, double *volatility, double *seniment,
	double delta, int N_obs, int N_sim, int optim_step);
void internal_replication_estimation(double *price, double *volatility, double delta, 
	int N_obs, int N_sim, int optim_step);
void test();

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("Argument for estimation type was not given!\n");

		exit(0);
	}
	
	if (strcmp(argv[1], "SSN") == 0) // Single simulation NPSMLE
	{
		main_ssn();
	}
	else if (strcmp(argv[1], "SSA") == 0) // Single simulation analytical
	{
		main_ssa();
	}
	else if(strcmp(argv[1], "SEA") == 0) // Single estimation analytical
	{
		main_sea();
	}
	else if(strcmp(argv[1], "JSN") == 0) // Joint simulation NPSMLE
	{
		main_jsn();
	}
	else if (strcmp(argv[1], "JEN") == 0) // Joint estimation NPSMLE
	{
		main_jen();
	}
	else if (strcmp(argv[1], "REP") == 0) // Replication of the study
	{
		main_rep();
	}
	else if (strcmp(argv[1], "TEST") == 0) // Test procedures
	{
		test();
	}
	else
	{
		printf("Incorrect procedure chosen! No action taken!\n");
		
		exit(0);
	}
}

void main_ssn()
{
	// Initialize random numbers
	NPSMLE::DeterministicStart::ran_seed = 123;

	NPSMLE::GLOB::LoggerType logger1{ NPSMLE::GLOB::log_loc_1 };
	NPSMLE::GLOB::LoggerType logger2{ NPSMLE::GLOB::log_loc_2 };
	NPSMLE::GLOB::LoggerType logger_std{ NPSMLE::GLOB::log_loc_std };

	// Parameters
	int N_obs = NPSMLE::GLOB::N_obs;
	int N_sim = NPSMLE::GLOB::N_sim;
	int sim_step = NPSMLE::GLOB::sim_step;
	int optim_step = NPSMLE::GLOB::optim_step;
	double alpha = NPSMLE::GLOB::alpha_s;
	double beta = NPSMLE::GLOB::beta_s;
	double sigma = NPSMLE::GLOB::sigma_s;
	double delta = NPSMLE::GLOB::delta;
	double alpha_s = NPSMLE::GLOB::alpha_s;

	// Simulate the Vasicek process
	double *process = (double*)malloc(N_obs * sizeof(double));
	NPSMLE::simulation_vasicek<RandomEngine, StartType>(process, alpha, beta, sigma, delta, N_obs, sim_step, alpha_s);

	constexpr int n_params = 3;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	std::vector<double> mean_params(n_params, 0.0);
	std::vector<double> var_params(n_params, 0.0);
	double best_objective_function_value = INFINITY;

	const int time_frame = NPSMLE::GLOB::time_frame;
	const double begin = NPSMLE::GLOB::begin;
	const double end = NPSMLE::GLOB::end;
	const double perturbation_param = NPSMLE::GLOB::perturbation_param;
	const double min_optim_diff = NPSMLE::GLOB::min_optim_diff;
	const int max_number_iter = NPSMLE::GLOB::max_number_iter;

	// Set number of threads
	omp_set_num_threads(omp_get_max_threads());

#pragma omp parallel default(none) firstprivate(N_obs, N_sim, optim_step, delta, params,\
time_frame, begin, end, perturbation_param, min_optim_diff, max_number_iter)\
shared(process, logger1, logger2, logger_std, best_objective_function_value, best_params,\
mean_params, var_params)
	{
#pragma omp single
		{
			printf("NUMBER OF THREADS: %d\n", omp_get_num_threads());
		}

		double objective_function_value = 0.0;

		// Create wrapper for the data
		double *local_process = (double*)malloc(N_obs * sizeof(double));

#pragma omp critical
		{
			memcpy(local_process, process, N_obs * sizeof(double));
		}

		NPSMLE::WrapperSimulated<RandomEngine, StartType> wrap_sim(N_obs, N_sim, optim_step, delta, local_process);
		void* data = static_cast<void*>(&wrap_sim);

		nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
		optimizer.set_min_objective(NPSMLE::simulated_ll_vasicek_optim<RandomEngine, StartType>, data);

#pragma omp critical
		{
			optimizer.set_ftol_rel(min_optim_diff);
			optimizer.set_maxeval(max_number_iter);
		}

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
			NPSMLE::param_initializer<RandomEngine, StartType>(params, begin, end);

			optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				logger1.write(params, objective_function_value);

				if (objective_function_value < best_objective_function_value)
				{
					best_objective_function_value = objective_function_value;
					best_params = params;
				}
			}
		}

#pragma omp single
		{
			printf("OPTIMIZATION, 2nd STEP \n");
		}

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
#pragma omp critical
			{
				params = NPSMLE::perturb<RandomEngine, StartType>(best_params, perturbation_param);
			}

			optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				NPSMLE::window_var(var_params, params, time_frame);
				NPSMLE::window_mean(mean_params, params, time_frame);
			}

#pragma omp critical
			{
				logger2.write(params, objective_function_value);
			}
		}
	}

	// Compute STD -> var(x) = sqrt( (sum(x_i ^ 2) - n * \mu ^ 2) / (n - 1) )
	for (int i = 0; i != n_params; i++)
	{
		var_params[i] = sqrt(var_params[i] - mean_params[i] * mean_params[i] * time_frame / (time_frame - 1));
	}
	logger_std.write(var_params);

	free(process);
}

void main_ssa()
{
	// Initialize random numbers
	NPSMLE::DeterministicStart::ran_seed = 123;

	// Parameters
	// Set the parameters
	int N_obs = NPSMLE::GLOB::N_obs;
	int sim_step = NPSMLE::GLOB::sim_step;
	double alpha = NPSMLE::GLOB::alpha_s;
	double beta = NPSMLE::GLOB::beta_s;
	double sigma = NPSMLE::GLOB::sigma_s;
	double delta = NPSMLE::GLOB::delta;
	double alpha_s = NPSMLE::GLOB::alpha_s;

	// Simulate the Vasicek process
	double *process = (double*)malloc(N_obs * sizeof(double));
	NPSMLE::simulation_vasicek<RandomEngine, StartType>(process, alpha, beta, sigma, delta, N_obs, sim_step, alpha_s);

	// Optimize
	internal_vasicek_estimation(process, delta, N_obs);

	free(process);
}

void main_sea()
{
	// Initialize random numbers
	NPSMLE::DeterministicStart::ran_seed = 123;

	// Parameters
	double delta = NPSMLE::GLOB::delta;

	// Load the data
	const std::string filename = NPSMLE::GLOB::data_loc;
	const int position = 3;
	const bool header = true;
	int N_obs;
	double *process = NPSMLE::loader(filename, position, header, N_obs);
	double *sentiment = &process[2 * N_obs];

	// Optimize
	internal_vasicek_estimation(sentiment, delta, N_obs);

	free(process);
}


void main_jsn()
{
	// Define random engine
	NPSMLE::DeterministicStart::ran_seed = 123;

	// Parameters
	double alpha_s = NPSMLE::GLOB::alpha_s;
	double beta_s = NPSMLE::GLOB::beta_s;
	double sigma_s = NPSMLE::GLOB::sigma_s;
	NPSMLE::JointParameters parameters = NPSMLE::JointParameters(NPSMLE::GLOB::gamma_p, NPSMLE::GLOB::mu_p,
		NPSMLE::GLOB::gamma_v, NPSMLE::GLOB::mu_v, NPSMLE::GLOB::beta_v, NPSMLE::GLOB::sigma_v, NPSMLE::GLOB::rho_pv);

	int N_obs = NPSMLE::GLOB::N_obs;
	int sim_step = NPSMLE::GLOB::sim_step;
	int N_sim = NPSMLE::GLOB::N_sim;
	int optim_step = NPSMLE::GLOB::optim_step;
	double delta = NPSMLE::GLOB::delta;

	// Generate buffers to hold stochastic price process from the model
	double * price = (double*)malloc(N_obs * sizeof(double));
	double * volatility = (double*)malloc(N_obs * sizeof(double));
	double * sentiment = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = NPSMLE::GLOB::p_0;
	double v_0 = NPSMLE::GLOB::v_0;
	double s_0 = NPSMLE::GLOB::s_0;

	// Simulate the process
	NPSMLE::simulation_vasicek<RandomEngine, StartType>(sentiment, alpha_s, beta_s, sigma_s, delta, N_obs, sim_step, s_0);
	NPSMLE::simulate_joint_process<RandomEngine, StartType>(price, volatility, sentiment, &parameters, delta, N_obs, optim_step, p_0, v_0);

#ifdef WINDOWS
	std::vector<double> x(N_obs), y(N_obs), z(N_obs);
	for (int i = 0; i != N_obs; ++i)
	{
		y[i] = price[i];
		x[i] = volatility[i];
		z[i] = sentiment[i];
	}
	cpplot::Figure plt(1000, 1000);
	plt.plot(y, "price", "line", 1, RED);
	plt.plot(z, "sentiment", "line", 1, BLACK);
	plt.plot(x, "volatility", "line", 1, BLUE);
	plt.legend();
	plt.xlabel("Price");
	plt.ylabel("Volatility");
	plt.show();
#endif

	// OPTIMIZATION
	// Vasicek process
	internal_vasicek_estimation(sentiment, delta, N_obs);

	// 2D price-volatility process
	internal_joint2D_estimation(price, volatility, sentiment, delta, N_obs, N_sim, optim_step);

	free(price);
	free(volatility);
	free(sentiment);
}

void main_jen()
{
	// Specify random seed
	NPSMLE::DeterministicStart::ran_seed = 123;

	// Load the data
	const std::string filename = NPSMLE::GLOB::data_loc;
	const int position = 3;
	const bool header = true;
	int N_obs;
	double *process = NPSMLE::loader(filename, position, header, N_obs);

	// Time distance between obs.
	int N_sim = NPSMLE::GLOB::N_sim;
	int optim_step = NPSMLE::GLOB::optim_step;
	double delta = NPSMLE::GLOB::delta;
	
	// Generate buffers to hold stochastic price process from the model
	double *price = process;
	double *volatility = &process[N_obs];
	double *sentiment = &process[2 * N_obs];

	// Data adjustments
	NPSMLE::volatilityFromVIX(volatility, N_obs);
	NPSMLE::log(price, N_obs);

#ifdef WINDOWS
	std::vector<double> x(N_obs), y(N_obs), z(N_obs);
	for (int i = 0; i != N_obs; ++i)
	{
		y[i] = price[i];
		x[i] = volatility[i];
		z[i] = sentiment[i];
	}
	cpplot::Figure plt(1000, 1000);
	plt.plot(y, "price", "line", 1, RED);
	plt.plot(z, "sentiment", "line", 1, BLACK);
	plt.plot(x, "volatility", "line", 1, BLUE);
	plt.legend();
	plt.xlabel("Price");
	plt.ylabel("Volatility");
	plt.show();
#endif

	// Optimizations
	// Vasicek process
	internal_vasicek_estimation(sentiment, delta, N_obs);

	// 2D price-volatility process
	internal_joint2D_estimation(price, volatility, sentiment, delta, N_obs, N_sim, optim_step);

	free(process);
}


void main_rep()
{
	// Define random engine
	NPSMLE::DeterministicStart::ran_seed = 123;

	// Parameters
	double mu = NPSMLE::GLOB::mu_rep;
	double alpha_0 = NPSMLE::GLOB::alpha_0_rep;
	double alpha_1 = NPSMLE::GLOB::alpha_1_rep;
	double alpha_2 = NPSMLE::GLOB::alpha_2_rep;
	double rho = NPSMLE::GLOB::rho_rep;
	NPSMLE::JointReplicationParameters parameters = NPSMLE::JointReplicationParameters(mu, alpha_0, alpha_1, alpha_2, rho);

	int N_obs = NPSMLE::GLOB::N_obs;
	int sim_step = NPSMLE::GLOB::sim_step;
	int N_sim = NPSMLE::GLOB::N_sim;
	int optim_step = NPSMLE::GLOB::optim_step;
	double delta = NPSMLE::GLOB::delta;

	// Generate buffers to hold stochastic price process from the model
	double * price = (double*)malloc(N_obs * sizeof(double));
	double * volatility = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = NPSMLE::GLOB::p_0_rep;
	double v_0 = NPSMLE::GLOB::v_0_rep;

	// Simulate the process
	NPSMLE::simulate_replication<RandomEngine, StartType>(price, volatility, parameters, delta, N_obs, sim_step, p_0, v_0);

#ifdef WINDOWS
	std::vector<double> x(N_obs), y(N_obs);
	for (int i = 0; i != N_obs; ++i)
	{
		y[i] = price[i];
		x[i] = volatility[i];
	}
	cpplot::Figure plt(1000, 1000);
	plt.plot(x, "volatility", "line", 1, BLUE);
	plt.plot(y, "price", "line", 1, RED);
	plt.legend();
	plt.xlabel("Price");
	plt.ylabel("Volatility");
	plt.show();
#endif

	// OPTIMIZATION
	internal_replication_estimation(price, volatility, delta, N_obs, N_sim, optim_step);

	free(price);
	free(volatility);
}

void internal_vasicek_estimation(double *process, double delta, int N_obs)
{
	// Initialize logger
	NPSMLE::GLOB::LoggerType logger1{ NPSMLE::GLOB::log_loc_1 };
	NPSMLE::GLOB::LoggerType logger2{ NPSMLE::GLOB::log_loc_2 };
	NPSMLE::GLOB::LoggerType logger_std{ NPSMLE::GLOB::log_loc_std };

	// Allocate and set additional optimization variables
	constexpr int time_frame = NPSMLE::GLOB::time_frame;
	constexpr int n_params = 3;
	std::vector<double> best_params(n_params);
	std::vector<double> params(n_params);
	std::vector<double> mean_params(n_params, 0.0);
	std::vector<double> var_params(n_params, 0.0);
	double objective_function_value = 0.0;
	double best_objective_function_value = INFINITY;

	// Create wrapper for the data
	NPSMLE::WrapperAnalytical wrap_analytical(N_obs, delta, process);
	void* data = static_cast<void*>(&wrap_analytical);

	// Define the optimizer and its properties
	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(NPSMLE::analytical_ll_vasicek_optim, data);
	optimizer.set_ftol_rel(NPSMLE::GLOB::min_optim_diff);
	optimizer.set_maxeval(NPSMLE::GLOB::max_number_iter);

	for (int i = 0; i < time_frame; ++i)
	{
		NPSMLE::param_initializer<RandomEngine, StartType>(params, NPSMLE::GLOB::begin, NPSMLE::GLOB::end);

		optimizer.optimize(params, objective_function_value);

		logger1.write(params, objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}
	}

	printf("OPTIMIZATION, 2nd STEP \n");

	for (int i = 0; i < time_frame; ++i)
	{
		params = NPSMLE::perturb<RandomEngine, StartType>(best_params, NPSMLE::GLOB::perturbation_param);

		optimizer.optimize(params, objective_function_value);

		NPSMLE::window_var(var_params, params, time_frame);
		NPSMLE::window_mean(mean_params, params, time_frame);

		logger2.write(params, objective_function_value);
	}

	// Compute STD -> var(x) = sqrt( (sum(x_i ^ 2) - n * \mu ^ 2) / (n - 1) )
	for (int i = 0; i != n_params; i++)
	{
		var_params[i] = sqrt(var_params[i] - mean_params[i] * mean_params[i] * time_frame / (time_frame - 1));
	}
	logger_std.write(var_params);
}

void internal_joint2D_estimation(double *price, double *volatility, double *sentiment, 
	double delta, int N_obs, int N_sim, int optim_step)
{
	// Initialize logger
	NPSMLE::GLOB::LoggerType logger1{ NPSMLE::GLOB::log_loc_3 };
	NPSMLE::GLOB::LoggerType logger2{ NPSMLE::GLOB::log_loc_4 };
	NPSMLE::GLOB::LoggerType logger_std{ NPSMLE::GLOB::log_loc_std_2 };

	constexpr int n_params = NPSMLE::GLOB::n_params;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	std::vector<double> mean_params(n_params, 0.0);
	std::vector<double> var_params(n_params, 0.0);
	double best_objective_function_value = INFINITY;

	const int time_frame = NPSMLE::GLOB::time_frame;
	const double begin = NPSMLE::GLOB::begin;
	const double end = NPSMLE::GLOB::end;
	const double perturbation_param = NPSMLE::GLOB::perturbation_param;
	const double min_optim_diff = NPSMLE::GLOB::min_optim_diff;
	const int max_number_iter = NPSMLE::GLOB::max_number_iter;

	// Set number of threads
	omp_set_num_threads(omp_get_max_threads());

#pragma omp parallel default(none) firstprivate(N_obs, N_sim, optim_step, delta, params, time_frame, begin, end,\
perturbation_param, min_optim_diff, max_number_iter) shared(price, volatility, sentiment, logger1, logger2,\
best_objective_function_value, best_params, mean_params, var_params)
	{
#pragma omp single
		{
			printf("NUMBER OF THREADS: %d\n", omp_get_num_threads());
		}
		
		double objective_function_value = 0.0;

		// Create wrapper for the data
		double *local_price = (double*)malloc(N_obs * sizeof(double));
		double *local_volatility = (double*)malloc(N_obs * sizeof(double));
		double *local_sentiment = (double*)malloc(N_obs * sizeof(double));

#pragma omp critical
		{
			memcpy(local_price, price, N_obs * sizeof(double));
			memcpy(local_volatility, volatility, N_obs * sizeof(double));
			memcpy(local_sentiment, sentiment, N_obs * sizeof(double));
		}

		NPSMLE::WrapperSimulatedJoint<RandomEngine, StartType> wrapper(price, volatility, sentiment, delta, N_obs, N_sim, optim_step);
		void *data = static_cast<void*>(&wrapper);

		nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
		optimizer.set_min_objective(NPSMLE::simulated_ll_joint<RandomEngine, StartType>, data);
		optimizer.set_ftol_rel(min_optim_diff);
		optimizer.set_maxeval(max_number_iter);

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
			NPSMLE::param_initializer<RandomEngine, StartType>(params, begin, end);

			optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				logger1.write(params, objective_function_value);

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
			printf("OPTIMIZATION, 2nd STEP \n");
		}

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
#pragma omp critical
			{
				params = NPSMLE::perturb<RandomEngine, StartType>(best_params, perturbation_param);
			}

			optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				NPSMLE::window_var(var_params, params, time_frame);
				NPSMLE::window_mean(mean_params, params, time_frame);
			}

#pragma omp critical
			{
				logger2.write(params, objective_function_value);
			}
		}

		free(local_price);
		free(local_volatility);
		free(local_sentiment);
	}

	// Compute STD -> var(x) = sqrt( (sum(x_i ^ 2) - n * \mu ^ 2) / (n - 1) )
	for (int i = 0; i != n_params; i++)
	{
		var_params[i] = sqrt(var_params[i] - mean_params[i] * mean_params[i] * time_frame / (time_frame - 1));
	}
	logger_std.write(var_params);
}


void internal_replication_estimation(double *price, double *volatility,
	double delta, int N_obs, int N_sim, int optim_step)
{
	// Initialize logger
	NPSMLE::GLOB::LoggerType logger1{ NPSMLE::GLOB::log_loc_3 };
	NPSMLE::GLOB::LoggerType logger2{ NPSMLE::GLOB::log_loc_4 };
	NPSMLE::GLOB::LoggerType logger_std{ NPSMLE::GLOB::log_loc_std_2 };

	constexpr int n_params = NPSMLE::GLOB::n_params_rep;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	std::vector<double> mean_params(n_params, 0.0);
	std::vector<double> var_params(n_params, 0.0);
	double best_objective_function_value = INFINITY;

	const int time_frame = NPSMLE::GLOB::time_frame;
	const double begin = NPSMLE::GLOB::begin;
	const double end = NPSMLE::GLOB::end;
	const double perturbation_param = NPSMLE::GLOB::perturbation_param;
	const double min_optim_diff = NPSMLE::GLOB::min_optim_diff;
	const int max_number_iter = NPSMLE::GLOB::max_number_iter;

	// Set number of threads
	omp_set_num_threads(omp_get_max_threads());

#pragma omp parallel default(none) firstprivate(N_obs, N_sim, optim_step, delta, params, time_frame, begin, end,\
perturbation_param, min_optim_diff, max_number_iter) shared(price, volatility, logger1, logger2,\
best_objective_function_value, best_params, mean_params, var_params)
	{
#pragma omp single
		{
			printf("NUMBER OF THREADS: %d\n", omp_get_num_threads());
		}

		double objective_function_value = 0.0;

		// Create wrapper for the data
		double *local_price = (double*)malloc(N_obs * sizeof(double));
		double *local_volatility = (double*)malloc(N_obs * sizeof(double));

#pragma omp critical
		{
			memcpy(local_price, price, N_obs * sizeof(double));
			memcpy(local_volatility, volatility, N_obs * sizeof(double));
		}

		NPSMLE::WrapperSimulatedReplication<RandomEngine, StartType> wrapper(price, volatility, delta, N_obs, N_sim, optim_step);
		void *data = static_cast<void*>(&wrapper);

		nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
		optimizer.set_min_objective(NPSMLE::simulated_ll_replication<RandomEngine, StartType>, data);
		optimizer.set_ftol_rel(min_optim_diff);
		optimizer.set_maxeval(max_number_iter);

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
			NPSMLE::param_initializer<RandomEngine, StartType>(params, begin, end);

			optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				logger1.write(params, objective_function_value);

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
			printf("OPTIMIZATION, 2nd STEP \n");
		}

#pragma omp for schedule(dynamic)
		for (int i = 0; i < time_frame; ++i)
		{
#pragma omp critical
			{
				params = NPSMLE::perturb<RandomEngine, StartType>(best_params, perturbation_param);
			}

			optimizer.optimize(params, objective_function_value);

#pragma omp critical
			{
				NPSMLE::window_var(var_params, params, time_frame);
				NPSMLE::window_mean(mean_params, params, time_frame);
			}

#pragma omp critical
			{
				logger2.write(params, objective_function_value);
			}
		}

		free(local_price);
		free(local_volatility);
	}

	// Compute STD -> var(x) = sqrt( (sum(x_i ^ 2) - n * \mu ^ 2) / (n - 1) )
	for (int i = 0; i != n_params; i++)
	{
		var_params[i] = sqrt(var_params[i] - mean_params[i] * mean_params[i] * time_frame / (time_frame - 1));
	}
	logger_std.write(var_params);
}