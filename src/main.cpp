#define INFINITY_CHECK
// #define JOINT
// #define JOINT_EXP
// #define SINGLE
#define JOINT_EXP_LATENT

#include "Header.h" 
#include "Other.h"
#include "Single.h"
#include "Joint.h"

#ifdef SINGLE
int main()
{
	typedef std::default_random_engine RandomEngine;

	int N_obs = 300;
	int N_sim = 500;
	int sim_step = 50;
	int optim_step = 10;
	double alpha = 0.06;
	double beta = 0.5;
	double sigma = 0.15;
	double delta = 1.0;
	double start = alpha;

	double * process = simulation_vasicek<RandomEngine>(alpha, beta, sigma, N_obs, start, delta, sim_step);
	// double * process = simulation_cir<RandomEngine>(alpha, beta, sigma, N_obs, start, delta, sim_step);
	/*
	WrapperAnalytical wrap_analytical(N_obs, delta, start, process);
	void * data = static_cast<void*>(&wrap_analytical);
	*/
	
	WrapperSimulated<RandomEngine> wrap_sim(N_obs, N_sim, optim_step, delta, process);
	void* data = static_cast<void*>(&wrap_sim);
	
	unsigned int n_params = 3;
	std::vector<double> params(n_params);
	double objective_function_value = 0.0;
	
	clock_t begin = clock();
	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_vasicek_optim, data);
	// optimizer.set_min_objective(analytical_ll_vasicek_optim, data);
	// optimizer.set_min_objective(simulated_ll_cir_optim, data);
	optimizer.set_ftol_rel(1e-4);
	// optimizer.set_lower_bounds(0.05);
	// optimizer.set_upper_bounds(1.0);
	
	// MEASURING TIMING
	int time_frame = 20;
	double avg_time = 0.0;
	for (int i = 0; i != time_frame; ++i)
	{
		params = {
			(0.1 + rand()) / (double)RAND_MAX,
			(0.1 + rand()) / (double)RAND_MAX,
			(0.1 + rand()) / (double)RAND_MAX
		};

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC / time_frame;

		std::cout << (clock() - begin) / (double)CLOCKS_PER_SEC << "\n";
		std::cout << params[0] << " - " << params[1] << " - " << params[2] << "\n";
		std::cout << objective_function_value << "\n";
	}

	std::cout << avg_time << std::endl;
	
	free(process);
	
	return 0;
}
#endif


#ifdef JOINT
int main()
{
	typedef std::default_random_engine RandomEngine;

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

	// JointParameters parameters = JointParameters(0.05, 0.1, 5.0, 0.05, 0.9, -0.5, 0.1, 0.27, 30.0, 0.5, 0.0);
	JointParameters parameters = JointParameters(0.05, 0.1, 5.0, 0.05, 0.0, 0.9, 0.0, 1.0, 0.27, 0.5, 0.0);
	// JointParameters parameters = JointParameters(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.06, 0.15, 0.0);

	// Observations and time - distance between obs.
	int N_obs = 500;
	int NN = 26;
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = NN;

	// Generates stochastic price process from the model P1(volatility stochastic without sentiment)
	double * price = (double*)malloc(N_obs * sizeof(double));
	double * volatility = (double*)malloc(N_obs * sizeof(double));
	double * sentiment = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = 7;
	double v_0 = 0.02;
	double s_0 = 0.27;

	// Simulate the process
	simulate_joint_process<RandomEngine>(price, volatility, sentiment, &parameters, delta, N_obs, M_obs, p_0, v_0, s_0);

	// return_sim = diff(p_sim, 1);
	// return_sim = [0; return_sim];

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 100;
	int M_sim = NN;

	/*
	WrapperSimulatedJoint<RandomEngine> wrapper(sentiment, delta, N_obs, N_sim, M_sim);
	void *data = static_cast<void*>(&wrapper);
	*/

	WrapperSimulatedJoint<RandomEngine> wrapper(price, volatility, sentiment, delta, N_obs, N_sim, M_sim);
	void *data = static_cast<void*>(&wrapper);

	int n_params = 8;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double objective_function_value = 0.0;
	double best_objective_function_value = (double)INFINITY;

	clock_t begin = clock();
	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_joint<RandomEngine>, data);
	optimizer.set_ftol_rel(1e-4);
	// optimizer.set_lower_bounds(0.05);
	// optimizer.set_upper_bounds(1.0);

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

		printf("TIME: %f \n", (clock() - begin) / (double)CLOCKS_PER_SEC);
		printf("PARAMS: ");
		for (int i = 0; i != n_params; ++i)
		{
			printf("%i: %f, ", i, params[i]);
		}
		printf("\n");
		printf("OBJ. FUN.: %f \n", objective_function_value);

		if (objective_function_value < best_objective_function_value)
		{
			best_objective_function_value = objective_function_value;
			best_params = params;
		}

		wrapper.update_random_buffer();
	}

	for (int i = 0; i != time_frame; ++i)
	{
		params = best_params;

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, objective_function_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(objective_function_value) != INFINITY)
		{
			++avg_counter;
		}

		printf("TIME: %f \n", (clock() - begin) / (double)CLOCKS_PER_SEC);
		printf("PARAMS: ");
		for (int i = 0; i != n_params; ++i)
		{
			printf("%i: %f, ", i, params[i]);
		}
		printf("\n");
		printf("OBJ. FUN.: %f \n", objective_function_value);

		wrapper.update_random_buffer();
	}	

	printf("AVG TIME: %f \n", avg_time / avg_counter); 

	free(price);
	free(volatility);
	free(sentiment);

return 0;
}

#endif

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

	// Specify pseudo-random generator to be used
	typedef std::default_random_engine RandomEngine;

	// JointParameters parameters = JointParameters(0.05, 0.1, 5.0, 0.05, 0.9, -0.5, 0.1, 0.27, 30.0, 0.5, 0.0);
	JointParameters parameters = JointParameters(0.05, 0.1, 1.5, 0.2, 0.6, 0.9, 0.0, 1.0, 0.27, 0.5, 0.0);

	// Observations and time - distance between obs.
	int N_obs = 1000;
	int NN = 26;
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = NN;

	// Generates stochastic price process from the model P1(volatility stochastic without sentiment)
	double * price = (double*)malloc(N_obs * sizeof(double));
	double * volatility = (double*)malloc(N_obs * sizeof(double));
	double * sentiment = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = 7;
	double v_0 = 0.02;
	double s_0 = 0.27;

	// Simulate the process
	simulate_joint_process<RandomEngine>(price, volatility, sentiment, &parameters, delta, N_obs, M_obs, p_0, v_0, s_0);

	// return_sim = diff(p_sim, 1);
	// return_sim = [0; return_sim];

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 200;
	int M_sim = NN;

	// OPTIMIZING VASICEK PROCESS
	WrapperAnalytical wrap_vas(N_obs, delta, s_0, sentiment);
	void *data_vas = (void*)(&wrap_vas);

	unsigned int n_params_vas = 3;
	std::vector<double> params_vas(n_params_vas);
	std::vector<double> best_params_vas(n_params_vas);
	double obj_fun_value_vas = 0.0;
	double best_obj_fun_value_vas = INFINITY;

	nlopt::opt optimizer_vas = nlopt::opt(nlopt::LN_NELDERMEAD, n_params_vas);
	optimizer_vas.set_min_objective(analytical_ll_vasicek_optim, data_vas);
	optimizer_vas.set_ftol_rel(1e-4);
	// optimizer_vasicek.set_lower_bounds(0.05);
	// optimizer_vasicek.set_upper_bounds(1.0);

	// MEASURING TIMING
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

	printf("VASICEK PARAMS: ");
	for (int i = 0; i != n_params_vas; ++i)
	{
		printf("%i, %f, ", i, best_params_vas[i]);
	}

	// OPTIMIZING THE REST
	WrapperSimulatedJointExpectation<RandomEngine> wrapper(price, volatility, sentiment, delta, N_obs, N_sim, M_sim, best_params_vas);
	void *data = (void*)(&wrapper);

	int n_params = 6;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double obj_fun_value = 0.0;
	double best_obj_fun_value = (double)INFINITY;

	clock_t begin = clock();
	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_joint_exp<RandomEngine>, data);
	optimizer.set_ftol_rel(1e-4);
	// optimizer.set_lower_bounds(0.05);
	// optimizer.set_upper_bounds(1.0);
	
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
		nlopt::result res = optimizer.optimize(params, obj_fun_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(obj_fun_value) != INFINITY)
		{
			++avg_counter;
		}

		printf("TIME: %f \n", (clock() - begin) / (double)CLOCKS_PER_SEC);
		printf("PARAMS: ");
		for (int i = 0; i != n_params; ++i)
		{
			printf("%i: %f, ", i, params[i]);
		}
		printf("\n");
		printf("OBJ. FUN.: %f \n", obj_fun_value);

		if (obj_fun_value < best_obj_fun_value)
		{
			best_obj_fun_value = obj_fun_value;
			best_params = params;
		}

		// wrapper.update_random_buffer();
	}

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.1);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, obj_fun_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(obj_fun_value) != INFINITY)
		{
			++avg_counter;
		}

		printf("TIME: %f \n", (clock() - begin) / (double)CLOCKS_PER_SEC);
		printf("PARAMS: ");
		for (int i = 0; i != n_params; ++i)
		{
			printf("%i: %f, ", i, params[i]);
		}
		printf("\n");
		printf("OBJ. FUN.: %f \n", obj_fun_value);

		// wrapper.update_random_buffer();
	}

	printf("AVG TIME: %f \n", avg_time / avg_counter);	

	free(price);
	free(volatility);
	free(sentiment);

	return 0;

}
#endif


#ifdef JOINT_EXP_LATENT
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

	// Specify pseudo-random generator to be used
	typedef std::default_random_engine RandomEngine;

	// JointParameters parameters = JointParameters(0.05, 0.1, 5.0, 0.05, 0.9, -0.5, 0.1, 0.27, 30.0, 0.5, 0.0);
	JointParameters parameters = JointParameters(0.05, 0.1, 1.5, 0.2, 0.6, 0.9, 0.0, 1.0, 0.27, 0.5, 0.0);

	// Observations and time - distance between obs.
	int N_obs = 1000;
	int NN = 26;
	double delta = 1.0;

	// Discretization steps used to generate observations
	int M_obs = NN;

	// Generates stochastic price process from the model P1(volatility stochastic without sentiment)
	double * price = (double*)malloc(N_obs * sizeof(double));
	double * volatility = (double*)malloc(N_obs * sizeof(double));
	double * sentiment = (double*)malloc(N_obs * sizeof(double));

	// Initial values of the processes
	double p_0 = 7;
	double v_0 = 0.02;
	double s_0 = 0.27;

	// Simulate the process
	simulate_joint_process<RandomEngine>(price, volatility, sentiment, &parameters, delta, N_obs, M_obs, p_0, v_0, s_0);

	// return_sim = diff(p_sim, 1);
	// return_sim = [0; return_sim];

	/* *********************************************************
	OPTIMIZATION
	********************************************************* */
	// Parameters
	int N_sim = 200;
	int M_sim = NN;

	// OPTIMIZING VASICEK PROCESS
	WrapperAnalytical wrap_vas(N_obs, delta, s_0, sentiment);
	void *data_vas = (void*)(&wrap_vas);

	unsigned int n_params_vas = 3;
	std::vector<double> params_vas(n_params_vas);
	std::vector<double> best_params_vas(n_params_vas);
	double obj_fun_value_vas = 0.0;
	double best_obj_fun_value_vas = INFINITY;

	nlopt::opt optimizer_vas = nlopt::opt(nlopt::LN_NELDERMEAD, n_params_vas);
	optimizer_vas.set_min_objective(analytical_ll_vasicek_optim, data_vas);
	optimizer_vas.set_ftol_rel(1e-4);
	// optimizer_vasicek.set_lower_bounds(0.05);
	// optimizer_vasicek.set_upper_bounds(1.0);

	// MEASURING TIMING
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

	printf("VASICEK PARAMS: ");
	for (int i = 0; i != n_params_vas; ++i)
	{
		printf("%i, %f, ", i, best_params_vas[i]);
	}

	// OPTIMIZING THE REST
	WrapperSimulatedJointExpectationLatent<RandomEngine> wrapper(price, sentiment, delta, N_obs, N_sim, M_sim, v_0, best_params_vas);
	void *data = (void*)(&wrapper);

	int n_params = 6;
	std::vector<double> params(n_params);
	std::vector<double> best_params(n_params);
	double obj_fun_value = 0.0;
	double best_obj_fun_value = (double)INFINITY;

	clock_t begin = clock();
	nlopt::opt optimizer = nlopt::opt(nlopt::LN_NELDERMEAD, n_params);
	optimizer.set_min_objective(simulated_ll_joint_exp_latent<RandomEngine>, data);
	optimizer.set_ftol_rel(1e-4);
	// optimizer.set_lower_bounds(0.05);
	// optimizer.set_upper_bounds(1.0);

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
		nlopt::result res = optimizer.optimize(params, obj_fun_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(obj_fun_value) != INFINITY)
		{
			++avg_counter;
		}

		printf("TIME: %f \n", (clock() - begin) / (double)CLOCKS_PER_SEC);
		printf("PARAMS: ");
		for (int i = 0; i != n_params; ++i)
		{
			printf("%i: %f, ", i, params[i]);
		}
		printf("\n");
		printf("OBJ. FUN.: %f \n", obj_fun_value);

		if (obj_fun_value < best_obj_fun_value)
		{
			best_obj_fun_value = obj_fun_value;
			best_params = params;
		}

		// wrapper.update_random_buffer();
	}

	for (int i = 0; i != time_frame; ++i)
	{
		params = perturb(best_params, 0.05);

		clock_t begin = clock();
		nlopt::result res = optimizer.optimize(params, obj_fun_value);
		avg_time += (clock() - begin) / (double)CLOCKS_PER_SEC;
		if (abs(obj_fun_value) != INFINITY)
		{
			++avg_counter;
		}

		printf("TIME: %f \n", (clock() - begin) / (double)CLOCKS_PER_SEC);
		printf("PARAMS: ");
		for (int i = 0; i != n_params; ++i)
		{
			printf("%i: %f, ", i, params[i]);
		}
		printf("\n");
		printf("OBJ. FUN.: %f \n", obj_fun_value);

		// wrapper.update_random_buffer();
	}

	printf("AVG TIME: %f \n", avg_time / avg_counter);

	free(price);
	free(volatility);
	free(sentiment);

	return 0;

}
#endif