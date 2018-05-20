#pragma once
#include "Header.h"
#include "Other.h"

// Uncomment type of the run -> whether random or deterministic seed 
// NOTE: Deterministic run of the program cannot be guaranteed in a parallel setting
typedef std::default_random_engine RandomEngine;
// typedef NPSMLE::RandomStart StartType; // uncomment if random run wanted
typedef NPSMLE::DeterministicStart StartType; // uncomment if deterministic run wanted
int NPSMLE::DeterministicStart::ran_seed;

namespace NPSMLE
{
	namespace GLOB
	{
		// Stopping criterion of difference between successive objective function values
		static constexpr double min_optim_diff = 1e-4;

		// Stopping criterion of the maximum number of iterations
		static constexpr int max_number_iter = 10000;

		// Number of iterations 
		static constexpr int time_frame = 16;

		// Location of data
#ifdef LINUX
		static const char *data_loc = R"(/home/cechf/npsmle/data.csv)";
#endif

#ifndef LINUX
		static const char *data_loc = R"(D:/Materials/Programming/Projekty/npsmle/data.csv)";
#endif

		// Name of the logger for sentiment process
		static const char *log_loc_1 = R"(/home/cechf/npsmle/logging/first_step_sentiment.txt)";
		static const char *log_loc_2 = R"(/home/cechf/npsmle/logging/second_step_sentiment.txt)";
		static const char *log_loc_std = R"(/home/cechf/npsmle/logging/std_sentiment.txt)"; // computed only for the second step

		// Uncomment the write mechanism - FileWriter for writing to files, ConsoleWriter for writing to console
		typedef NPSMLE::FileWriter LoggerType;
		// typedef NPSMLE::ConsoleWriter LoggerType;

		// Parameter initialization
		/* Begin and end represent (begin, end] boundaries for std::uniform_distribution<double>
		used in the initialization of parameters in the first step of the estimation*/
		static constexpr double begin = 0.01;
		static constexpr double end = 1.0;
		/* perturbation_param is used for perturbation of the best parameters in the second step of the estimation*/
		static constexpr double perturbation_param = 0.1;

		// PARAMETERS - Single
		static constexpr int N_obs = 1000; // In case of data estimations, the length of observations will be detected automatically
		static constexpr int N_sim = 1000;
		static constexpr int sim_step = 100;
		static constexpr int optim_step = 100;
		static constexpr double delta = 1.0;

		// Sentiment process
		static constexpr double alpha_s = 0.2;
		static constexpr double beta_s = 1.0; // 38.5;
		static constexpr double sigma_s = 0.1; // 0.1;
		static constexpr double s_0 = alpha_s;

		// Joint process
		// 2.0, ..., 0.1, 2.0, 0.2, 0.1, 0.15, -0.5
		// 0.1, ..., 0.1, 1.0, 0.01, 0.1, 0.05, -0.5
		static constexpr int n_params = 7; // number of parameters in the price-volatility process
		static constexpr double gamma_p = 0.25; // 0.2;
		static constexpr double gamma_v = 2.0; // 0.2;
		static constexpr double mu_v = 0.1; // 0.01;
		static constexpr double beta_v = 0.1; // 0.03;
		static constexpr double sigma_v = 0.15; //0.05;
		static constexpr double rho_pv = -0.5;
		static constexpr double mu_p = mu_v + beta_v * alpha_s; // in order to be stationary -> otherwise blows up
		static constexpr double p_0 = mu_p;
		static constexpr double v_0 = mu_v + beta_v * alpha_s;

		// Replication process
		static constexpr double alpha_0_rep = -0.1;
		static constexpr double alpha_1_rep = 3.0;
		static constexpr double alpha_2_rep = 0.2;
		static constexpr double rho_rep = -0.6125;
		static const double mu_rep = ::exp(alpha_0_rep / alpha_1_rep) * 0.5; // cannot be constexpr due to exp
		static constexpr double p_0_rep = 0.1;
		static constexpr double v_0_rep = 0.1;
		static constexpr int n_params_rep = 5;

		// Name of logger for joint process
		static const char *log_loc_3 = R"(/home/cechf/npsmle/logging/first_step_joint.txt)";
		static const char *log_loc_4 = R"(/home/cechf/npsmle/logging/second_step_joint.txt)";
		static const char *log_loc_std_2 = R"(/home/cechf/npsmle/logging/std_joint.txt)"; // computed only for the second step
	}
}