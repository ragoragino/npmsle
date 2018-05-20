#pragma once
#include "Header.h"

namespace NPSMLE
{
	inline double st_dev(double * process, int length)
	{
		double mean = 0.0;
		double variance = 0.0;
		for (int i = 0; i != length; ++i)
		{
			mean += process[i];
		}

		mean /= length;

		for (int i = 0; i != length; ++i)
		{
			variance += (process[i] - mean) * (process[i] - mean);
		}

		variance /= (length - 1.0);

		return sqrt(variance);
	}

	inline void window_var(std::vector<double>& var, const std::vector<double>& y, int N)
	{
		assert(var.size() == y.size());

		int size = y.size();
		for (int i = 0; i != size; i++)
		{
			var[i] += y[i] * y[i] / (N - 1);
		}
	}

	inline double mean(double * process, int length)
	{
		double mean = 0.0;
		for (int i = 0; i != length; ++i)
		{
			mean += process[i];
		}

		return mean /= length;;
	}

	inline void window_mean(std::vector<double>& mean, const std::vector<double>& y, int N)
	{
		assert(mean.size() == y.size());

		int size = y.size();
		for (int i = 0; i != size; i++)
		{
			mean[i] += y[i] / N;
		}
	}

	inline void normalize(double * data, int length)
	{
		double data_mean = mean(data, length);
		double data_st_dev = st_dev(data, length);

		for (int i = 0; i != length; i++)
		{
			data[i] = (data[i] - data_mean) / data_st_dev;
		}
	}

	inline void log(double * data, int length)
	{
		for (int i = 0; i != length; i++)
		{
			data[i] = ::log(data[i]);
		}
	}

	inline void volatilityFromVIX(double * data, int length)
	{
		for (int i = 0; i != length; i++)
		{
			data[i] = sqrt(pow(data[i] / 100.0, 2.0) / 365.0);
		}
	}

	inline void operation(double * data, int length, double(*operation)(double x))
	{
		for (int i = 0; i != length; i++)
		{
			data[i] = operation(data[i]);
		}
	}

	// Classes to initialize the random seed mechanism
	class RandomStart
	{
	public:
		RandomStart() : random_dev() { };

		std::random_device::result_type operator()() {
			return random_dev();
		};

	private:
		std::random_device random_dev;
	};

	class DeterministicStart
	{
	public:
		DeterministicStart() {};

		int operator()() {
			return ran_seed++;
		};

		static int ran_seed;
	};

	template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
	inline std::vector<double> perturb(const std::vector<double>& values, double scale = 0.1)
	{
		std::vector<double> result{ values };
		static GeneratorType generator;
		generator.seed(GeneratorSeed()());
		std::normal_distribution<double> distribution(0.0, 1.0);

		int result_size = (int)result.size();
		for (int i = 0; i != result_size; ++i)
		{
			result[i] += distribution(generator) * sqrt(fabs(scale * result[i]));
		}

		return result;
	}


	template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
	inline void param_initializer(std::vector<double>& params, double begin = 0.01, double end = 1.0)
	{
		GeneratorType generator;
		generator.seed(GeneratorSeed()());
		std::uniform_real_distribution<double> distribution(begin, end);

		for (double& param : params)
		{
			param = distribution(generator);
		}
	}

	// Classes to wrap all the necessary data for loglikelihood routines (analytical or simulated) passing through NLOPT functions
	class WrapperAnalytical
	{
	public:
		WrapperAnalytical(int N_obs, double delta, double * process) :
			N_obs(N_obs), delta(delta), process(process)
		{}

		int N_obs;
		double delta;
		double * process;
	};

	template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
	class WrapperSimulated
	{
	public:
		WrapperSimulated(int N_obs, int N_sim, int step, double delta,
			double * process) :
			N_obs(N_obs), N_sim(N_sim), step(step), delta(delta),
			process(process)
		{
			simulated_process = (double*)malloc(N_sim * sizeof(double));

			GeneratorType generator;
			generator.seed(GeneratorSeed()());
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

	template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
	class WrapperSimulatedJoint
	{
	public:
		WrapperSimulatedJoint(double * price, double * volatility, double * sentiment, double dt, int N_obs, int N_sim,
			int M_sim) : N_obs(N_obs), N_sim(N_sim), M_sim(M_sim), dt(dt), price(price), volatility(volatility), sentiment(sentiment)
		{
			int buffer_size = N_sim * M_sim;

			// Allocate buffers to hold values for Wiener processes
			wiener_price = (double*)malloc(buffer_size * sizeof(double));
			wiener_volatility = (double*)malloc(buffer_size * sizeof(double));

			// Allocate buffers to hold simulated values for individual processses
			simulated_price = (double*)malloc(N_sim * sizeof(double));
			simulated_volatility = (double*)malloc(N_sim * sizeof(double));

			// Interpolate the sentiment process
			interpolated_sentiment = (double*)malloc((N_obs - 1) * M_sim * sizeof(double));
			double difference;
			for (int i = 1; i != N_obs; i++)
			{
				difference = (sentiment[i] - sentiment[i - 1]) / M_sim;
				for (int j = 0; j != M_sim; j++)
				{
					interpolated_sentiment[(i - 1) * M_sim + j] = sentiment[i - 1] + j * difference;
				}
			}

			// Allocate and initialize buffers to hold random values
			GeneratorType generator;
			generator.seed(GeneratorSeed()());
			std::normal_distribution<double> distribution(0.0, 1.0);
			random_buffer_price = (double*)malloc(buffer_size * sizeof(double));
			random_buffer_volatility = (double*)malloc(buffer_size * sizeof(double));
			for (int i = 0; i != buffer_size; ++i)
			{
				random_buffer_price[i] = distribution(generator);
				random_buffer_volatility[i] = distribution(generator);
			}
		}

		~WrapperSimulatedJoint()
		{
			free(wiener_price);
			free(wiener_volatility);
			free(simulated_price);
			free(simulated_volatility);
			free(random_buffer_price);
			free(random_buffer_volatility);
			free(interpolated_sentiment);
		}

		int N_obs, N_sim, M_sim;
		double dt;
		double *price, *volatility, *sentiment; // original processes
		double *simulated_price, *simulated_volatility; // buffers to hold simulted processes 
		double *wiener_price, *wiener_volatility; // buffers to hold Wiener processes
		double *random_buffer_price, *random_buffer_volatility; // buffers to hold random draws
		double *interpolated_sentiment;
	};

	struct JointParameters
	{
		JointParameters(double gamma_p, double mu_p, double gamma_v, double mu_v, double beta_v, double sigma_v,
			double rho_pv) : gamma_p(gamma_p), mu_p(mu_p), gamma_v(gamma_v), mu_v(mu_v), beta_v(beta_v),
			sigma_v(sigma_v), rho_pv(rho_pv) {}

		double gamma_p, mu_p, gamma_v, mu_v, beta_v, sigma_v, rho_pv;
	};

	template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
	class WrapperSimulatedReplication
	{
	public:
		WrapperSimulatedReplication(double * price, double * volatility, double dt,
			int N_obs, int N_sim, int M_sim) : N_obs(N_obs), N_sim(N_sim),
			M_sim(M_sim), dt(dt), price(price), volatility(volatility)
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
		double dt;
		double *price, *volatility; // original processes
		double * wiener_buffer_volatility, *wiener_buffer_price;
		double *simulated_price, *simulated_volatility; // buffers to hold simulted processes 
		double *random_buffer_price, *random_buffer_volatility; // buffers to hold random draws
	};

	struct JointReplicationParameters
	{
		JointReplicationParameters(double mu, double alpha_0, double alpha_1, double alpha_2, double rho) :
			mu(mu), alpha_0(alpha_0), alpha_1(alpha_1), alpha_2(alpha_2), rho(rho) {}

		double mu, alpha_0, alpha_1, alpha_2, rho;
	};

	bool checker(double objective_function)
	{
		if (fabs(objective_function - max_double) < NPSMLE_FP_ERROR)
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	// Logging output to files
	class FileWriter
	{
	public:
		FileWriter(const char * name) : pFile(fopen(name, "w"))
		{
			if (pFile == NULL)
			{
				printf("ERROR: Logging file was not able to open!\n");

				exit(1);
			}
		};

		~FileWriter()
		{
			fclose(pFile);
		};

		void write(const std::vector<double>& params);
		void write(const std::vector<double>& params, double obj_function);

	private:
		FILE * pFile;
	};

	inline void FileWriter::write(const std::vector<double>& params)
	{
		if (pFile != NULL)
		{
			int params_size = (int)params.size();
			for (int i = 0; i != params_size; ++i)
			{
				fprintf(pFile, "%f, ", params[i]);
			}

			fprintf(pFile, "\n");

			fflush(pFile);
		}
	}

	inline void FileWriter::write(const std::vector<double>& params, double obj_function)
	{
		if (pFile != NULL)
		{
			if (checker(obj_function))
			{
				fprintf(pFile, "%f, ", obj_function);

				int params_size = (int)params.size();
				for (int i = 0; i != params_size; ++i)
				{
					fprintf(pFile, "%f, ", params[i]);
				}

				fprintf(pFile, "\n");

				fflush(pFile);
			}
		}
	}

	// Logging output to console
	class ConsoleWriter
	{
	public:
		ConsoleWriter(const char *) {}; // const char * must be passed in order to unify with FileWriter

		void write(const std::vector<double>& params);
		void write(const std::vector<double>& params, double obj_function);
	};

	inline void ConsoleWriter::write(const std::vector<double>& params)
	{
		int params_size = (int)params.size();
		for (int i = 0; i != params_size; ++i)
		{
			printf("%f, ", params[i]);
		}

		printf("\n");
	}

	inline void ConsoleWriter::write(const std::vector<double>& params, double obj_function)
	{
		if (checker(obj_function))
		{
			printf("%f, ", obj_function);

			int params_size = (int)params.size();
			for (int i = 0; i != params_size; ++i)
			{
				printf("%f, ", params[i]);
			}

			printf("\n");
		}
	}

	// Function to extract value on position specified by parameter position 
	// from a line delimited by tabulator
	void comma_parser(std::string line, std::vector<double>& position, int length)
	{
		std::stringstream str_stream(line);

		std::string value;
		for (int i = 0; i != length; i++)
		{
			std::getline(str_stream, value, ',');

			position.push_back(std::stod(value));
		}
	}


	// Function to extract a column of doubles from a file
	double * loader(const std::string& filename, int positions, bool header, int& count)
	{
		std::vector<double> data;
		std::ifstream input(filename, std::ios::in);

		// Control the state of the fstream and extract the column from the file
		if (!input.is_open())
		{
			printf("Failed to open: %s\n", filename.c_str());
		}
		else
		{
			// Skip the header if present
			if (header)
			{
				std::string line;
				std::getline(input, line, '\n');
			}

			// Parse the text and fill the data structure
			for (std::string line; std::getline(input, line, '\n');)
			{
				comma_parser(line, data, positions);
			}

			input.close();
		}

		// Update the size of the data
		count = (int)data.size() / 3;

		// Reallocate the vector
		double *data_ptr = new double[data.size()];
		for (int i = 0; i != count; i++)
		{
			for (int j = 0; j != positions; j++)
			{
				data_ptr[j * count + i] = data[i * positions + j];
			}
		}

		return data_ptr;
	}
}