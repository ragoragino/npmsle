#pragma once
#include "Header.h"

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

inline double mean(double * process, int length)
{
	double mean = 0.0;
	for (int i = 0; i != length; ++i)
	{
		mean += process[i];
	}

	return mean /= length;;
}

template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
std::vector<double> perturb(const std::vector<double>& values, double scale = 0.1)
{
	std::vector<double> result{ values };
	GeneratorType generator;
	generator.seed(GeneratorSeed()());
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

template<typename GeneratorType = std::mt19937_64, typename GeneratorSeed = RandomStart>
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

struct JointParameters
{
	JointParameters(double gamma_p, double mu_p, double sigma_p, double lambda_v, double mu_v, double beta_v, double sigma_v,
		double rho_pv, double lambda_s, double mu_s, double sigma_s, double rho_vs) :
		gamma_p(gamma_p), mu_p(mu_p), sigma_p(sigma_p), lambda_v(lambda_v), mu_v(mu_v), beta_v(beta_v), sigma_v(sigma_v),
		rho_pv(rho_pv), lambda_s(lambda_s), mu_s(mu_s), sigma_s(sigma_s), rho_vs(rho_vs) {}

	double gamma_p, mu_p, sigma_p, lambda_v, mu_v, beta_v, sigma_v, rho_pv, lambda_s, mu_s, sigma_s, rho_vs;
};

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
		return ran_seed;
	};

	static int ran_seed;
};



// Logging output to files
class FileWriter
{
public:
	FileWriter(const char * name) : pFile(fopen(name, "w")) 
	{
		if (pFile == NULL)
		{
			printf("ERROR: Logging file was not able to open!");
		}
	};

	~FileWriter()
	{
		fclose(pFile);
	};

	void write(double time, const std::vector<double>& params, double obj_fun_value);
	void write(const char * input);

	template<typename T>
	void write(const char * input, T value);

private:
	FILE * pFile;
};


inline void FileWriter::write(double time, const std::vector<double>& params, double obj_fun_value)
{
	if (pFile != NULL)
	{
		fprintf(pFile, "TIME: %f \n", time);

		fprintf(pFile, "PARAMS: ");
		for (int i = 0; i != params.size(); ++i)
		{
			fprintf(pFile, "%i: %f, ", i, params[i]);
		}

		fprintf(pFile, "\nOBJ. FUN.: %f \n\n", obj_fun_value);

		fflush(pFile);
	}
}

inline void FileWriter::write(const char * input)
{
	if (pFile != NULL)
	{
		fprintf(pFile, input);

		fflush(pFile);
	}
}

template<typename T>
inline void FileWriter::write(const char * input, T value)
{
	static_assert(std::is_arithmetic<T>::value, "Integral type required!");

	if (pFile != NULL)
	{
		fprintf(pFile, input, value);

		fflush(pFile);
	}
}

// Logging output to console
class ConsoleWriter
{
public:
	void write(double time, const std::vector<double>& params, double obj_fun_value);
	void write(const char * input);

	template<typename T>
	void write(const char * input, T value);

	void write(double time, const double *params, int size, double obj_fun_value);
};


inline void ConsoleWriter::write(double time, const std::vector<double>& params, double obj_fun_value)
{
	printf("TIME: %f \n", time);

	printf("PARAMS: ");
	for (int i = 0; i != params.size(); ++i)
	{
		printf("%i: %f, ", i, params[i]);
	}

	printf("\nOBJ. FUN.: %f \n\n", obj_fun_value);
}


inline void ConsoleWriter::write(double time, const double *params, int size, double obj_fun_value)
{
	printf("TIME: %f \n", time);

	printf("PARAMS: ");
	for (int i = 0; i != size; ++i)
	{
		printf("%i: %f, ", i, params[i]);
	}

	printf("\nOBJ. FUN.: %f \n\n", obj_fun_value);
}

inline void ConsoleWriter::write(const char * input)
{
	printf(input);
}

template<typename T>
inline void ConsoleWriter::write(const char * input, T value)
{
	static_assert(std::is_arithmetic<T>::value, "Integral type required!");

	printf(input, value);
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
