#include "Header.h"
#include "Other.h"

// Test of the rolling window standard deviation computation
void test_window_std()
{
	int length = 1;
	std::vector<double> x = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
	int time_frame = x.size();
	std::vector<double> mean(length, 0.0);
	std::vector<double> var(length, 0.0);

	for (int i = 0; i != time_frame; i++)
	{
		std::vector<double> new_vec = { x[i] };
		NPSMLE::window_var(var, new_vec, time_frame);
		NPSMLE::window_mean(mean, new_vec, time_frame);
	}

	// Compute STD -> var(x) = sqrt( (sum(x_i ^ 2) - n * \mu ^ 2) / (n - 1) )
	for (int i = 0; i != length; i++)
	{
		var[i] = sqrt(var[i] - mean[i] * mean[i] * time_frame / (time_frame - 1));
	}

	assert(fabs(var[0] - 2.738613) < NPSMLE_FP_ERROR);
}

// Function gathering individual tests
void test()
{
	test_window_std();
}