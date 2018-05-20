#pragma once

#ifdef WINDOWS
#include "D:\Materials\Programming\Projekty\cpplot\src\Figure.h"
#endif

// C headers
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>

// C++ headers
#include <random>
#include <iostream>
#include <functional>
#include <sstream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <chrono>
#include <bitset>

// OMP
#include <omp.h>

// NLOPT
#include "nlopt.hpp"

namespace NPSMLE
{
	// Undefine max macro
#ifdef max
#undef max
#endif

	// Define max of double 
	static constexpr double max_double = std::numeric_limits<double>::max();
}

// Directive to use in the test procedures and checking for a difference from max_double
#define NPSMLE_FP_ERROR 0.0001


/*
R DATA LOAD:
load("D:\\Downloads\\merged_data.RData")
n_row = nrow(merged_data)
x = matrix(nrow = n_row, ncol = 3)
x[, 1] = merged_data$SP
x[, 2] = merged_data$VIX
x[, 3] = merged_data$sm_bull
colnames(x) = c("SP", "VIX", "SM")
write.csv(x, file="D:\\Materials\\Programming\\Projekty\\npsmle\\data.csv", row.names=FALSE)
*/

/* MSVC compile
cl /EHsc main.cpp /I D:\Materials\Programming\Projekty\npsmle\include 
D:\Materials\Programming\Projekty\npsmle\libs\libnlopt-0.lib /openmp /DWINDOWS
*/