#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#include <random>
#include <iostream>
#include <functional>
#include <sstream>
#include <fstream>
#include <string>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <omp.h>

#include "nlopt.hpp"

// TODO
/* 
1. Problem s estimaciou v 2D, kedy v druhej faze estimacie casto 
hadze inf hodnoty - otazka, ako zefektivnit tuto druhu cast?
*/

/*
R LOAD:
load("D:\\Downloads\\merged_data.RData")
n_row = nrow(merged_data)
x = matrix(nrow = n_row, ncol = 3)
x[, 1] = merged_data$SP
x[, 2] = merged_data$VIX
x[, 3] = merged_data$sm_bull
colnames(x) = c("SP", "VIX", "SM")
write.csv(x, file="D:\\Materials\\Programming\\Projekty\\npsmle\\data.csv", row.names=FALSE)
*/