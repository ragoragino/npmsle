Parallel C++ 11 implementation of a nonparametric simulated maximum likelihood estimation. The implementation is done according to the study:
Kristensen, Dennis, and Yongseok Shin. "Estimation of dynamic models with nonparametric simulated maximum likelihood." Journal of Econometrics 167.1 (2012): 76-94.

## Getting Started

The library provides functionality for estimating Vasicek process with an analytical method and the NPMSLE on simulated or empirical data. 
It also provides functionality for estimating a 3-D process, with Vasicek process as a base incorporated to the slightly adjusted Heston stochastic volatility model, also with simulated and empirical counterparts. The final option is provided as a replication of the original jump-diffusion process presented in the paper, section 4.1.2.

### Prerequisites

OpenMP version 4.7

NLopt Library (https://github.com/stevengj/nlopt) version 2.4.2

### Installing

Makefile is provided for default building.

## Running the tests

Run main.exe -TEST.

## Built With

cmake version 2.8.12.2

gcc version 7.2.1

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.