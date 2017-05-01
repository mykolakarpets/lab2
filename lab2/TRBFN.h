#pragma once
#include <vector>
#include <array>
#include <tuple>
#include <type_traits>
#include <cassert>
#include <random>
#include <iostream>
#include <iomanip>
#include "dkm.hpp"
#include <thread>

//Degugging flags
#define OUT_W_MATRIX
#define OUT_ERROR
#define STR(x) #x
#define PRINT(F) \
	std::cout <<STR(F)<<F<<std::endl;

#define DOWN_ERROR_VALUE 0.01
#define LEARNING_COEF -0.5
#define MAX_ITERATIONS 200

class TRBFN
{
	int neurons_count;
	int categories_count;

	//std::vector<std::pair<std::array<double,2>,int>> mu;
	std::vector<std::array<double, 2>> mu;
	std::vector<double> beta;

	std::vector<std::vector<double>> W;
	std::vector<double> bias;

public:
	TRBFN();

	TRBFN(int neurons_count, int categories_count);

	TRBFN(std::vector<std::vector<double>> W, std::vector<double> bias);

	void configure_mu_beta(std::vector<std::array<double, 3>> & learnSet);

	void configure_W(std::vector<std::array<double, 3>> & learnSet, int outNum);

	std::vector<double> unnorm_out(std::vector<double> & netOut);

	std::vector<std::array<double, 2>> getMu();

	std::vector<std::vector<double>> getW();

	std::vector<double> getBeta();

	std::pair<std::vector<std::array<double, 2>>, std::vector<std::vector<double>>>
		getLearnSet(std::vector<std::array<double, 3>>& learnSet);

	void 
		learn(std::vector<std::array<double,3>> & learnSet);
	/*
		Structure of the learnSet: 

	    | x1 | x2 | category |
		----------------------
		|....|....|..........|
		|....|....|..........|
		|....|....|..........|
		|....|....|..........|
		|....|....|..........|

	*/
	
	std::vector<int> 
		category(std::vector<std::array<double, 2>> & dataSet);

	int 
		category(std::array<double, 2> & vector);

	std::vector<double> 
		output(std::array<double, 2> & testVect);

	std::vector<std::vector<double>> 
		output(std::vector<std::array<double, 2>> & testSet);

	std::vector<double>
		normalized_output(std::array<double, 2> & testVect);

	std::vector<std::vector<double>>
		normalized_output(std::vector<std::array<double, 2>> & testSet);

	std::vector<double> 
		activation_function(std::array<double, 2> & vector);

	std::vector<std::array<double, 2>> 
		line(std::pair<double,double> & x_range, std::pair<double, double> & y_range, double presicion); // In work

	static double 
		euclidean_distance(std::array<double, 2> vector)
	{
		return sqrt(pow(vector[0], 2.) + pow(vector[1], 2.));
	}

	static std::array<double, 2> 
		vects_diff(std::array<double, 2> & a, std::array<double, 2> & b);

	static std::array<double, 2> 
		vects_sum(std::array<double, 2> & a, std::array<double, 2> & b);

	static double 
		vects_mult(std::array<double, 2> & a, std::array<double, 2> & b);

	double 
		network_error(std::vector<std::array<double, 2>> & testSet, std::vector<std::vector<double>>& tempSet, int category);

	void 
		out_w_matrix();

	void
		out_error(std::vector<std::array<double, 2>> & testSet, std::vector<std::vector<double>>& tempSet, int category);

	~TRBFN();
};

