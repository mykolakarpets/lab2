#pragma once
#include <vector>
#include <array>
#include <tuple>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <random>
#include "dkm.hpp"
#include <iomanip>

#define DEBUG
#define DOWN_ERROR_VALUE 0.1
#define LEARNING_COEF 0.0001

class TRBFN
{
	int neurons_count;
	int categories_count;

	//std::vector<std::pair<std::array<double,2>,int>> mu;
	std::vector<std::array<double, 2>> mu;
	std::vector<double> beta;
	std::vector<std::vector<double>> W;

	void configure_mu_beta(std::vector<std::array<double, 3>> & learnSet);

	void configure_W(std::vector<std::array<double, 3>> & learnSet);

	std::pair<std::vector<std::array<double,2>>,std::vector<std::vector<double>>> 
		getLearnSet(std::vector<std::array<double, 3>>& learnSet);

public:
	TRBFN();

	TRBFN(int neurons_count, int categories_count);

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

		x1 - first coodnitate
		x2 - second coordinate
		category - number of category (from 1)

	*/
	
	std::vector<int> 
		category(std::vector<std::array<double, 2>> & dataSet);//In work

	int 
		category(std::array<double, 2> & vector);//In work

	std::vector<double> 
		output(std::array<double, 2> & testVect);

	std::vector<std::vector<double>> 
		output(std::vector<std::array<double, 2>> & testSet);

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
		network_error(std::vector<std::array<double, 2>> & testSet, std::vector<std::vector<double>>& tempSet);

	void 
		debugCheck();

	void
		vectorOut(std::vector<double> & param, std::string caption);

	void
		vectorOut(std::vector<std::vector<double>> & param, std::string caption);

	void
		vectorOut(std::vector<std::array<double,2>> & param, std::string caption);

	~TRBFN();
};

