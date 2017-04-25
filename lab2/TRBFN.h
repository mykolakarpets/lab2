#pragma once
#include <vector>
#include "dkm.hpp"

class TRBFN
{
	int neurons_count;
	int categories_count;

	//std::vector<std::pair<std::array<double,2>,int>> mu;
	std::vector<std::array<double, 2>> mu;
	std::vector<double> beta;
	std::vector<std::vector<double>> W;
public:
	TRBFN();
	TRBFN(int neurons_count, int categories_count);

	void learn(std::vector<std::array<double,3>> & learnSet);
	
	int category(std::vector<std::array<double, 2>> & dataSet);
	std::vector<std::array<double, 2>> line(std::pair<double,double> & x_range,
		std::pair<double, double> & y_range, double presicion);

	static double euclidean_distance(std::array<double, 2> vector)
	{
		return sqrt(pow(vector[0], 2.) + pow(vector[1], 2.));
	}

	~TRBFN();
};

