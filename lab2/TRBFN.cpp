#include "TRBFN.h"



TRBFN::TRBFN()
{

}

TRBFN::TRBFN(int neurons_count, int categories_count)
{
	this->neurons_count = neurons_count;
	this->categories_count = categories_count;
}

void TRBFN::
learn(std::vector<std::array<double, 3>>& learnSet)
{	
	//Converting data to get prototypes(mu)------------------------------------------------------------------------
	std::vector<std::vector<std::array<double, 2>>> dividedByCategories;
	for (size_t i = 0; i < categories_count; i++)
	{
		std::vector<std::array<double, 2>> newVector;
		dividedByCategories.push_back(newVector);
	}

	for each (std::array<double,3> var in learnSet)
	{
		int numOfCategory = var[2];
		std::array<double, 2> coordinates;
		coordinates[0] = var[0];
		coordinates[1] = var[1];
		dividedByCategories[numOfCategory].push_back(coordinates);
	}

	std::tuple<std::vector<std::array<double, 2>>, std::vector<int32_t>> resultOfClustering;
	for (size_t i = 0; i < categories_count; i++)
	{
		resultOfClustering = dkm::kmeans_lloyd(dividedByCategories[i], neurons_count / categories_count);
		for each (std::array<double, 2> var in std::get<0>(resultOfClustering))
		{
			mu.push_back(var);
		}
	}

	//Counting betas------------------------------------------------------------------------------------------------

	//First we need to clasterize learnSet
	std::vector<int> clasterizedLearnSet;
	double minDist , curDist;
	int numberOfMinClaster;
	for (size_t testNum = 0; testNum < learnSet.size(); testNum++)
	{
		minDist = euclidean_distance({ learnSet[testNum][0] - mu[0][0], learnSet[testNum][1] - mu[0][1] });
		numberOfMinClaster = 0;
		for (size_t muNum = 0; muNum < mu.size(); muNum++)
		{
			curDist = euclidean_distance({ learnSet[testNum][0] - mu[muNum][0], learnSet[testNum][1] - mu[muNum][1] });
			if (curDist < minDist)
			{
				minDist = curDist;
				numberOfMinClaster = muNum;
			}
		}
		clasterizedLearnSet.push_back(numberOfMinClaster);
	}

	//

}

int TRBFN::category(std::vector<std::array<double, 2>>& dataSet)
{
	return 0;
}

std::vector<std::array<double, 2>> TRBFN::line(std::pair<double, double>& x_range, std::pair<double, double>& y_range, double presicion)
{
	return std::vector<std::array<double, 2>>();
}


TRBFN::~TRBFN()
{
}
