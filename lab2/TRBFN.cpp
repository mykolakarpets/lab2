#include "TRBFN.h"
#include <type_traits>
#include <cassert>


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
	//For each category we choose neurons_count/categories count centers(prototypes)

	//Counting number of prototypes for each category(eliminating situation when neurons_count%categories_count != 0)
	std::vector<int> numOfPrototypes;
	for (size_t i = 0; i < categories_count; i++)
	{
		numOfPrototypes.push_back(neurons_count / categories_count);
	}
	int remainder = neurons_count - neurons_count / categories_count;
	std::vector<int>::iterator iter = numOfPrototypes.begin();
	while (remainder > 0)
	{
		(*(iter++))++;
	}
	for (size_t i = 0; i < categories_count; i++)
	{
		resultOfClustering = dkm::kmeans_lloyd(dividedByCategories[i], numOfPrototypes[i]);
		for each (std::array<double, 2> var in std::get<0>(resultOfClustering))
		{
			mu.push_back(var);
		}
	}

	//Counting betas------------------------------------------------------------------------------------------------

	//First we need to clasterize learnSet
	std::vector<int> clasterizedLearnSet; // contains number of claster for each point from learnSet
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

	//Counting betas
	int m;
	double sum;
	for (size_t betaNum = 0; betaNum < neurons_count; betaNum++)
	{
		m = 0;
		sum = 0;
		for (size_t dataNum = 0; dataNum < learnSet.size(); dataNum++)
		{
			if (clasterizedLearnSet[dataNum] == betaNum)
			{
				m++;
				sum += euclidean_distance({ learnSet[dataNum][0] - mu[betaNum][0], learnSet[dataNum][1] - mu[betaNum][1] });
			}
		}
		beta.push_back(1 / (2 * pow(sum / m, 2.)));
	}

	//Counting W matrix



}

std::vector<int> TRBFN::category(std::vector<std::array<double, 2>>& dataSet)
{
	
}

template< class T>
int TRBFN::category(std::array<double, 2>& vector)
{
	std::vector<double> act_func = activation_function(vector);
	
}

std::vector<double> TRBFN::output(std::array<double, 2>& testVect)
{
	std::vector<double> result;
	std::vector<double> act_func = activation_function(testVect);
	double sum;
	for (size_t outNum = 0; outNum < categories_count; outNum++)
	{
		sum = 0;
		for (size_t neuronNum = 0; neuronNum < neurons_count; neuronNum++)
		{
			sum += W[neuronNum][outNum] * act_func[neuronNum];
		}

		sum += 1; //bias(зміщення)
		result.push_back(sum);
	}
	return result;
}

std::vector<std::vector<double>> TRBFN::output(std::vector<std::array<double, 2>>& testSet)
{
	std::vector<std::vector<double>> result;
	for each (std::array<double, 2> vector in testSet)
	{
		result.push_back(output(vector));
	}
	return result;
}

std::vector<double> TRBFN::activation_function(std::array<double, 2> & vector)
{
	std::vector<double> result;
	std::array<double, 2> difference;
	for (size_t i = 0; i < neurons_count; i++)
	{
		difference = { vector[0] - mu[i][0], vector[1] - mu[i][1]};
		result.push_back(exp(-pow((euclidean_distance(difference)), 2.) / (2 * beta[i])));
	}
	return result;	
}

std::vector<std::array<double, 2>> TRBFN::line(std::pair<double, double>& x_range, std::pair<double, double>& y_range, double presicion)
{
	return std::vector<std::array<double, 2>>();
}


std::array<double, 2> TRBFN::vects_diff(std::array<double, 2>& a, std::array<double, 2>& b)
{
	return {a[0] - b[0], a[1] - b[1]};
}

std::array<double, 2> TRBFN::vects_sum(std::array<double, 2>& a, std::array<double, 2>& b)
{
	return { a[0] + b[0], a[1] + b[1] };
}

double TRBFN::vects_mult(std::array<double, 2>& a, std::array<double, 2>& b)
{
	return (a[0] * b[0]) + (a[1] * b[1]);
}

void TRBFN::debugCheck()
{
#ifdef DEBUG
	assert(neurons_count == mu.size());
	assert(neurons_count == beta.size());
	assert(W.size() == neurons_count);
	for each (std::vector<double> var in W)
	{
		assert(var.size() == categories_count);
	}
#endif
}

TRBFN::~TRBFN()
{
}
