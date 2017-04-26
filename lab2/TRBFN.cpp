#include "TRBFN.h"

void 
TRBFN::configure_mu_beta(std::vector<std::array<double, 3>> & learnSet)
{
	std::vector<std::array<double, 2>> convertedData;

	for each (auto learnVect in learnSet)
		convertedData.push_back({ learnVect[0], learnVect[1] });

	auto resultOfClustering = dkm::kmeans_lloyd(convertedData, neurons_count);

	//Filling mu
	mu.clear();
	for each (auto mean in std::get<0>(resultOfClustering))
	{
		mu.push_back(mean);
	}

	//Counting betas
	int m;
	double sum;
	for (int betaNum = 0; betaNum < neurons_count; betaNum++)
	{
		m = 0;
		sum = 0;
		for (int dataNum = 0; dataNum < (int)convertedData.size(); dataNum++)
		{
			if (std::get<1>(resultOfClustering)[dataNum] == betaNum)
			{
				m++;
				sum += euclidean_distance({ learnSet[dataNum][0] - mu[betaNum][0], learnSet[dataNum][1] - mu[betaNum][1] });
			}
		}
		beta[betaNum] = 1 / (2 * pow((double)(sum / m), 2.));
	}
}

void 
TRBFN::configure_W(std::vector<std::array<double, 3>>& learnSet)
{
	std::pair<std::vector<std::array<double, 2>>, std::vector<std::vector<double>>> transformedLearnSet = getLearnSet(learnSet);
	std::vector<std::array<double, 2>> learnVectors = transformedLearnSet.first;
	std::vector<std::vector<double>> learnOutputs = transformedLearnSet.second;

	double sum;
	std::vector<double> actFuncRes;
	std::vector<double> networkRes;
	while (network_error(transformedLearnSet.first, transformedLearnSet.second) > DOWN_ERROR_VALUE)
	{
		for (int neuronNum = 0; neuronNum < neurons_count; neuronNum++)
		{
			for (int outNum = 0; outNum < categories_count; outNum++)
			{
				sum = 0;
				for (int learnNum = 0; learnNum < (int)learnVectors.size(); learnNum++)
				{
					actFuncRes = activation_function(learnVectors[learnNum]);
					networkRes = output(learnVectors[learnNum]);

					sum += actFuncRes[neuronNum] * (learnOutputs[learnNum][outNum] - networkRes[outNum]);
				}
				W[neuronNum][outNum] += sum * LEARNING_COEF;
			}
		}
	}
}

std::pair<std::vector<std::array<double, 2>>, std::vector<std::vector<double>>> 
TRBFN::getLearnSet(std::vector<std::array<double, 3>>& learnSet)
{
	std::pair<std::vector<std::array<double, 2>>, std::vector<std::vector<double>>> resultPair;
	std::vector<std::array<double, 2>> resultVectArr;
	std::vector<std::vector<double>> resultVectVect;
	std::vector<double> resultVect;
	for each (std::array<double, 3> var in learnSet)
	{
		resultVectArr.push_back({ var[0],var[1] });
		resultVect.clear();
		for (int catNum = 0; catNum < categories_count; catNum++)
		{
			if (catNum == var[2]) resultVect.push_back(1);
			else resultVect.push_back(0);
		}
		resultVectVect.push_back(resultVect);
	}
	return std::make_pair(resultVectArr, resultVectVect);
}

TRBFN::TRBFN()
{

}

TRBFN::TRBFN(int neurons_count, int categories_count)
{
	this->neurons_count = neurons_count;
	this->categories_count = categories_count;
	
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0, 1);
	
	//Generating random value for mu, beta, W

	std::vector<double> addVect;
	for (int i = 0; i < neurons_count; i++)
	{
		mu.push_back({dis(gen), dis(gen)});
		beta.push_back(dis(gen));
		addVect.clear();
		for (int j = 0; j < categories_count; j++)
			addVect.push_back(dis(gen));

		W.push_back(addVect);
	}
}

void TRBFN::
learn(std::vector<std::array<double, 3>>& learnSet)
{	
	configure_mu_beta(learnSet);
	configure_W(learnSet);
}

std::vector<int> 
TRBFN::category(std::vector<std::array<double, 2>>& dataSet)
{
	std::vector<int> result;
	for each (auto vector in dataSet)
	{
		result.push_back(category(vector));
	}
	return result;
}

int 
TRBFN::category(std::array<double, 2>& vector)
{
	std::vector<double> network_output = output(vector);
	double max = network_output[0];
	int nMax = 0;
	for (int category = 0; category < categories_count; category++)
	{
		if (network_output[category] > max)
		{
			max = network_output[category];
			nMax = category;
		}
	}
	return nMax;
}

std::vector<double> 
TRBFN::output(std::array<double, 2>& testVect)
{
	std::vector<double> result;
	std::vector<double> act_func = activation_function(testVect);
	double sum;
	for (int outNum = 0; outNum < categories_count; outNum++)
	{
		sum = 0;
		for (int neuronNum = 0; neuronNum < neurons_count; neuronNum++)
		{
			sum += W[neuronNum][outNum] * act_func[neuronNum];
		}

		sum += 1; //bias(зміщення)
		result.push_back(sum);
	}
	return result;
}

std::vector<std::vector<double>> 
TRBFN::output(std::vector<std::array<double, 2>>& testSet)
{
	std::vector<std::vector<double>> result;
	for each (std::array<double, 2> vector in testSet)
	{
		result.push_back(output(vector));
	}
	return result;
}

std::vector<double> 
TRBFN::activation_function(std::array<double, 2> & vector)
{
	std::vector<double> result;
	std::array<double, 2> difference;
	for (int i = 0; i < neurons_count; i++)
	{
		difference = { vector[0] - mu[i][0], vector[1] - mu[i][1]};
		result.push_back(exp(-pow((euclidean_distance(difference)), 2.) / (2 * beta[i])));
	}
	return result;	
}

std::vector<std::array<double, 2>> 
TRBFN::line(std::pair<double, double>& x_range, std::pair<double, double>& y_range, double presicion)
{
	return std::vector<std::array<double, 2>>();
}

std::array<double, 2> 
TRBFN::vects_diff(std::array<double, 2>& a, std::array<double, 2>& b)
{
	return {a[0] - b[0], a[1] - b[1]};
}

std::array<double, 2> 
TRBFN::vects_sum(std::array<double, 2>& a, std::array<double, 2>& b)
{
	return { a[0] + b[0], a[1] + b[1] };
}

double 
TRBFN::vects_mult(std::array<double, 2>& a, std::array<double, 2>& b)
{
	return (a[0] * b[0]) + (a[1] * b[1]);
}

double 
TRBFN::network_error(std::vector<std::array<double, 2>> & testSet, std::vector<std::vector<double>>& tempSet)
{
	debugCheck();
	double result = 0;
	std::vector<std::vector<double>> network_output = output(testSet);
	for (int nTest = 0; nTest < (int)testSet.size(); nTest++)
	{
		for (int nOut = 0; nOut < categories_count; nOut++)
		{
			result += pow(tempSet[nTest][nOut] - network_output[nTest][nOut], 2.);
		}
	}
	return result;
}

void 
TRBFN::debugCheck()
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
