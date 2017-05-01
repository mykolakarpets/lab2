#include "TRBFN.h"

void 
TRBFN::configure_mu_beta(std::vector<std::array<double, 3>> & learnSet)
{
	mu.clear();
	beta.clear();
	for (int category = 0; category < categories_count; category++)
	{
		std::vector<std::array<double, 2>> convertedData;

		for each (auto learnVect in learnSet)
			if((int)learnVect[2] == category)
				convertedData.push_back({ learnVect[0], learnVect[1] });

		auto resultOfClustering = dkm::kmeans_lloyd(convertedData, neurons_count / categories_count);

		//Filling mu
	
		for each (auto mean in std::get<0>(resultOfClustering))
		{
			mu.push_back(mean);
		}
		//Counting betas
		
		int m;
		double sum;
		for (int betaNum = (category * (neurons_count / categories_count)); betaNum < ((category+1) * (neurons_count / categories_count)); betaNum++)
		{
			m = 0;
			sum = 0;
			//std::cout << "convertedData.size(): "<<(int)convertedData.size() << std::endl;
			for (int dataNum = 0; dataNum < (int)convertedData.size(); dataNum++)
			{
				if (std::get<1>(resultOfClustering)[dataNum] == (betaNum - (category * (neurons_count / categories_count))))
				{
					m++;
					sum += euclidean_distance({ convertedData[dataNum][0] - mu[betaNum][0], convertedData[dataNum][1] - mu[betaNum][1] });
				}
			}
			//PRINT(beta[betaNum]);
			beta.push_back( 1 / (2 * pow((double)(sum / m), 2.)));
			//PRINT(beta[betaNum]);
		}
		

	}
}

void 
TRBFN::configure_W(std::vector<std::array<double, 3>>& learnSet, int outNum)
{
	std::pair<std::vector<std::array<double, 2>>, std::vector<std::vector<double>>> transformedLearnSet = getLearnSet(learnSet);
	std::vector<std::array<double, 2>> learnVectors = transformedLearnSet.first;
	std::vector<std::vector<double>> learnOutputs = transformedLearnSet.second;

	double sum;
	std::vector<double> actFuncRes;
	std::vector<double> networkRes;
	//out_error(learnVectors, learnOutputs);
	int count = 0;
	while ((network_error(learnVectors, learnOutputs, outNum) != 0) && (count++ < MAX_ITERATIONS ))
	{
		//out_w_matrix();
		out_error(learnVectors, learnOutputs, outNum);
		for (int neuronNum = 0; neuronNum < neurons_count; neuronNum++)
		{
			sum = 0;
			for (int learnNum = 0; learnNum < (int)learnVectors.size(); learnNum++)
			{
				actFuncRes = activation_function(learnVectors[learnNum]);

				networkRes = normalized_output(learnVectors[learnNum]);

				sum += actFuncRes[neuronNum] * (learnOutputs[learnNum][outNum] - networkRes[outNum]);
				//std::cout << "sum += " << actFuncRes[neuronNum] << " * (" << learnOutputs[learnNum][outNum] << " - " << networkRes[outNum] << ")" << std::endl;
			}
			W[neuronNum][outNum] += sum * LEARNING_COEF;
			//if(outNum == 1) PRINT((sum * LEARNING_COEF));
			bias[outNum] += sum * LEARNING_COEF;
			
		}
	}
}

std::vector<double> TRBFN::unnorm_out(std::vector<double>& netOut)
{
	std::vector<double> result;
	double sum_exps = 0;
	for each (auto var in netOut)
	{
		sum_exps += exp(var);
	}

	for each (auto var in netOut)
	{
		result.push_back(-log((var / sum_exps)));
	}

	return result;
}

std::vector<std::array<double, 2>> TRBFN::getMu()
{
	return mu;
}

std::vector<std::vector<double>> TRBFN::getW()
{
	std::vector<std::vector<double>> result = W;
	result.push_back(bias);
	return result;
}

std::vector<double> TRBFN::getBeta()
{
	return beta;
}

std::pair<std::vector<std::array<double, 2>>, std::vector<std::vector<double>>> 
TRBFN::getLearnSet(std::vector<std::array<double, 3>>& learnSet)
{
	std::pair<std::vector<std::array<double, 2>>, std::vector<std::vector<double>>> resultPair;
	std::vector<std::array<double, 2>> resultVectArr;
	std::vector<std::vector<double>> resultVectVect;
	for each (std::array<double, 3> var in learnSet)
	{
		resultVectArr.push_back({ var[0],var[1] });
		std::vector<double> resultVect;
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

	for (int j = 0; j < categories_count; j++)
		bias.push_back(dis(gen));
}

TRBFN::TRBFN(std::vector<std::vector<double>> W, std::vector<double> bias)
{
	this->W = W;
	this->bias = bias;
}

void TRBFN::
learn(std::vector<std::array<double, 3>> &learnSet)
{	
	configure_mu_beta(learnSet);
	for (int i = 0; i < categories_count; i++)
	{
		configure_W(learnSet, i);
	}

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
	network_output = unnorm_out(network_output);
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
			sum += W[neuronNum][outNum] *act_func[neuronNum];
		}

		sum += bias[outNum]; //bias(зміщення)
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

std::vector<double> TRBFN::normalized_output(std::array<double, 2>& testVect)
{
	int out = category(testVect);
	std::vector<double> result;
	for (int i = 0; i < categories_count; i++)
	{
		result.push_back(i == out ? 1 : 0);
	}
	return result;
}

std::vector<std::vector<double>> TRBFN::normalized_output(std::vector<std::array<double, 2>>& testSet)
{
	std::vector<std::vector<double>> result;
	for each (std::array<double, 2> vector in testSet)
	{
		result.push_back(normalized_output(vector));
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
TRBFN::network_error(std::vector<std::array<double, 2>> & testSet, std::vector<std::vector<double>>& tempSet, int category)
{
	double result = 0;
	std::vector<std::vector<double>> network_output = normalized_output(testSet);

	for (int nTest = 0; nTest < (int)testSet.size(); nTest++)
	{
		if(tempSet[nTest][category])
		result += pow((int)tempSet[nTest][category] - (int)network_output[nTest][category], 2.);
		
	}
	return result;
}

void 
TRBFN::out_w_matrix()
{
#ifdef OUT_W_MATRIX
	using namespace std;

	cout << "W matrix: " << endl;
	for each (auto row in W)
	{
		for each (auto elem in row)
		{
			cout << setw(10) << elem;
		}
		cout << endl;
	}

#endif

}

void TRBFN::out_error(std::vector<std::array<double, 2>> & testSet, std::vector<std::vector<double>>& tempSet, int category)
{
#ifdef OUT_ERROR
	std::cout << category<<":"<<network_error(testSet, tempSet, category) << std::endl;

#endif
}

TRBFN::~TRBFN()
{
}
