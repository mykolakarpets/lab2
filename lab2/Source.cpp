#include <iostream>
#include <fstream>
#include <vector>
#include <fstream>
#include <istream>
#include <string>
#include "TRBFN.h"

int main(int argc, char ** argv)
{

	std::ifstream fin("dataset.csv");
	std::vector<std::array<double, 3>> learnSet;
	std::string x, y, category;
	while (!fin.eof())
	{
		x = "";
		y = "";
		category = "";
		std::getline(fin, x, ',');
		std::getline(fin, y, ',');
		std::getline(fin, category);

		learnSet.push_back({ std::stod(x),std::stod(y), (double)(std::stoi(category) - 1) });
	}

	TRBFN network(20, 2);
	network.learn(learnSet);

	std::ofstream out("means.txt");
	for each (auto var in network.getMu())
	{
		out << var[0] << "," << var[1] << std::endl;
	}

	std::ofstream outW("weights.txt");
	auto configured_network = network.getW();
	for each (auto var in configured_network)
	{
		for (int i = 0; i < var.size(); i++)
		{
			outW << var[i];
			if (i != var.size() - 1) outW << ",";
		}
		outW << std::endl;
	}

	std::ofstream outBeta("betas.txt");
	auto betas = network.getBeta();
	for (int i = 0; i < betas.size(); i++)
	{
		outBeta << betas[i];
		if (i != betas.size() - 1) outBeta << std::endl;

	}

	return 0;
}