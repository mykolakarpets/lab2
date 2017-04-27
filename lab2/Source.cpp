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

		learnSet.push_back({ std::stod(x),std::stod(y), std::stod(category) });
	}

	TRBFN network(20, 2);
	network.learn(learnSet);



	return 0;
}