#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <string>

template <class T>
class csv_istream_iterator :
	public std::iterator<std::input_iterator_tag, T>
{
	std::istream * _input;
	char _delim;
	std::string _value;
public:
	csv_istream_iterator(char delim = ',') : _input(0), _delim(delim) {}
	csv_istream_iterator(std::istream & in, char delim = ',') : _input(&in), _delim(delim) { ++*this; }

	const T operator *() const {
		std::istringstream ss(_value);
		T value;
		ss >> value;
		return value;
	}

	std::istream & operator ++() {
		if (!(getline(*_input, _value, _delim)))
		{
			_input = 0;
		}
		return *_input;
	}

	bool operator !=(const csv_istream_iterator & rhs) const {
		return _input != rhs._input;
	}

	bool operator ==(const csv_istream_iterator & rhs) const {
		return _input == rhs._input;
	}
};

