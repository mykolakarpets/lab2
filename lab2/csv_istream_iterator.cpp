#include "csv_istream_iterator.h"

template <>
const std::string csv_istream_iterator<std::string>::operator *() const {
	return _value;
}
