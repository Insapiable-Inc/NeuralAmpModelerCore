#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace nam
{
// Read cursor over a model's flat weight vector. Layers consume it with
// `*(weights++)`, exactly as with a `std::vector<float>::iterator`, but reading past
// the last weight throws instead of reading out of bounds — a truncated or damaged
// model file becomes a load error, not memory corruption.
class weights_iterator
{
public:
  explicit weights_iterator(const std::vector<float>& weights)
  : _data(weights.data())
  , _size(weights.size())
  {
  }

  float operator*() const
  {
    if (_pos >= _size)
      throw std::runtime_error("Model weights are truncated: the file has " + std::to_string(_size)
                               + " weights, fewer than its architecture needs.");
    return _data[_pos];
  }

  weights_iterator& operator++()
  {
    ++_pos;
    return *this;
  }

  weights_iterator operator++(int)
  {
    weights_iterator old = *this;
    ++_pos;
    return old;
  }

  // Every weight consumed (no more, no fewer).
  bool at_end() const { return _pos == _size; }
  size_t consumed() const { return _pos; }
  size_t size() const { return _size; }

  // Throws unless exactly every weight was consumed.
  void expect_end(const char* model) const
  {
    if (_pos != _size)
      throw std::runtime_error(std::string("Weight mismatch in ") + model + ": the file has " + std::to_string(_size)
                               + " weights, the architecture uses " + std::to_string(_pos) + ".");
  }

private:
  const float* _data;
  size_t _size;
  size_t _pos = 0;
};
} // namespace nam
