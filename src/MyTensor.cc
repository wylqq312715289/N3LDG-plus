#include "MyTensor.h"

//useful functions
dtype fequal(const dtype &x) {
  return x;
}

dtype ftanh(const dtype &x) {
  return tanh(x);
}

dtype fsigmoid(const dtype &x) {
  return 1.0 / (1.0 + exp(-x));
}

dtype frelu(const dtype &x) {
  if (x <= 0) return 0;
  return x;
}

dtype fleaky_relu(const dtype &x) {
  if (x < 0) return (0.1 * x);
  return x;
}

dtype fselu(const dtype &x) {
  dtype lambda = 1.0507009873554804934193349852946;
  dtype alpha = 1.6732632423543772848170429916717;
  if (x <= 0) return lambda * alpha * (exp(x) - 1);
  return lambda * x;
}

dtype fexp(const dtype &x) {
  return exp(x);
}

dtype flog(const dtype &x) {
  return log(x);
}

//derive function
dtype dequal(const dtype &x, const dtype &y) {
  return 1;
}

dtype dtanh(const dtype &x, const dtype &y) {
  return (1 + y) * (1 - y);
}

dtype dleaky_relu(const dtype &x, const dtype &y) {
  if (x < 0) return 0.1;
  return 1;
}

dtype dselu(const dtype &x, const dtype &y) {
  dtype lambda = 1.0507009873554804934193349852946;
  dtype alpha = 1.6732632423543772848170429916717;
  if (x <= 0) return lambda * alpha + y;
  return lambda;
}

dtype dsigmoid(const dtype &x, const dtype &y) {
  return (1 - y) * y;
}

dtype drelu(const dtype &x, const dtype &y) {
  if (x <= 0) return 0;
  return 1;
}

dtype dexp(const dtype &x, const dtype &y) {
  return y;
}

dtype dlog(const dtype &x, const dtype &y) {
  if (x < 0.001) return 1000;
  return 1.0 / x;
}

