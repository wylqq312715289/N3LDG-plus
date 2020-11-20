#include "Node.h"

dtype fexp(const dtype &x) {
  return exp(x);
}

dtype flog(const dtype &x) {
  return log(x);
}

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

dtype dsqrt(dtype y) {
  return 0.5 / y;
}

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

dtype fsqrt(const dtype &x) {
  return sqrt(x);
}

string addressToString(const void *p) {
  std::stringstream ss;
  ss << p;
  return ss.str();
}

set<pair<vector<Node *>, int> *> &globalPoolReferences() {
  static set<pair<vector<Node *>, int> *> o;
  return o;
}

bool &globalPoolEnabled() {
  static bool pool_enabled = true;
  return pool_enabled;
}

bool &globalLimitedDimEnabled() {
  static bool enabled = false;
  return enabled;
}

int NextTwoIntegerPowerNumber(int number) {
  int result = 1;
  while (number > result) {
    result <<= 1;
  }
  return result;
}

void validateEqualNodeDims(const vector<Node *> &nodes) {
  for (int i = 1; i < nodes.size(); ++i) {
    if (nodes.at(i)->getDim() != nodes.front()->getDim()) {
      cerr << boost::format(
          "validateEqualNodeDims - first node size is %1%, but %2%st is %3%") %
          nodes.size() % i % nodes.front()->getDim() << endl;
      abort();
    }
  }
}

/* *
 * return tuple<exp, pair<max_i, max>, sum>
 * */
std::tuple<std::unique_ptr<n3ldg_cpu::Tensor1D>, std::pair<int, dtype>, dtype> toExp(const Node &node) {
  dtype max = node.getVal().v[0];
  int max_j = 0;
  for (int j = 1; j < node.getDim(); ++j) {
    if (node.getVal().v[j] > max) {
      max = node.getVal().v[j];
      max_j = j;
    }
  }

  std::unique_ptr<n3ldg_cpu::Tensor1D> exp(new n3ldg_cpu::Tensor1D);
  exp->init(node.getDim());
  exp->vec() = (node.getVal().vec() - max).exp();
  dtype sum = static_cast<Eigen::Tensor<dtype, 0>>(exp->vec().sum())(0);
  return std::make_tuple(std::move(exp), std::make_pair(max_j, max), sum);
}

#if USE_GPU
void clearNodes(std::vector<Node*> &nodes) {
    n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
    profiler.BeginEvent("clearNodes");
    std::vector<dtype*> val_and_losses;
    vector<int> dims;
    val_and_losses.reserve(2 * nodes.size());
    for (Node *n : nodes) {
        val_and_losses.push_back(n->getLoss().value);
        dims.push_back(n->getDim());
    }
    n3ldg_cuda::BatchMemset(val_and_losses, val_and_losses.size(), dims, 0.0f);
#if TEST_CUDA
    for (Node *node : nodes) {
        node->loss().verify("clearNodes");
    }
#endif
    profiler.EndEvent();
}
#endif

#if USE_GPU

ActivatedEnum ToActivatedEnum(N3LDGActivated func) {
    if (func == ftanh) {
        return ActivatedEnum::TANH;
    } else if (func == fsigmoid) {
        return ActivatedEnum::SIGMOID;
    } else if (func == frelu) {
        return ActivatedEnum::RELU;
    } else if (func == fleaky_relu) {
        return ActivatedEnum::LEAKY_RELU;
    } else if (func == fselu) {
        return ActivatedEnum::SELU;
    } else {
        abort();
    }
}

#endif

