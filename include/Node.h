#ifndef BasicNode
#define BasicNode

#include <iomanip>
#include <functional>
#include <string>
#include <tuple>
#include <memory>
#include <utility>
#include <vector>
#include <set>
#include "MyTensor-def.h"
#include "MyLib.h"
#include "profiler.h"
#include <boost/format.hpp>
#if USE_GPU
#include "N3LDG_cuda.h"
using n3ldg_cuda::Tensor1D;
using n3ldg_cuda::Tensor2D;
#else
using n3ldg_cpu::Tensor1D;
using n3ldg_cpu::Tensor2D;
#endif

dtype fexp(const dtype &x);

dtype flog(const dtype &x);

dtype dequal(const dtype &x, const dtype &y);

dtype dtanh(const dtype &x, const dtype &y);

dtype dleaky_relu(const dtype &x, const dtype &y);

dtype dselu(const dtype &x, const dtype &y);

dtype dsigmoid(const dtype &x, const dtype &y);

dtype drelu(const dtype &x, const dtype &y);

dtype dexp(const dtype &x, const dtype &y);

dtype dlog(const dtype &x, const dtype &y);

dtype dsqrt(dtype y);

//useful functions
dtype fequal(const dtype &x);

dtype ftanh(const dtype &x);

dtype fsigmoid(const dtype &x);

dtype frelu(const dtype &x);

dtype fleaky_relu(const dtype &x);

dtype fselu(const dtype &x);

dtype fsqrt(const dtype &x);

class Executor;
class Node;

class NodeContainer {
 public:
  virtual void addNode(Node *node) = 0;
};

string addressToString(const void *p);

class Node {
 public:
#if USE_GPU
  virtual void initOnHostAndDevice(int ndim) {
      dim_ = ndim;
      val_.initOnMemoryAndDevice(ndim);
      loss_.initOnMemoryAndDevice(ndim);
      n3ldg_cuda::Memset(val_.value, dim_, 0.0f);
      n3ldg_cuda::Memset(loss_.value, dim_, 0.0f);
  }
#endif

  virtual void compute() = 0;
  virtual void backward() = 0;

  virtual Executor *generate() = 0;

  virtual bool typeEqual(Node *other) {
    if (node_type_.compare(other->node_type_) != 0) {
      return false;
    }
    if (dim_ != other->dim_) {
      return false;
    }
    return true;
  }

  virtual string typeSignature() const {
    return node_type_ + "-" + std::to_string(dim_);
  }

  virtual void addParent(Node *parent) {
    if (degree_ >= 0) {
      parents_.push_back(parent);
      parent->degree_++;
      parent->depth_ = std::max(depth_ + 1, parent->depth_);
    }
  }

  const Tensor1D &getVal() const {
    return val_;
  }

  Tensor1D &val() {
    return val_;
  }

  const Tensor1D &getLoss() const {
    return loss_;
  }

  Tensor1D &loss() {
    return loss_;
  }

  int getDim() const {
    return dim_;
  }

  int getDegree() const {
    return degree_;
  }

  void setDegree(int degree) {
    degree_ = degree;
  }

  const string &getNodeName() const {
    return node_name_;
  }

  void setNodeName(const string &node_name) {
    node_name_ = node_name;
  }

  void setNodeIndex(int node_index) {
    node_index_ = node_index;
  }

  int getNodeIndex() const {
    return node_index_;
  }

  virtual int getDepth() const {
    return depth_;
  }

  const string &getNodeType() const {
    return node_type_;
  }

  const vector<Node *> getParents() const {
    return parents_;
  }

  virtual void clear() {
    parents_.clear();
    degree_ = 0;
    depth_ = 0;
#if !USE_GPU || TEST_CUDA
    loss_.zero();
#endif
  }

  Node(const Node &) = delete;

  virtual ~Node() = default;

  int getColumn() const {
    if (column_ == 0) {
      cerr << "Node getColumn - column is 0" << endl;
      abort();
    }
    return column_;
  }

  int getRow() const {
    return getDim() / column_;
  }

  Mat valMat() {
    if (column_ == 0) {
      cerr << "Node valMat - column is 0" << endl;
      abort();
    }
    return Mat(val().v, getRow(), column_);
  }

  Mat gradMat() {
    if (column_ == 0) {
      cerr << "Node gradMat - column is 0" << endl;
      abort();
    }
    return Mat(loss().v, getRow(), column_);
  }

  void setColumn(int column) {
    if (getDim() % column != 0) {
      cerr << boost::format("MatrixNode setColumn - dim:%1% column:%2%") % getDim() % column
           << endl;
      abort();
    }
    column_ = column;
  }

 protected:
  void afterForward(NodeContainer &container, vector<Node *> &ins) {
    for (Node *in : ins) {
      in->addParent(this);
    }
    container.addNode(this);
  }

  Node(const string &node_type, int dim = 0) : dim_(dim) {
    degree_ = 0;
    node_type_ = node_type;
  }

  virtual void setDim(int dim) {
    dim_ = dim;
  }

  virtual void init(int ndim) {
    if (ndim <= 0) {
      cerr << boost::format("Node init - dim is less than 0:%1% type:%2%") % ndim %
          getNodeType() << endl;
      abort();
    }
    dim_ = ndim;
    val_.init(dim_);
    loss_.init(dim_);
  }

 private:
  std::vector<Node *> parents_;
  Tensor1D val_;
  Tensor1D loss_;
  int dim_;
  int degree_ = 0;
  int depth_ = 0;
  string node_type_;
  string node_name_;
  int node_index_;
  int column_ = 0;
};

set<pair<vector<Node *>, int> *> &globalPoolReferences();

bool &globalPoolEnabled();

bool &globalLimitedDimEnabled();

int NextTwoIntegerPowerNumber(int number);

template<typename T>
class Poolable {
 public:
  static T *newNode(int key) {
    if (!globalPoolEnabled()) {
      T *node = new T;
      node->initNode(key);
      return node;
    }
    int original_key = key;
    if (globalLimitedDimEnabled()) {
      key = NextTwoIntegerPowerNumber(key);
    }
    auto it = pool_.find(key);
    if (it == pool_.end()) {
      pool_.insert(make_pair(key, make_pair(vector<Node *>(), 0)));
      it = pool_.find(key);
      globalPoolReferences().insert(&it->second);
    }
    auto &p = it->second;
    vector<Node *> &v = p.first;
    T *node;
    if (p.second > v.size()) {
      abort();
    } else if (v.size() == p.second) {
      node = new T;
      node->initNode(key);
      node->setNodeDim(original_key);
      v.push_back(node);
      ++p.second;
    } else {
      node = static_cast<T *>(v.at(p.second));
      node->setNodeDim(original_key);
      ++p.second;
      Node *n = static_cast<Node *>(node);
      n->clear();
    }
    return node;
  }

  virtual void initNode(int dim) = 0;
  virtual void setNodeDim(int dim) = 0;

 private:
  static map<int, pair<vector<Node *>, int>> pool_;
};

template<typename T>
map<int, pair<vector<Node *>, int>> Poolable<T>::pool_;

void validateEqualNodeDims(const vector<Node *> &nodes);

auto cpu_get_node_val = [](Node *node) {
  return node->val().v;
};

auto cpu_get_node_loss = [](Node *node) {
  return node->loss().v;
};

#if USE_GPU

auto gpu_get_node_val = [](Node *node) {
    return node->val().value;
};

auto gpu_get_node_loss = [](Node *node) {
    return node->loss().value;
};

#endif

typedef Node *PNode;

class UniInputNode : public Node {
 public:
  UniInputNode(const string &node_type) : Node(node_type) {}

  virtual bool typeEqual(Node *other) override {
    UniInputNode *o = static_cast<UniInputNode *>(other);
    return Node::typeEqual(other) && input_->getDim() == o->input_->getDim();
  }

  virtual string typeSignature() const override {
    return Node::typeSignature() + "-" + to_string(input_->getDim());
  }

  void forward(NodeContainer &container, Node &input) {
    if (!isDimLegal(input)) {
      cerr << boost::format("dim:%1% input dim:%2%") % Node::getDim() % input.getDim() <<
           endl;
      abort();
    }
    input_ = &input;
    vector<Node *> ins = {input_};
    Node::afterForward(container, ins);
  }

  Node *getInput() const {
    return input_;
  }

 protected:
  virtual bool isDimLegal(const Node &input) const = 0;

 private:
  Node *input_;
  friend class UniInputExecutor;
};

template<typename T>
std::vector<Node *> toNodePointers(const std::vector<T *> &vec) {
  std::vector<Node *> results;
  for (T *p : vec) {
    results.push_back(p);
  }
  return results;
}

/* *
 * return tuple<exp, pair<max_i, max>, sum>
 * */
std::tuple<std::unique_ptr<n3ldg_cpu::Tensor1D>, std::pair<int, dtype>, dtype> toExp(const Node &node);

#if USE_GPU
void clearNodes(std::vector<Node*> &nodes);
#endif

class Executor {
 public:
  std::vector<PNode> batch;
  virtual ~Executor() = default;

#if USE_GPU
  vector<dtype *> getVals() {
      vector<dtype *> vals;
      for (Node * node : batch) {
          vals.push_back(node->getVal().value);
      }
      return vals;
  }

  vector<dtype *> getGrads() {
      vector<dtype *> grads;
      for (Node * node : batch) {
          grads.push_back(node->getLoss().value);
      }
      return grads;
  }
#else
  virtual int calculateFLOPs() = 0;

  virtual int calculateActivations() {
    int sum = 0;
    for (Node *node : batch) {
      sum += node->getDim();
    }
    return sum;
  }
#endif

  int getDim() const {
    return batch.at(batch.size() - 1)->getDim();
  }

  string getNodeType() const {
    return batch.front()->getNodeType();
  }

  string getSignature() const {
    return batch.front()->typeSignature();
  }

  int getCount() const {
    return batch.size();
  }

  void forwardFully() {
    Node *first = batch.front();
    for (int i = 1; i < batch.size(); ++i) {
      if (!first->typeEqual(batch.at(i))) {
        cerr << "type not equal in the same batch - node_type:" << first->getNodeType() <<
             endl;
        abort();
      }
    }

    n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
    profiler.BeginEvent(batch.front()->getNodeType() + " forward");
    forward();

    profiler.EndCudaEvent();
    for (Node *node : batch) {
      node->setDegree(-1);
    }
  }

  void backwardFully() {
    n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
    profiler.BeginEvent(getNodeType() + " backward");
    backward();
    profiler.EndEvent();
  }

  virtual void backward() {
    for (Node *node : batch) {
      node->backward();
    }
  }

  virtual bool addNode(PNode in) {
    if (in == nullptr) {
      cerr << "in is nullptr" << endl;
      abort();
    }
    if (batch.empty()) {
      return false;
    }

    if (batch[0]->typeEqual(in)) {
      batch.push_back(in);
      return true;
    }

    return false;
  }

 protected:
  virtual void forward() {
    for (Node *node : batch) {
      node->compute();
    }
  }

  int defaultFLOPs() {
    int sum = 0;
    for (Node *node : batch) {
      sum += node->getDim();
    }
    return sum;
  }

#if TEST_CUDA
  void testForward() {
      Executor::forward();

      for (Node *node : batch) {
          n3ldg_cuda::Assert(node->getVal().verify((getNodeType() + " forward").c_str()));
      }
  }

  void testForwardInpputs(const function<vector<Node*>(Node &node)> &get_inputs) {
      for (Node *node : batch) {
          vector<Node*> inputs = get_inputs(*node);
          for (Node *input : inputs) {
              n3ldg_cuda::Assert(input->getVal().verify((getNodeType() +
                              " forward input").c_str()));
          }
      }
  }

  void testBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
      Executor::backward();

      for (Node *node : batch) {
          auto inputs = get_inputs(*node);
          for (pair<Node*, string> &input : inputs) {
              n3ldg_cuda::Assert(input.first->getLoss().verify((getNodeType() +
                              " backward " + input.second).c_str()));
          }
      }
  }

  void testBeforeBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
      for (Node *node : batch) {
          auto inputs = get_inputs(*node);
          for (pair<Node*, string> &input : inputs) {
              n3ldg_cuda::Assert(input.first->getLoss().verify((getNodeType() +
                              " backward " + input.second + " node index:" + to_string(input.first->getNodeIndex())).c_str()));
          }
      }
  }
#endif
};

auto get_inputs = [](Node &node) {
  UniInputNode &uni_input = static_cast<UniInputNode &>(node);
  vector<Node *> inputs = {uni_input.getInput()};
  return inputs;
};

class UniInputExecutor : public Executor {
 protected:
#if TEST_CUDA
  void testForwardInpputs() {
      for (Node *node : batch) {
          vector<Node*> inputs = get_inputs(*node);
          for (Node *input : inputs) {
              n3ldg_cuda::Assert(input->getVal().verify((getNodeType() +
                              " forward input").c_str()));
          }
      }
  }

  void testBeforeBackward() {
      auto get_inputs = [](Node &node) {
          UniInputNode &uni_input = static_cast<UniInputNode&>(node);
          vector<pair<Node*, string>> inputs = {make_pair(uni_input.input_, "input")};
          return inputs;
      };
      Executor::testBeforeBackward(get_inputs);
  }

  void testBackward() {
      auto get_inputs = [](Node &node) {
          UniInputNode &uni_input = static_cast<UniInputNode&>(node);
          vector<pair<Node*, string>> inputs = {make_pair(uni_input.input_, "input")};
          return inputs;
      };
      Executor::testBackward(get_inputs);
  }
#endif
};

typedef Executor *PExecutor;

#if USE_GPU

typedef dtype N3LDGActivated(const dtype &x);
ActivatedEnum ToActivatedEnum(N3LDGActivated func);
#endif

#endif
