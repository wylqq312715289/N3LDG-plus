#ifndef BucketOP
#define BucketOP

/*
*  BucketOP.h:
*  a bucket operation, for padding mainly
*  usually an inputleaf node, degree = 0
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/
#include <vector>
#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

using namespace Eigen;
using std::vector;

class BucketNode : public Node, public Poolable<BucketNode> {
 public:
  BucketNode() : Node("bucket") {}

  void initNode(int dim) override {
    init(dim);
  }

  void setNodeDim(int dim) override {
    setDim(dim);
  }

//    virtual void init(int ndim) override {
//#if USE_GPU
//        Node::initOnHostAndDevice(ndim);
//#else
//        Node::init(ndim);
//#endif
//    }

  void forward(Graph &graph, const vector<dtype> &input) {
    if (input.size() != getDim()) {
      cerr << boost::format("input size %1% is not equal to dim %2%") % input.size() %
          getDim() << endl;
      abort();
    }
    input_ = input;
    graph.addNode(this);
  }

  void forward(Graph &graph, dtype v) {
    vector<dtype> input;
    for (int i = 0; i < getDim(); ++i) {
      input.push_back(v);
    }
    forward(graph, input);
  }

  void forward(Graph &graph) {
    forward(graph, 0);
  }

  void compute() override {
    abort();
  }

  void backward() override {
    abort();
  }

  PExecutor generate() override;

 protected:

 private:
  vector<dtype> input_;
  friend class BucketExecutor;
};

namespace n3ldg_plus {

Node *bucket(Graph &graph, int dim, float v);

Node *bucket(Graph &graph, const vector<float> &v);
}

class BucketExecutor : public Executor {
 public:
#if !USE_GPU
  int calculateFLOPs() override {
    return 0;
  }
#endif

  void forward() override {
#if USE_GPU
    int count = batch.size();
    vector<dtype*> ys;
    vector<dtype> cpu_x;
    cpu_x.reserve(getDim() * count);
    for (Node *node : batch) {
        BucketNode *bucket = static_cast<BucketNode*>(node);
        ys.push_back(bucket->val().value);
        for (int i = 0; i < getDim(); ++i) {
            cpu_x.push_back(bucket->input_.at(i));
        }
    }
    n3ldg_cuda::BucketForward(cpu_x, count, getDim(), ys);
#if TEST_CUDA
    for (Node *node : batch) {
        BucketNode *bucket = static_cast<BucketNode*>(node);
        dtype *v = node->val().v;
        for (int i = 0; i < getDim(); ++i) {
            v[i] = bucket->input_.at(i);
        }
        n3ldg_cuda::Assert(node->val().verify("bucket forward"));
    }
#endif
#else
    for (Node *node : batch) {
      BucketNode *bucket = static_cast<BucketNode *>(node);
      node->val() = bucket->input_;
    }
#endif
  }

  void backward() override {}
};

#endif
