#include "Concat.h"

PExecutor ConcatNode::generate() {
  ConcatExecutor *exec = new ConcatExecutor();
  exec->batch.push_back(this);
#if USE_GPU
  exec->inCount = this->ins.size();
  exec->outDim = 0;
  for (int d : inDims) {
      exec->outDim += d;
  }
#endif
  return exec;
}

Executor *ScalarConcatNode::generate() {
  return new ScalarConcatExecutor;
}

namespace n3ldg_plus {

Node *concat(Graph &graph, vector<Node *> inputs) {
  int dim = 0;
  for (Node *in : inputs) {
    dim += in->getDim();
  }
  ConcatNode *concat = ConcatNode::newNode(dim);
  concat->forward(graph, inputs);
  return concat;
}

Node *scalarConcat(Graph &graph, vector<Node *> inputs) {
  ScalarConcatNode *concat = ScalarConcatNode::newNode(inputs.size());
  concat->forward(graph, inputs);
  return concat;
}

}

