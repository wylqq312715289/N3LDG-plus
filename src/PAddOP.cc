#include "PAddOP.h"

namespace n3ldg_plus {
Node *add(Graph &graph, vector<Node *> inputs) {
  int dim = inputs.front()->getDim();
  PAddNode *result = PAddNode::newNode(dim);
  result->forward(graph, inputs);
  return result;
}
}

PExecutor PAddNode::generate() {
  PAddExecutor *exec = new PAddExecutor();
  exec->batch.push_back(this);
  exec->in_count = ins.size();
  exec->dim = getDim();
  return exec;
}

