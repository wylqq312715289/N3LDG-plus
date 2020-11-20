#include "Split.h"

namespace n3ldg_plus {
Node *split(Graph &graph, int dim, Node &input, int offset) {
  SplitNode *split = SplitNode::newNode(dim);
  split->forward(graph, input, offset);
  return split;
}
}

Executor *SplitNode::generate() {
  SplitExecutor *executor = new SplitExecutor;
  executor->batch.push_back(this);
  return executor;
}

