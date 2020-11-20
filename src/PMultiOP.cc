#include "PMultiOP.h"

Executor *PMultiNode::generate() {
  PMultiExecutor *exec = new PMultiExecutor();
  exec->batch.push_back(this);
  exec->dim = getDim();
  return exec;
};

namespace n3ldg_plus {

Node *pointwiseMultiply(Graph &graph, Node &a, Node &b) {
  if (a.getDim() != b.getDim()) {
    cerr << boost::format("a dim:%1% b dim:%2%") % a.getDim() % b.getDim() << endl;
    abort();
  }
  PMultiNode *node = PMultiNode::newNode(a.getDim());
  node->forward(graph, a, b);
  return node;
}

}
