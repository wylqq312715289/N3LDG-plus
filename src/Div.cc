#include "Div.h"

namespace n3ldg_plus {
Node *div(Graph &graph, Node &numerator, Node &denominator) {
  DivNode *result = DivNode::newNode(numerator.getDim());
  result->forward(graph, numerator, denominator);
  return result;
}
}

Executor *DivNode::generate() {
  DivExecutor *executor = new DivExecutor();
  return executor;
}

Executor *FullDivNode::generate() {
  return new FullDivExecutor();
}

namespace n3ldg_plus {
Node *fullDiv(Graph &graph, Node &numerator, Node &denominator) {
  FullDivNode *result = FullDivNode::newNode(numerator.getDim());
  result->forward(graph, numerator, denominator);
  return result;
}
}
