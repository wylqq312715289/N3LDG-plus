#include "Sub.h"

namespace n3ldg_plus {

Node *sub(Graph &graph, Node &minuend, Node &subtrahend);

}

Executor *SubNode::generate() {
  SubExecutor *executor = new SubExecutor();
  return executor;
}
