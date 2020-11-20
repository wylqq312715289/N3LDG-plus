#include "Pooling.h"

PExecutor PoolNode::generate() {
  PoolExecutor *exec = new PoolExecutor();
  exec->batch.push_back(this);
  return exec;
}

PExecutor SumPoolNode::generate() {
  SumPoolExecutor *exec = new SumPoolExecutor();
  exec->batch.push_back(this);
#if USE_GPU
  exec->dim = getDim();
#endif
  return exec;
}

PExecutor AvgPoolNode::generate() {
  AvgPoolExecutor *exec = new AvgPoolExecutor();
  exec->batch.push_back(this);
#if USE_GPU
  exec->dim = getDim();
#endif
  return exec;
}

namespace n3ldg_plus {
Node *maxPool(Graph &graph, vector<Node *> &inputs) {
  int dim = inputs.front()->getDim();
  for (int i = 1; i < inputs.size(); ++i) {
    if (dim != inputs.at(i)->getDim()) {
      cerr << "dim not equal" << endl;
      abort();
    }
  }

  MaxPoolNode *pool = MaxPoolNode::newNode(dim);
  pool->forward(graph, inputs);
  return pool;
}

Node *sumPool(Graph &graph, vector<Node *> &inputs) {
  int dim = inputs.front()->getDim();
  SumPoolNode *pool = SumPoolNode::newNode(dim);
  pool->forward(graph, inputs);
  return pool;
}

Node *averagePool(Graph &graph, vector<Node *> &inputs) {
  int dim = inputs.front()->getDim();
  AvgPoolNode *pool = AvgPoolNode::newNode(dim);
  pool->forward(&graph, inputs);
  return pool;
}
}

