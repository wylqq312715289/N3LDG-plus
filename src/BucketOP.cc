#include "BucketOP.h"

namespace n3ldg_plus {

Node *bucket(Graph &graph, int dim, float v) {
  BucketNode *bucket = BucketNode::newNode(dim);
  bucket->forward(graph, v);
  return bucket;
}

Node *bucket(Graph &graph, const vector<float> &v) {
  BucketNode *bucket = BucketNode::newNode(v.size());
  bucket->forward(graph, v);
  return bucket;
}

}

PExecutor BucketNode::generate() {
  BucketExecutor *exec = new BucketExecutor();
  exec->batch.push_back(this);
  return exec;
}

