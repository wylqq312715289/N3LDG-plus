#include "AtomicOP.h"

PExecutor TanhNode::generate() {
  return new ActivationExecutor<ActivatedEnum::TANH>;
}

PExecutor DropoutNode::generate() {
  DropoutExecutor *exec = new DropoutExecutor();
  exec->batch.push_back(this);
  exec->is_training = isTraning();
  exec->drop_value = drop_value_;
  exec->dim = getDim();
  return exec;
}

Executor *MaxScalarNode::generate() {
  MaxScalarExecutor *executor = new MaxScalarExecutor();
  return executor;
}

Executor *ScalarToVectorNode::generate() {
  ScalarToVectorExecutor *executor = new ScalarToVectorExecutor();
  return executor;
}

Executor *SumNode::generate() {
  SumExecutor *e = new SumExecutor();
  return e;
}

Executor *ScaledNode::generate() {
  return new ScaledExecutor;
}

namespace n3ldg_plus {

Node *maxScalar(Graph &graph, Node &input) {
  MaxScalarNode *node = MaxScalarNode::newNode(1);
  node->forward(graph, input);
  return node;
}

Node *tanh(Graph &graph, Node &input) {
  TanhNode *result = TanhNode::newNode(input.getDim());
  result->forward(graph, input);
  return result;
}

Node *sigmoid(Graph &graph, Node &input) {
  SigmoidNode *result = SigmoidNode::newNode(input.getDim());
  result->forward(graph, input);
  return result;
}

Node *relu(Graph &graph, Node &input) {
  ReluNode *result = ReluNode::newNode(input.getDim());
  result->forward(graph, input);
  return result;
}

Node *sqrt(Graph &graph, Node &input) {
  SqrtNode *result = SqrtNode::newNode(input.getDim());
  result->forward(graph, input);
  return result;
}

Node *scalarToVector(Graph &graph, int dim, Node &input) {
  ScalarToVectorNode *node = ScalarToVectorNode::newNode(dim);
  node->forward(graph, input);
  return node;
}

Node *vectorSum(Graph &graph, Node &input) {
  SumNode *sum = SumNode::newNode(1);
  sum->forward(graph, input);
  return sum;
}

Node *exp(Graph &graph, Node &input) {
  ExpNode *node = ExpNode::newNode(input.getDim());
  node->forward(graph, input);
  return node;
}

Node *dropout(Graph &graph, Node &input, dtype dropout, bool is_training) {
  DropoutNode *node = DropoutNode::newNode(input.getDim());
  node->setIsTraining(is_training);
  node->setDropValue(dropout);
  node->forward(graph, input);
  return node;
}

Node *scaled(Graph &graph, Node &input, dtype factor) {
  ScaledNode *node = ScaledNode::newNode(input.getDim());
  node->setFactor(factor);
  node->forward(graph, input);
  return node;
}

}
