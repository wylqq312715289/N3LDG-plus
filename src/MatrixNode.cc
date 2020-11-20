#include "MatrixNode.h"

Executor *MatrixConcatNode::generate() {
  return new MatrixConcatExecutor;
}

Executor *MatrixAndVectorPointwiseMultiNode::generate() {
  return new MatrixAndVectorPointwiseMultiExecutor;
}

Executor *MatrixColSumNode::generate() {
  return new MatrixColSumExecutor;
}

Executor *MatrixAndVectorMultiNode::generate() {
  return new MatrixAndVectorMultiExecutor;
}

namespace n3ldg_plus {

MatrixNode *concatToMatrix(Graph &graph, const vector<Node *> &inputs) {
  int input_dim = inputs.front()->getDim();
  MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
  node->forward(graph, inputs);
  return node;
}

MatrixNode *matrixPointwiseMultiply(Graph &graph, Node &matrix, Node &vec) {
  MatrixAndVectorPointwiseMultiNode *node = MatrixAndVectorPointwiseMultiNode::newNode(
      matrix.getDim());
  node->forward(graph, matrix, vec);
  return node;
}

Node *matrixColSum(Graph &graph, Node &input) {
  MatrixColSumNode *node = MatrixColSumNode::newNode(input.getColumn());
  node->forward(graph, input);
  return node;
}

Node *matrixAndVectorMulti(Graph &graph, Node &matrix, Node &vec) {
  MatrixAndVectorMultiNode *node = MatrixAndVectorMultiNode::newNode(matrix.getRow());
  node->forward(graph, matrix, vec);
  return node;
}

}
