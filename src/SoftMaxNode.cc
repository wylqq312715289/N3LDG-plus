#include "SoftMaxNode.h"

namespace n3ldg_plus {

Node *minusMaxScalar(Graph &graph, Node &input) {
  int dim = input.getDim();
  using namespace n3ldg_plus;

  Node *max_scalar = maxScalar(graph, input);
  Node *scalar_to_vector = scalarToVector(graph, dim, *max_scalar);
  Node *subtracted = sub(graph, input, *scalar_to_vector);
  return subtracted;
}

Node *softmax(Graph &graph, Node &input) {
  using namespace n3ldg_plus;
  Node *subtracted = minusMaxScalar(graph, input);
  Node *exp = n3ldg_plus::exp(graph, *subtracted);
  Node *sum = vectorSum(graph, *exp);
  Node *div = n3ldg_plus::div(graph, *exp, *sum);
  return div;
}

}

