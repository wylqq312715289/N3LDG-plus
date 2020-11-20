#include "Attention.h"

namespace n3ldg_plus {

pair<Node *, Node *> dotAttention(Graph &cg, Node &key_matrix, Node &value_matrix,
                                  Node &guide) {
  Node *matrix = n3ldg_plus::matrixPointwiseMultiply(cg, key_matrix, guide);
  Node *sum = n3ldg_plus::matrixColSum(cg, *matrix);
  Node *scaled_weight = n3ldg_plus::scaled(cg, *sum, 1.0 / ::sqrt((dtype) guide.getDim()));
  scaled_weight = n3ldg_plus::softmax(cg, *scaled_weight);
  Node *hidden = n3ldg_plus::matrixAndVectorMulti(cg, value_matrix, *scaled_weight);
  return make_pair(hidden, sum);
}

Node *dotAttentionWeights(Graph &cg, Node &key_matrix, Node &guide) {
  Node *matrix = n3ldg_plus::matrixPointwiseMultiply(cg, key_matrix, guide);
  Node *sum = n3ldg_plus::matrixColSum(cg, *matrix);
  Node *scaled_weight = n3ldg_plus::scaled(cg, *sum, 1.0 / ::sqrt((dtype) guide.getDim()));
  scaled_weight = n3ldg_plus::softmax(cg, *scaled_weight);
  return scaled_weight;
}

}

namespace n3ldg_plus {

vector<Node *> additiveAttentionWeights(Graph &graph, AdditiveAttentionParams &params,
                                        vector<Node *> &values,
                                        Node &guide) {
  Node *q = linear(graph, params.q, guide);
  vector<Node *> weights;

  for (int idx = 0; idx < values.size(); idx++) {
    Node *k = linear(graph, params.k, *values.at(idx));
    Node *sum = add(graph, {k, q});
    Node *nonlinear = tanh(graph, *sum);
    Node *w = linear(graph, params.w3t, *nonlinear);
    weights.push_back(w);
  }
  return weights;
}

}

