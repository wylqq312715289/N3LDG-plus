#include "layer_normalization.h"

Node *layerNormalization(Graph &graph, LayerNormalizationParams &params,
                         Node &input_layer) {
  using namespace n3ldg_plus;
  Node *sum = vectorSum(graph, input_layer);
  Node *avg = scaled(graph, *sum, 1.0 / input_layer.getDim());
  Node *avg_vector = scalarToVector(graph, input_layer.getDim(), *avg);
  Node *zeros_around = sub(graph, input_layer, *avg_vector);
  Node *square = pointwiseMultiply(graph, *zeros_around, *zeros_around);
  Node *square_sum = vectorSum(graph, *square);
  Node *eps = bucket(graph, 1, 1e-6);
  square_sum = add(graph, {square_sum, eps});
  Node *var = scaled(graph, *square_sum, 1.0 / input_layer.getDim());
  Node *standard_deviation = sqrt(graph, *var);
  Node *g = embedding(graph, params.g(), 0, true);
  Node *factor = div(graph, *g, *standard_deviation);

  Node *scaled = pointwiseMultiply(graph, *factor, *zeros_around);
  Node *biased = bias(graph, params.b(), *scaled);
  return biased;
}
