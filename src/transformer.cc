#include "transformer.h"

void initPositionalEncodingParam(Param &param, int dim, int max_sentence_len) {
  param.init(dim, max_sentence_len);
  for (int pos_i = 0; pos_i < max_sentence_len; ++pos_i) {
    for (int dim_i = 0; dim_i < dim; ++dim_i) {
      dtype v;
      if (dim_i % 2 == 0) {
        int half = dim_i / 2;
        v = sin(pos_i / pow(1e4, 2.0 * half / dim));
      } else {
        int half = (dim_i - 1) / 2;
        v = cos(pos_i / pow(1e4, 2.0 * half / dim));
      }
      param.val[pos_i][dim_i] = v;
    }
  }
#if USE_GPU
  param.val.copyFromHostToDevice();
#endif
}

namespace n3ldg_plus {

vector<Node *> transformerEncoder(Graph &graph, TransformerEncoderParams &params,
                                  vector<Node *> &inputs,
                                  dtype dropout,
                                  bool is_training) {
  using namespace n3ldg_plus;
  n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
  vector<Node *> pos_encoded_layer;
  int sentence_len = inputs.size();
  pos_encoded_layer.reserve(sentence_len);
  for (int i = 0; i < sentence_len; ++i) {
    Node *embedding = n3ldg_plus::embedding(graph, params.positionalEncodingParam(), i, false);
    Node *input = linear(graph, params.inputLinear(), *inputs.at(i));
    Node *pos_encoded = add(graph, {input, embedding});
    pos_encoded = n3ldg_plus::dropout(graph, *pos_encoded, dropout, is_training);
    pos_encoded_layer.push_back(pos_encoded);
  }

  int layer_count = params.layerCount();

  vector<Node *> last_layer = pos_encoded_layer;
  for (int i = 0; i < layer_count; ++i) {
    profiler.BeginEvent("multi head");
    if (last_layer.size() != sentence_len) {
      cerr << "transformer - last_layer.size():" << last_layer.size() << " sentence_len:"
           << sentence_len << endl;
      abort();
    }
    auto &layer_params = *params.layerParams().ptrs().at(i);

    int head_count = params.headCount();
    vector<Node *> key_heads, value_heads;
    key_heads.reserve(head_count);
    value_heads.reserve(head_count);
    auto &attention_head_params = layer_params.multiHeadAttentionParams();
    vector<Node *> keys, values;
    keys.reserve(sentence_len);
    values.reserve(sentence_len);
    for (int m = 0; m < sentence_len; ++m) {
      Node *kv_input = last_layer.at(m);
      Node *k = linear(graph, attention_head_params.k(), *kv_input);
      keys.push_back(k);
      Node *v = linear(graph, attention_head_params.v(), *kv_input);
      values.push_back(v);
    }
    int section_dim = keys.front()->getDim() / head_count;
    for (int j = 0; j < head_count; ++j) {
      vector<Node *> split_keys, split_values;
      split_keys.reserve(sentence_len);
      split_values.reserve(sentence_len);
      for (int m = 0; m < sentence_len; ++m) {
        split_keys.push_back(split(graph, section_dim, *keys.at(m), section_dim * j));
      }
      Node *key_matrix = concatToMatrix(graph, split_keys);
      key_heads.push_back(key_matrix);
      for (int m = 0; m < sentence_len; ++m) {
        split_values.push_back(split(graph, section_dim, *values.at(m), section_dim * j));
      }
      Node *value_matrix = concatToMatrix(graph, split_values);
      value_heads.push_back(value_matrix);
    }
    profiler.EndEvent();

    vector<Node *> sub_layer;
    for (int j = 0; j < sentence_len; ++j) {
      profiler.BeginEvent("multi head");
      vector<Node *> attended_segments;
      attended_segments.reserve(head_count);
      auto &attention_head_params = layer_params.multiHeadAttentionParams();
      Node *q_input = last_layer.at(j);
      Node *q = linear(graph, attention_head_params.q(), *q_input);

      for (int k = 0; k < head_count; ++k) {
        Node *split_q = split(graph, section_dim, *q, section_dim * k);
        Node *attended = n3ldg_plus::dotAttention(graph, *key_heads.at(k),
                                                  *value_heads.at(k), *split_q).first;
        if (attended->getDim() * head_count != params.hiddenDim()) {
          cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%") %
              attended->getDim() % head_count % params.hiddenDim()
               << endl;
          abort();
        }
        attended_segments.push_back(attended);
      }

      Node *concated = concat(graph, attended_segments);
      profiler.EndEvent();
      concated = n3ldg_plus::dropout(graph, *concated, dropout, is_training);
      Node *added = add(graph, {concated, last_layer.at(j)});
      Node *normed = layerNormalization(graph, layer_params.layerNormA(), *added);
      Node *t = linear(graph, layer_params.ffnInnerParams(), *normed);
      t = relu(graph, *t);
      t = linear(graph, layer_params.ffnOutterParams(), *t);
      t = n3ldg_plus::dropout(graph, *t, dropout, is_training);
      t = add(graph, {normed, t});
      Node *normed2 = layerNormalization(graph, layer_params.layerNormB(), *t);
      sub_layer.push_back(normed2);
    }
    last_layer = sub_layer;
  }

  return last_layer;
}

}
