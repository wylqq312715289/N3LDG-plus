#include "UniOP.h"

Executor *LinearNode::generate() {
  LinearExecutor *exec = new LinearExecutor();
  exec->batch.push_back(this);
  exec->inDim = param->W.inDim();
  exec->outDim = param->W.outDim();
  exec->param = param;
  return exec;
};

namespace n3ldg_plus {
Node *linearWordVector(Graph &graph, int dim, Param &word_vectors, Node &input,
                       int offset) {
  if (dim + offset > word_vectors.inDim()) {
    cerr << boost::format("linearWordVector - dim:%1% offset%2% vocabulary_size:%3%") %
        dim % offset % word_vectors.inDim() << endl;
    abort();
  }

  if (input.getDim() != word_vectors.outDim()) {
    cerr << boost::format("LinearWordVectorNode - input dim:%1% word vector dim:%2%") %
        input.getDim() % word_vectors.outDim() << endl;
    abort();
  }

  LinearWordVectorNode *node = LinearWordVectorNode::newNode(dim);
  node->setParam(word_vectors, offset);
  node->forward(graph, input);
  return node;
}
}

Executor *LinearWordVectorNode::generate() {
  LinearWordVectorExecutor *exec = new LinearWordVectorExecutor();
  exec->batch.push_back(this);
  exec->inDim = param_->outDim();
  exec->outDim = getDim();
  exec->param = param_;
  return exec;
}

Executor *BiasNode::generate() {
  return new BiasExecutor;
}

namespace n3ldg_plus {

Node *linear(Graph &graph, UniParams &params, Node &input) {
  int dim = params.W.outDim();
  LinearNode *uni = LinearNode::newNode(dim);
  uni->setParam(params);
  uni->forward(graph, input);
  return uni;
}

Node *uni(Graph &graph, UniParams &params, Node &input, ActivatedEnum activated_type) {
  int dim = params.W.outDim();

  Node *uni = linear(graph, params, input);

  UniInputNode *activated;
  if (activated_type == ActivatedEnum::TANH) {
    activated = TanhNode::newNode(dim);
  } else if (activated_type == ActivatedEnum::SIGMOID) {
    activated = SigmoidNode::newNode(dim);
  } else if (activated_type == ActivatedEnum::RELU) {
    activated = ReluNode::newNode(dim);
  } else {
    cerr << "uni - unsupported activated " << activated << endl;
    abort();
  }

  activated->forward(graph, *uni);

  return activated;
}

Node *bias(Graph &graph, BiasParam &param, Node &input) {
  int dim = input.getDim();
  BiasNode *node = BiasNode::newNode(dim);
  node->setParam(param);
  node->forward(graph, input);
  return node;
}

}
