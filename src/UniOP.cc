#include "UniOP.h"

PExecute UniNode::generate(bool bTrain, dtype cur_drop_factor) {
  UniExecute *exec = new UniExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  exec->inDim = param->W.inDim();
  exec->outDim = param->W.outDim();
  exec->param = param;
  exec->activate = activate;
  exec->derivate = derivate;
  return exec;
};

PExecute LinearUniNode::generate(bool bTrain, dtype cur_drop_factor) {
  LinearUniExecute *exec = new LinearUniExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
};

PExecute LinearNode::generate(bool bTrain, dtype cur_drop_factor) {
  LinearExecute *exec = new LinearExecute();
  exec->batch.push_back(this);
  exec->inDim = param->W.inDim();
  exec->outDim = param->W.outDim();
  exec->param = param;
  exec->bTrain = bTrain;
  return exec;
};
