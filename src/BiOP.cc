#include "BiOP.h"

PExecute BiNode::generate(bool bTrain, dtype cur_drop_factor) {
  BiExecute *exec = new BiExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  exec->inDim1 = param->W1.inDim();
  exec->inDim2 = param->W2.inDim();
  exec->outDim = param->W1.outDim();
  exec->param = param;
  exec->activate = activate;
  exec->derivate = derivate;
  return exec;
};
PExecute LinearBiNode::generate(bool bTrain, dtype cur_drop_factor) {
  LinearBiExecute *exec = new LinearBiExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
};
