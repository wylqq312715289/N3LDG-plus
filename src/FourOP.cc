#include "FourOP.h"
PExecute FourNode::generate(bool bTrain, dtype cur_drop_factor) {
  FourExecute *exec = new FourExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor * drop_value;
  exec->inDim1 = param->W1.inDim();
  exec->inDim2 = param->W2.inDim();
  exec->inDim3 = param->W3.inDim();
  exec->inDim4 = param->W4.inDim();
  exec->outDim = param->W1.outDim();
  exec->param = param;
  exec->activate = activate;
  exec->derivate = derivate;
  return exec;
};
PExecute LinearFourNode::generate(bool bTrain, dtype cur_drop_factor) {
  LinearFourExecute *exec = new LinearFourExecute();
  exec->batch.push_back(this);
  exec->drop_factor = cur_drop_factor * drop_value;
  exec->inDim1 = param->W1.inDim();
  exec->inDim2 = param->W2.inDim();
  exec->inDim3 = param->W3.inDim();
  exec->inDim4 = param->W4.inDim();
  exec->outDim = param->W1.outDim();
  exec->param = param;
  exec->bTrain = bTrain;
  return exec;
};
