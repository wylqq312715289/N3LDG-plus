#include "TriOP.h"

#if USE_GPU
PExecute TriNode::generate(bool bTrain, dtype drop_factor) {
    TriExecute* exec = new TriExecute();
    exec->batch.push_back(this);
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->inDim3 = param->W3.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    exec->bTrain = bTrain;
    exec->drop_factor = drop_factor;
    return exec;
}


PExecute LinearTriNode::generate(bool bTrain, dtype drop_factor) {
    LinearTriExecute* exec = new LinearTriExecute();
    exec->batch.push_back(this);
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->inDim3 = param->W3.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->bTrain = bTrain;
    exec->drop_factor = drop_factor;
    return exec;
}
#elif USE_BASE
PExecute TriNode::generate(bool bTrain, dtype drop_factor) {
    TriExecute* exec = new TriExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->dim = dim;
    exec->drop_factor = drop_factor;
    return exec;
};

PExecute LinearTriNode::generate(bool bTrain, dtype drop_factor) {
    LinearTriExecute* exec = new LinearTriExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = drop_factor;
    return exec;
};
#else
PExecute TriNode::generate(bool bTrain, dtype drop_factor) {
  TriExecute *exec = new TriExecute();
  exec->batch.push_back(this);
  exec->inDim1 = param->W1.inDim();
  exec->inDim2 = param->W2.inDim();
  exec->inDim3 = param->W3.inDim();
  exec->outDim = param->W1.outDim();
  exec->param = param;
  exec->activate = activate;
  exec->derivate = derivate;
  exec->bTrain = bTrain;
  exec->drop_factor = drop_factor;
  return exec;
}

PExecute LinearTriNode::generate(bool bTrain, dtype drop_factor) {
  LinearTriExecute *exec = new LinearTriExecute();
  exec->batch.push_back(this);
  exec->inDim1 = param->W1.inDim();
  exec->inDim2 = param->W2.inDim();
  exec->inDim3 = param->W3.inDim();
  exec->outDim = param->W1.outDim();
  exec->param = param;
  exec->bTrain = bTrain;
  exec->drop_factor = drop_factor;
  return exec;
}
#endif
