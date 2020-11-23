#include "AttentionHelp.h"

PExecute AttentionSoftMaxNode::generate(bool bTrain, dtype cur_drop_factor) {
  AttentionSoftMaxExecute *exec = new AttentionSoftMaxExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
#if USE_GPU
  exec->dim = dim;
#endif
  return exec;
}

PExecute AttentionSoftMaxVNode::generate(bool bTrain, dtype cur_drop_factor) {
  AttentionSoftMaxVExecute *exec = new AttentionSoftMaxVExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
#if USE_GPU
  exec->dim = dim;
#endif
  return exec;
}
