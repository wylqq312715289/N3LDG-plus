#include "Pooling.h"

PExecute PoolNode::generate(bool bTrain, dtype cur_drop_factor) {
  PoolExecute *exec = new PoolExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
}

PExecute SumPoolNode::generate(bool bTrain, dtype cur_drop_factor) {
  SumPoolExecute *exec = new SumPoolExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
#if USE_GPU
  exec->dim = dim;
#endif
  return exec;
}

PExecute AvgPoolNode::generate(bool bTrain, dtype cur_drop_factor) {
  AvgPoolExecute *exec = new AvgPoolExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
#if USE_GPU
  exec->dim = dim;
#endif
  return exec;
}
