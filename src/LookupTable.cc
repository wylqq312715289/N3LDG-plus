#include "LookupTable.h"

PExecute LookupNode::generate(bool bTrain, dtype cur_drop_factor) {
  LookupExecute *exec = new LookupExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
#if USE_GPU
  exec->table = param;
    exec->dim = dim;
#endif
  return exec;
}
