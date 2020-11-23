#include "Concat.h"

PExecute ConcatNode::generate(bool bTrain, dtype cur_drop_factor) {
  ConcatExecute *exec = new ConcatExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
#if USE_GPU
  exec->inCount = this->ins.size();
    exec->outDim = 0;
    for (int d : inDims) {
        exec->outDim += d;
    }
#endif
  return exec;
}
