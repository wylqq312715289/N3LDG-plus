#include "PAddOP.h"

PExecute PAddNode::generate(bool bTrain, dtype cur_drop_factor) {
  PAddExecute *exec = new PAddExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor * drop_value;
  exec->in_count = ins.size();
  exec->dim = dim;
  return exec;
}
