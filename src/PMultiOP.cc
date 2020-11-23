#include "PMultiOP.h"

PExecute PMultiNode::generate(bool bTrain, dtype cur_drop_factor) {
  PMultiExecute *exec = new PMultiExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  exec->dim = dim;
  return exec;
};
