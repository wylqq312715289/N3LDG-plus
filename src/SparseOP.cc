#include "SparseOP.h"

PExecute SparseNode::generate(bool bTrain, dtype cur_drop_factor) {
  SparseExecute *exec = new SparseExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
}
