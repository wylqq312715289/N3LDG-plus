#include "TransferOP.h"

PExecute TransferNode::generate(bool bTrain, dtype cur_drop_factor) {
  TransferExecute *exec = new TransferExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
};
