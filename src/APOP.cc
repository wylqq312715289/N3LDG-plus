#include "APOP.h"

PExecute APNode::generate(bool bTrain, dtype cur_drop_factor) {
  APExecute *exec = new APExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
}
