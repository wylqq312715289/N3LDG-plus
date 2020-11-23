#include "ActionOP.h"

PExecute ActionNode::generate(bool bTrain, dtype cur_drop_factor) {
  ActionExecute *exec = new ActionExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
}
