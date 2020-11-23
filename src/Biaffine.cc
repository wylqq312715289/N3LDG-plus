#include "Biaffine.h"

PExecute BiaffineNode::generate(bool bTrain, dtype cur_drop_factor) {
  BiaffineExecute *exec = new BiaffineExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
};
