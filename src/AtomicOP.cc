#include "AtomicOP.h"

PExecute ActivateNode::generate(bool bTrain, dtype cur_drop_factor) {
  ActivateExecute *exec = new ActivateExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
};

PExecute TanhNode::generate(bool bTrain, dtype cur_drop_factor) {
  TanhExecute *exec = new TanhExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  exec->dim = dim;
  return exec;
};

PExecute SigmoidNode::generate(bool bTrain, dtype cur_drop_factor) {
  SigmoidExecute *exec = new SigmoidExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  exec->dim = dim;
  return exec;
};

PExecute ReluNode::generate(bool bTrain, dtype cur_drop_factor) {
  ReluExecute *exec = new ReluExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
};

PExecute IndexNode::generate(bool bTrain, dtype cur_drop_factor) {
  IndexExecute *exec = new IndexExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
}

PExecute PSubNode::generate(bool bTrain, dtype cur_drop_factor) {
  PSubExecute *exec = new PSubExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
}

PExecute PDotNode::generate(bool bTrain, dtype cur_drop_factor) {
  PDotExecute *exec = new PDotExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
}
