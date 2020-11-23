#include "BucketOP.h"

using namespace Eigen;

PExecute BucketNode::generate(bool bTrain, dtype cur_drop_factor) {
  BucketExecute *exec = new BucketExecute();
  exec->batch.push_back(this);
  exec->bTrain = bTrain;
  exec->drop_factor = cur_drop_factor;
  return exec;
}


