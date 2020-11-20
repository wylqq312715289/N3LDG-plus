#ifndef _SOFTMAXLOSS_H_
#define _SOFTMAXLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include "Node.h"

dtype softMaxLoss(PNode x, const vector<dtype> &answer, Metric &eval,
                  dtype batchsize);

dtype softMaxLoss(Node &node, int answer, Metric &metric, int batchsize);

#if USE_GPU
dtype softMaxLoss(const std::vector<PNode> &x, const std::vector<int> &answers,
        n3ldg_cuda::DeviceInt &correct,
        int batchsize = 1);
#endif

dtype cost(PNode x, const vector<dtype> &answer, int batchsize = 1);

#endif /* _SOFTMAXLOSS_H_ */
