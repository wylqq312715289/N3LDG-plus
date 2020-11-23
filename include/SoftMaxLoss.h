#ifndef _SOFTMAXLOSS_H_
#define _SOFTMAXLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include "Node.h"

dtype softMaxLoss(PNode x, const vector<dtype> &answer, Metric &eval,
                  dtype batchsize);

#if USE_GPU
dtype softMaxLoss(const std::vector<PNode> &x, const std::vector<int> &answers,
        n3ldg_cuda::DeviceInt &correct,
        int batchsize = 1) ;
#endif

#if USE_GPU
void softMaxPredict(PNode x, int &y);
#else
dtype predict(PNode x, int &y);
#endif

dtype cost(PNode x, const vector<dtype> &answer, int batchsize = 1);

bool SoftMax(PNode x, vector<dtype> &answer);

#endif /* _SOFTMAXLOSS_H_ */
