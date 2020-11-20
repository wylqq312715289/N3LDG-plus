#include "MaxProbabilityLoss.h"

std::pair<dtype, std::vector<int>> maxLogProbabilityLoss(std::vector<Node *> &nodes,
                                                         const std::vector<int> &answers,
                                                         dtype factor) {
  n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
  profiler.BeginEvent("crossEntropyLoss");
  dtype loss = crossEntropyLoss(nodes, answers, factor);
  profiler.EndCudaEvent();
  profiler.BeginEvent("predict");
  vector<int> predicted_ids = predict(nodes);
  profiler.EndCudaEvent();
  return make_pair(loss, predicted_ids);
}
