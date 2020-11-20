#ifndef N3LDG_MAX_PROBABILITY_LOSS_H
#define N3LDG_MAX_PROBABILITY_LOSS_H

#include <vector>
#include <utility>
#include "Loss.h"
#include "Node.h"

#include "MyLib.h"

std::pair<dtype, std::vector<int>> maxLogProbabilityLoss(std::vector<Node *> &nodes,
                                                         const std::vector<int> &answers,
                                                         dtype factor);

#endif
