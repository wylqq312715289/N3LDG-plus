#ifndef N3LDGPLUS_LOSS_H
#define N3LDGPLUS_LOSS_H

#include <Node.h>

vector<int> cpuPredict(const vector<Node *> &nodes);

#if USE_GPU

vector<int> gpuPredict(const vector<Node *> &nodes);

#endif

vector<int> predict(const vector<Node *> &nodes);

dtype cpuCrossEntropyLoss(vector<Node *> &nodes, const vector<int> &answers, dtype factor);

dtype crossEntropyLoss(vector<Node *> &nodes, const vector<int> &answers, dtype factor);

float cpuMultiCrossEntropyLoss(vector<Node *> &nodes, const vector<vector<int>> &answers,
                               dtype factor);

float cpuKLLoss(vector<Node *> &nodes, const vector<shared_ptr<vector<dtype>>> &answers,
                dtype factor);

pair<float, vector<int>> KLLoss(vector<Node *> &nodes,
                                const vector<shared_ptr<vector<dtype>>> &answers,
                                dtype factor);

float multiCrossEntropyLoss(vector<Node *> &nodes, const vector<vector<int>> &answers,
                            dtype factor);

#endif
