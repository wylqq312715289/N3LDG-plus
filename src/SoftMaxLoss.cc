#include "SoftMaxLoss.h"

dtype softMaxLoss(PNode x, const vector<dtype> &answer, Metric &eval,
                  dtype batchsize) {
  int nDim = x->dim;
  int labelsize = answer.size();
  if (labelsize != nDim) {
    std::cerr << "softmax_loss error: dim size invalid" << std::endl;
    return -1.0;
  }

  NRVec<dtype> scores(nDim);

  dtype cost = 0.0;
  int optLabel = -1;
  for (int i = 0; i < nDim; ++i) {
    if (answer[i] >= 0) {
      if (optLabel < 0 || x->val[i] > x->val[optLabel])
        optLabel = i;
    }
  }

  dtype sum1 = 0, sum2 = 0, maxScore = x->val[optLabel];
  for (int i = 0; i < nDim; ++i) {
    scores[i] = -1e10;
    if (answer[i] >= 0) {
      scores[i] = exp(x->val[i] - maxScore);
      if (answer[i] == 1)
        sum1 += scores[i];
      sum2 += scores[i];
    }
  }
  cost += (log(sum2) - log(sum1)) / batchsize;
  if (answer[optLabel] == 1)
    eval.correct_label_count++;
  eval.overall_label_count++;

  for (int i = 0; i < nDim; ++i) {
    if (answer[i] >= 0) {
      x->loss[i] = (scores[i] / sum2 - answer[i]) / batchsize;
    }
  }

  return cost;

}

#if USE_GPU
dtype softMaxLoss(const std::vector<PNode> &x, const std::vector<int> &answers,
        n3ldg_cuda::DeviceInt &correct,
        int batchsize) {
    std::vector<dtype*> vals, losses;
    vals.reserve(x.size());
    losses.reserve(x.size());
    for (PNode n : x) {
        vals.push_back(n->val.value);
        losses.push_back(n->loss.value);
    }
    n3ldg_cuda::SoftMaxLoss(vals, losses, correct.value, answers,
            batchsize, x.size(), x.at(0)->dim);
    return -1.0f;
}
#endif

#if USE_GPU
void softMaxPredict(PNode x, int &y) {
    y = n3ldg_cuda::Predict(x->val.value, x->dim);
}
#else
dtype predict(PNode x, int &y) {
  int nDim = x->dim;

  int optLabel = -1;
  for (int i = 0; i < nDim; ++i) {
    if (optLabel < 0 || x->val[i] > x->val[optLabel])
      optLabel = i;
  }

  dtype prob = 0.0;
  dtype sum = 0.0;
  NRVec<dtype> scores(nDim);
  dtype maxScore = x->val[optLabel];
  for (int i = 0; i < nDim; ++i) {
    scores[i] = exp(x->val[i] - maxScore);
    sum += scores[i];
  }
  prob = scores[optLabel] / sum;
  y = optLabel;
  return prob;
}
#endif

dtype cost(PNode x, const vector<dtype> &answer, int batchsize) {
  int nDim = x->dim;
  int labelsize = answer.size();
  if (labelsize != nDim) {
    std::cerr << "softmax_loss error: dim size invalid" << std::endl;
    return -1.0;
  }

  NRVec<dtype> scores(nDim);

  dtype cost = 0.0;

  int optLabel = -1;
  for (int i = 0; i < nDim; ++i) {
    if (answer[i] >= 0) {
      if (optLabel < 0 || x->val[i] > x->val[optLabel])
        optLabel = i;
    }
  }

  dtype sum1 = 0, sum2 = 0, maxScore = x->val[optLabel];
  for (int i = 0; i < nDim; ++i) {
    scores[i] = -1e10;
    if (answer[i] >= 0) {
      scores[i] = exp(x->val[i] - maxScore);
      if (answer[i] == 1)
        sum1 += scores[i];
      sum2 += scores[i];
    }
  }
  cost += (log(sum2) - log(sum1)) / batchsize;
  return cost;
}

bool SoftMax(PNode x, vector<dtype> &answer) {
  answer.clear();
  int nDim = x->dim;

  dtype sum = 0.0;

  for (int i = 0; i < nDim; ++i) {
    float tmp = exp(x->val[i]);
    sum += tmp;
    answer.push_back(tmp);
  }

  for (int i = 0; i < nDim; ++i) {
    answer[i] /= sum;
  }
  return true;

}



