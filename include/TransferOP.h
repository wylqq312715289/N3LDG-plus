/*
 * TransferOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef TransferOP_H_
#define TransferOP_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"
#include "Alphabet.h"

class TransferParams {
 public:
  vector<Param> W;
  PAlphabet elems;
  int nVSize;
  int nInSize;
  int nOutSize;

 public:
  TransferParams() {
    nVSize = 0;
  }

  void exportAdaParams(ModelUpdate &ada) {
    for (int idx = 0; idx < nVSize; idx++) {
      ada.addParam(&(W[idx]));
    }
  }

  void initial(PAlphabet alpha, int nOSize, int nISize) {
    elems = alpha;
    nVSize = elems->size();
    nInSize = nISize;
    nOutSize = nOSize;
    W.resize(nVSize);
    for (int idx = 0; idx < nVSize; idx++) {
      W[idx].initial(nOSize, nISize);
    }
  }

  int getElemId(const string &strFeat) {
    return elems->from_string(strFeat);
  }

  // will add it
  void save(std::ofstream &os) const {

  }

  // will add it
  void load(std::ifstream &is) {

  }

};

class TransferNode : public Node {
 public:
  PNode in;
  int xid;
  TransferParams *param;

 public:
  TransferNode() : Node() {
    in = NULL;
    xid = -1;
    param = NULL;

  }

  void setParam(TransferParams *paramInit) {
    param = paramInit;
  }

  void clearValue() {
    Node::clearValue();
    in = NULL;
    xid = -1;
  }

 public:
  void forward(Graph *cg, PNode x, const string &strNorm) {
    in = x;
    xid = param->getElemId(strNorm);
    if (xid < 0) {
      std::cout << "TransferNode warning: could find the label: " << strNorm << std::endl;
    }
    degree = 0;
    in->addParent(this);
  }

 public:
  void compute() {
    if (xid >= 0) {
      val.mat() = param->W[xid].val.mat() * in->val.mat();
    }
  }

  void backward() {
    if (xid >= 0) {
      param->W[xid].grad.mat() += loss.mat() * in->val.tmat();
      in->loss.mat() += param->W[xid].val.mat().transpose() * loss.mat();
    }
  }

 public:
  PExecute generate(bool bTrain, dtype cur_drop_factor);

  // better to rewrite for deep understanding
  bool typeEqual(PNode other) {
    bool result = Node::typeEqual(other);
    if (!result) return false;

    TransferNode *conv_other = (TransferNode *) other;
    if (param != conv_other->param) {
      return false;
    }
    if (xid != conv_other->xid) {
      return false;
    }

    return true;
  }

};

class TransferExecute : public Execute {
 public:
  void forward() {
    int count = batch.size();
    //#pragma omp parallel for
    for (int idx = 0; idx < count; idx++) {
      batch[idx]->compute();
      batch[idx]->forward_drop(bTrain, drop_factor);
    }
  }

  void backward() {
    int count = batch.size();
    //#pragma omp parallel for
    for (int idx = 0; idx < count; idx++) {
      batch[idx]->backward_drop();
      batch[idx]->backward();
    }
  }
};

#endif /* TransferOP_H_ */
