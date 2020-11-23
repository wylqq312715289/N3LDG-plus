#ifndef BucketOP
#define BucketOP

/*
*  BucketOP.h:
*  a bucket operation, for padding mainly
*  usually an inputleaf node, degree = 0
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

using namespace Eigen;



class BucketNode : public Node {
  public:
    BucketNode() : Node() {
        node_type = "bucket";
    }
  public:
    virtual void clearValue() {
        //Node::clearValue();
#if !USE_GPU || TEST_CUDA
        loss = 0;
        degree = 0;
#endif
        if (drop_value > 0)drop_mask = 1;
        parents.clear();
    }

    virtual void init(int ndim, dtype dropout) {
        Node::init(ndim, -1);
    }

  public:
    void forward(Graph *cg, dtype value) {
#if TEST_CUDA
        val  = value;
        loss = 0;
#endif
#if USE_GPU
        n3ldg_cuda::Memset(val.value, dim, value);
        n3ldg_cuda::Memset(loss.value, dim, 0.0f);
#if TEST_CUDA
        n3ldg_cuda::Assert(val.verify("bucket forward"));
        n3ldg_cuda::Assert(loss.verify("loss verify"));
#endif
#else
        val = value;
        loss = 0;
#endif
        degree = 0;
        cg->addNode(this);
    }

    //value already assigned
    void forward(Graph *cg) {
#if USE_GPU
        n3ldg_cuda::Memset(loss.value, dim, 0.0f);
#else
        loss = 0;
#endif
        degree = 0;
        cg->addNode(this);
    }

    void forwardArr(Graph *cg, dtype *value) {
#if USE_GPU
      abort();
#else
      Vec(val.v, dim) = Vec(value, dim);
      degree = 0;
      cg->addNode(this);
#endif
    }

    void compute() {

    }

    void backward() {

    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};

#if USE_GPU
class BucketExecute : public Execute {
  public:
    void  forward() override {}

    void backward() override {}
};

#else
class BucketExecute : public Execute {
  public:
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
        }
    }
};
#endif


#endif
