#ifndef N3LDG_PLUS_LAYER_NORMALIZATION_H
#define N3LDG_PLUS_LAYER_NORMALIZATION_H

#include "MyLib.h"
#include "LookupTable.h"
#include "Node.h"
#include "PAddOP.h"
#include "BucketOP.h"
#include "Div.h"
#include "AtomicOP.h"
#include "Graph.h"
#include "Pooling.h"
#include "Sub.h"
#include "PMultiOP.h"
#include "Param.h"
#include "UniOP.h"

class LayerNormalizationParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
  , public TransferableComponents
#endif
{
 public:
  LayerNormalizationParams(const string &name) : g_(name + "-g"), b_(name + "-b") {}

  void init(int dim) {
    g_.init(dim, 1);
    g_.val.assignAll(1);
    b_.initAsBias(dim);
  }

  Param &g() {
    return g_;
  }

  BiasParam &b() {
    return b_;
  }

  Json::Value toJson() const override {
    Json::Value json;
    json["g"] = g_.toJson();
    json["b"] = b_.toJson();
    return json;
  }

  void fromJson(const Json::Value &json) override {
    g_.fromJson(json["g"]);
    b_.fromJson(json["b"]);
  }

#if USE_GPU
  std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
      return {&g_, &b_};
  }
#endif

 protected:
  virtual std::vector<Tunable<BaseParam> *> tunableComponents() override {
    return {&g_, &b_};
  }

 private:
  Param g_;
  BiasParam b_;
};

Node *layerNormalization(Graph &graph, LayerNormalizationParams &params,
                         Node &input_layer);
#endif
