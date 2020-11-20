#ifndef N3LDG_PLUS_SOFTMAX_NODE
#define N3LDG_PLUS_SOFTMAX_NODE

#include "Layer.h"
#include "AtomicOP.h"
#include "Sub.h"
#include "AtomicOP.h"
#include "Div.h"
#include "Split.h"

#include <boost/format.hpp>

namespace n3ldg_plus {

Node *minusMaxScalar(Graph &graph, Node &input);

Node *softmax(Graph &graph, Node &input);

};

#endif
