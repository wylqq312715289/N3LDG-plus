#ifndef ATTENTION_HELP
#define ATTENTION_HELP

/*
*  AttentionHelp.h:
*  attention softmax help nodes
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "SoftMaxNode.h"
#include "Concat.h"
#include "Split.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "Pooling.h"
#include "MatrixNode.h"
#include <memory>

namespace n3ldg_plus {

Node *attention(Graph &graph, vector<Node *> &inputs, vector<Node *> &scores);

}

#endif
