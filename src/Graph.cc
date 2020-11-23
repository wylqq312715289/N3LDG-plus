#include "Graph.h"

using namespace Eigen;

int GetDegree(std::map<void *, int> &degree_map, PNode p) {
  auto it = degree_map.find(p);
  if (it == degree_map.end()) {
    degree_map.insert(std::pair<void *, int>(p, p->degree));
    return p->degree;
  } else {
    return it->second;
  }
}

void DecreaseDegree(std::map<void *, int> &degree_map, PNode p) {
  auto it = degree_map.find(p);
  if (it == degree_map.end()) {
    degree_map.insert(std::pair<void *, int>(p, p->degree - 1));
  } else {
    --(it->second);
  }
}

void Insert(const PNode node, NodeMap &node_map) {
  size_t x_hash = node->typeHashCode();
  auto it = node_map.find(x_hash);
  if (it == node_map.end()) {
    std::vector<PNode> v = {node};
    node_map.insert(std::make_pair<size_t, std::vector<PNode>>(std::move(x_hash), std::move(v)));
  } else {
    it->second.push_back(node);
  }
}

int Size(const NodeMap &map) {
  int sum = 0;
  for (auto it : map) {
    sum += it.second.size();
  }
  return sum;
}
