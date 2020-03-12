#ifndef DivOP
#define DivOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class DivNode : public Node, public Poolable<DivNode> {
public:
    DivNode() : Node("div_node") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    int getKey() const override {
        return getDim();
    }

    bool typeEqual(Node* other) override {
        return getNodeType() == other->getNodeType();
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void forward(Graph &graph, Node &numerator, Node &denominator) {
        if (getDim() != numerator.getDim() || 1 != denominator.getDim()) {
            cerr << boost::format("dim:%1% minuend:%2% subtrahend:%3%") % getDim() %
                numerator.getDim() % denominator.getDim() << endl;
            abort();
        }
        numerator_ = &numerator;
        denominator_ = &denominator;
        vector<Node*> ins = {numerator_, denominator_};
        afterForward(graph, ins);
    }

    void compute() override {
        val().vec() = numerator_->getVal().vec() / denominator_->getVal()[0];
    }

    void backward() override {
        numerator_->loss().vec() += getLoss().vec() / denominator_->getVal()[0];

        dtype square = denominator_->getVal()[0] * denominator_->getVal()[0];

        for (int i = 0; i < getDim(); ++i) {
            denominator_->loss()[0] -= getLoss()[i] * numerator_->getVal()[i] / square;
        }
    }

    PExecutor generate() override;

private:
    Node *numerator_;
    Node *denominator_;
    friend class DivExecutor;
};

namespace n3ldg_plus {
    Node *div(Graph &graph, Node &numerator, Node &denominator) {
        DivNode *result = DivNode::newNode(numerator.getDim());
        result->forward(graph, numerator, denominator);
        return result;
    }
}

#if USE_GPU
class DivExecutor : public Executor {
public:
    vector<const dtype*> numerators;
    vector<const dtype*> denominators;
    vector<int> dims;

    void forward() override {
        vector<dtype*> results;
        for (Node *node : batch) {
            DivNode *div = static_cast<DivNode*>(node);
            numerators.push_back(div->numerator_->getVal().value);
            denominators.push_back(div->denominator_->getVal().value);
            results.push_back(div->getVal().value);
            dims.push_back(node->getDim());
        }

        n3ldg_cuda::DivForward(numerators, denominators, batch.size(), dims, results);
#if TEST_CUDA
        Executor::testForward();
        cout << "div tested" << endl;
#endif
    }

    void backward() override {
        vector<const dtype*> losses;
        vector<dtype*> numerator_losses, denominator_losses;
        for (Node *node : batch) {
            DivNode *div = static_cast<DivNode*>(node);
            losses.push_back(node->getLoss().value);
            numerator_losses.push_back(div->numerator_->getLoss().value);
            denominator_losses.push_back(div->denominator_->getLoss().value);
        }

        n3ldg_cuda::DivBackward(losses, denominators, numerators, batch.size(), dims,
                numerator_losses, denominator_losses);
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            DivNode &div = static_cast<DivNode&>(node);
            vector<pair<Node*, string>> results = {make_pair(div.denominator_, "denominator"),
                    make_pair(div.numerator_, "numerator")};
            return results;
        };
        Executor::testBackward(get_inputs);
        cout << "div backward tested" << endl;
#endif
    }

};
#else
class DivExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
};
#endif

Executor *DivNode::generate() {
    DivExecutor * executor = new DivExecutor();
    return executor;
}

class FullDivNode : public Node, public Poolable<FullDivNode>  {
public:
    FullDivNode() : Node("full_div") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    int getKey() const override {
        return getDim();
    }

    void forward(Graph &graph, Node &numerator, Node &denominator) {
        if (getDim() != numerator.getDim() || getDim() != denominator.getDim()) {
            cerr << boost::format("dim:%1% minuend:%2% subtrahend:%3%") % getDim() %
                numerator.getDim() % denominator.getDim() << endl;
            abort();
        }
        numerator_ = &numerator;
        denominator_ = &denominator;
        vector<Node*> ins = {numerator_, denominator_};
        afterForward(graph, ins);
    }

    void compute() override {
        val().vec() = numerator_->getVal().vec() / denominator_->getVal().vec();
    }

    void backward() override {
        numerator_->loss().vec() += getLoss().vec() / denominator_->getVal().vec();
        denominator_->loss().vec() -= getLoss().vec() * numerator_->getVal().vec() /
            denominator_->getVal().vec().square();
    }

    PExecutor generate() override;

private:
    Node *numerator_;
    Node *denominator_;
    friend class FullDivExecutor;
};

#if USE_GPU
class FullDivExecutor : public Executor {
public:
    vector<const dtype*> numerators;
    vector<const dtype*> denominators;

    void forward() override {
        vector<dtype*> results;
        for (Node *node : batch) {
            FullDivNode *div = static_cast<FullDivNode*>(node);
            numerators.push_back(div->numerator_->getVal().value);
            denominators.push_back(div->denominator_->getVal().value);
            results.push_back(div->getVal().value);
        }

        n3ldg_cuda::FullDivForward(numerators, denominators, batch.size(), getDim(), results);
#if TEST_CUDA
        Executor::testForward();
        cout << "div tested" << endl;
#endif
    }

    void backward() override {
        vector<const dtype*> losses;
        vector<dtype*> numerator_losses, denominator_losses;
        for (Node *node : batch) {
            FullDivNode *div = static_cast<FullDivNode*>(node);
            losses.push_back(node->getLoss().value);
            numerator_losses.push_back(div->numerator_->getLoss().value);
            denominator_losses.push_back(div->denominator_->getLoss().value);
        }

        n3ldg_cuda::FullDivBackward(losses, denominators, numerators, batch.size(), getDim(),
                numerator_losses, denominator_losses);
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            FullDivNode &div = static_cast<FullDivNode&>(node);
            vector<pair<Node*, string>> results = {make_pair(div.denominator_, "denominator"),
                    make_pair(div.numerator_, "numerator")};
            return results;
        };
        Executor::testBackward(get_inputs);
        cout << "div backward tested" << endl;
#endif
    }

};
#else
class FullDivExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
};
#endif

Executor *FullDivNode::generate() {
    return new FullDivExecutor();
}

namespace n3ldg_plus {
    Node *fullDiv(Graph &graph, Node &numerator, Node &denominator) {
        FullDivNode *result = FullDivNode::newNode(numerator.getDim());
        result->forward(graph, numerator, denominator);
        return result;
    }
}

#endif
