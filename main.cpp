#include <iostream>
#include <stdexcept>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
using namespace std;

template <typename T>
class backop;
template <typename T>
class addBackward;
template <typename T>
class subBackward;
template <typename T>
class mulBackward;
template <typename T>
class divBackward;
template <typename T>
class gradTensor;

template <typename T>
gradTensor<T>& operator+ (gradTensor<T>& first, gradTensor<T>& second){
    if (first.getData().shape() != second.getData().shape()) {
        throw std::invalid_argument("Shape mismatch for +");
    }
    xt::xarray<T> newData = first.getData() + second.getData();
    auto ret = new gradTensor<T> (newData);
    auto* op = new addBackward<T>(&first, &second);
    ret->setSource(op);
    return *ret;
}

template <typename T>
gradTensor<T>& operator- (gradTensor<T>& first, gradTensor<T>& second){
    if (first.getData().shape() != second.getData().shape()) {
        throw std::invalid_argument("Shape mismatch for +");
    }
    xt::xarray<T> newData = first.getData() - second.getData();
    auto ret = new gradTensor<T> (newData);
    auto* op = new subBackward<T>(&first, &second);
    ret->setSource(op);
    return *ret;
}

template <typename T>
gradTensor<T>& operator* (gradTensor<T>& first, gradTensor<T>& second){
    if (first.getData().shape() != second.getData().shape()) {
        throw std::invalid_argument("Shape mismatch for +");
    }
    xt::xarray<T> newData = first.getData() * second.getData();
    auto ret = new gradTensor<T> (newData);
    auto* op = new mulBackward<T>(&first, &second);
    ret->setSource(op);
    return *ret;
}

template <typename T>
gradTensor<T>& operator/ (gradTensor<T>& first, gradTensor<T>& second){
    if (first.getData().shape() != second.getData().shape()) {
        throw std::invalid_argument("Shape mismatch for +");
    }
    xt::xarray<T> newData = first.getData() / second.getData();
    auto ret = new gradTensor<T> (newData);
    auto* op = new divBackward<T>(&first, &second);
    ret->setSource(op);
    return *ret;
}

template <typename T>
class gradTensor {
    private:
    xt::xarray<T> data;
    xt::xarray<T> grad;
    backop<T>* source;

    public:
    const xt::xarray<T>& getData() const { return data; }
    const xt::xarray<T>& getGrad() const { return grad; }
    backop<T>* getSource() const { return source; }
    void setSource(backop<T>* op) { source = op; }


    gradTensor() : data(), grad(), source(nullptr) {}
    gradTensor(const xt::xarray<T>& d) : data(d), grad(xt::zeros_like(d)), source(nullptr) {}

    void backward(const xt::xarray<T>& grad_current);
    ~gradTensor();

    friend gradTensor<T>& operator+<> (gradTensor<T>& first, gradTensor<T>& second);
    friend gradTensor<T>& operator-<> (gradTensor<T>& first, gradTensor<T>& second);
    friend gradTensor<T>& operator*<> (gradTensor<T>& first, gradTensor<T>& second);
    friend gradTensor<T>& operator/<> (gradTensor<T>& first, gradTensor<T>& second);
};

template <typename T>
gradTensor<T>::~gradTensor() {
    delete source;
}

template <typename T>
class backop {
public:
    virtual void backward(const xt::xarray<T>& accum_grad) = 0;
    virtual ~backop() {}
};

template <typename T>
class addBackward : public backop<T> {
    gradTensor<T>* arg1;
    gradTensor<T>* arg2;
public:
    addBackward(gradTensor<T>* a1, gradTensor<T>* a2) : arg1(a1), arg2(a2) {}
    void backward(const xt::xarray<T>& accum_grad) override {
        arg1->backward(accum_grad);
        arg2->backward(accum_grad);
    }
};

template <typename T>
class subBackward : public backop<T> {
    gradTensor<T>* arg1;
    gradTensor<T>* arg2;
public:
    subBackward(gradTensor<T>* a1, gradTensor<T>* a2) : arg1(a1), arg2(a2) {}
    void backward(const xt::xarray<T>& accum_grad) override {
        arg1->backward(accum_grad);
        arg2->backward(-accum_grad);
    }
};

template <typename T>
class mulBackward : public backop<T> {
    gradTensor<T>* arg1;
    gradTensor<T>* arg2;
public:
    mulBackward(gradTensor<T>* a1, gradTensor<T>* a2) : arg1(a1), arg2(a2) {}
    void backward(const xt::xarray<T>& accum_grad) override {
        arg1->backward(arg2->getData() * accum_grad);
        arg2->backward(arg1->getData() * accum_grad);
    }
};

template <typename T>
class divBackward : public backop<T> {
    gradTensor<T>* arg1;
    gradTensor<T>* arg2;
public:
    divBackward(gradTensor<T>* a1, gradTensor<T>* a2) : arg1(a1), arg2(a2) {}
    void backward(const xt::xarray<T>& accum_grad) override {
        arg1->backward((1.0 / arg2->getData()) * accum_grad);
        auto squared = arg2->getData() * arg2->getData();
        arg2->backward((-arg1->getData() / squared) * accum_grad);
    }
};

template <typename T>
void gradTensor<T>::backward(const xt::xarray<T>& grad_current) {
    if (grad.shape() != grad_current.shape()) {
        throw invalid_argument("Gradient shape mismatch");
    }
    grad += grad_current;
    if (source) {
        source->backward(grad_current);
    }
}

int main(){
    xt::xarray<double> tensor = {1.0, 2.0, 3.0};
    xt::xarray<double> tensor1 = {4.0, 5.0, 6.0};
    const xt::xarray<double> start = {1.0, 1.0, 1.0};

    gradTensor<float> x(tensor);
    gradTensor<float> y(tensor1);
    const gradTensor<float> init(start);

    auto z = x*x + y*y;
    z.backward(start);

    cout << y.getGrad() << endl;
}