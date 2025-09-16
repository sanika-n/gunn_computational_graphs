#include <iostream>
#include <memory>
#include <stdexcept>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"

using namespace std;

template <typename T>
class gradTensor;

template <typename T>
class backop {
public:
    virtual void backward(const xt::xarray<T>& accum_grad) = 0;
    virtual ~backop() = default;
};

// ===================== gradTensor =====================
template <typename T>
class gradTensor : public enable_shared_from_this<gradTensor<T>> {
private:
    xt::xarray<T> data;
    xt::xarray<T> grad;
    shared_ptr<backop<T>> source;

public:
    gradTensor() : data(), grad(), source(nullptr) {}
    gradTensor(const xt::xarray<T>& d) 
        : data(d), grad(xt::zeros_like(d)), source(nullptr) {}

    const xt::xarray<T>& getData() const { return data; }
    const xt::xarray<T>& getGrad() const { return grad; }

    void setSource(shared_ptr<backop<T>> op) { source = op; }
    shared_ptr<backop<T>> getSource() const { return source; }

    void backward(const xt::xarray<T>& grad_current) {
        if (grad.shape() != grad_current.shape()) {
            throw invalid_argument("Gradient shape mismatch");
        }
        grad += grad_current;
        if (source) {
            source->backward(grad_current);
        }
    }

    void backward() {
        xt::xarray<T> grad_current = xt::ones_like(data);
        backward(grad_current);
    }
};

// ===================== Backward Ops =====================
template <typename T>
class addBackward : public backop<T> {
    shared_ptr<gradTensor<T>> arg1, arg2;
public:
    addBackward(shared_ptr<gradTensor<T>> a1, shared_ptr<gradTensor<T>> a2) 
        : arg1(a1), arg2(a2) {}

    void backward(const xt::xarray<T>& accum_grad) override {
        arg1->backward(accum_grad);
        arg2->backward(accum_grad);
    }
};

template <typename T>
class subBackward : public backop<T> {
    shared_ptr<gradTensor<T>> arg1, arg2;
public:
    subBackward(shared_ptr<gradTensor<T>> a1, shared_ptr<gradTensor<T>> a2) 
        : arg1(a1), arg2(a2) {}

    void backward(const xt::xarray<T>& accum_grad) override {
        arg1->backward(accum_grad);
        arg2->backward(-accum_grad);
    }
};

template <typename T>
class mulBackward : public backop<T> {
    shared_ptr<gradTensor<T>> arg1, arg2;
public:
    mulBackward(shared_ptr<gradTensor<T>> a1, shared_ptr<gradTensor<T>> a2) 
        : arg1(a1), arg2(a2) {}

    void backward(const xt::xarray<T>& accum_grad) override {
        arg1->backward(arg2->getData() * accum_grad);
        arg2->backward(arg1->getData() * accum_grad);
    }
};

template <typename T>
class divBackward : public backop<T> {
    shared_ptr<gradTensor<T>> arg1, arg2;
public:
    divBackward(shared_ptr<gradTensor<T>> a1, shared_ptr<gradTensor<T>> a2) 
        : arg1(a1), arg2(a2) {}

    void backward(const xt::xarray<T>& accum_grad) override {
        arg1->backward((1.0 / arg2->getData()) * accum_grad);
        auto squared = arg2->getData() * arg2->getData();
        arg2->backward((-arg1->getData() / squared) * accum_grad);
    }
};

// ===================== Operator Overloads =====================
template <typename T>
shared_ptr<gradTensor<T>> operator+(shared_ptr<gradTensor<T>> first,
                                    shared_ptr<gradTensor<T>> second) {
    if (first->getData().shape() != second->getData().shape()) {
        throw invalid_argument("Shape mismatch for +");
    }
    auto newData = first->getData() + second->getData();
    auto ret = make_shared<gradTensor<T>>(newData);
    ret->setSource(make_shared<addBackward<T>>(first, second));
    return ret;
}

template <typename T>
shared_ptr<gradTensor<T>> operator-(shared_ptr<gradTensor<T>> first,
                                    shared_ptr<gradTensor<T>> second) {
    if (first->getData().shape() != second->getData().shape()) {
        throw invalid_argument("Shape mismatch for -");
    }
    auto newData = first->getData() - second->getData();
    auto ret = make_shared<gradTensor<T>>(newData);
    ret->setSource(make_shared<subBackward<T>>(first, second));
    return ret;
}

template <typename T>
shared_ptr<gradTensor<T>> operator*(shared_ptr<gradTensor<T>> first,
                                    shared_ptr<gradTensor<T>> second) {
    if (first->getData().shape() != second->getData().shape()) {
        throw invalid_argument("Shape mismatch for *");
    }
    auto newData = first->getData() * second->getData();
    auto ret = make_shared<gradTensor<T>>(newData);
    ret->setSource(make_shared<mulBackward<T>>(first, second));
    return ret;
}

template <typename T>
shared_ptr<gradTensor<T>> operator/(shared_ptr<gradTensor<T>> first,
                                    shared_ptr<gradTensor<T>> second) {
    if (first->getData().shape() != second->getData().shape()) {
        throw invalid_argument("Shape mismatch for /");
    }
    auto newData = first->getData() / second->getData();
    auto ret = make_shared<gradTensor<T>>(newData);
    ret->setSource(make_shared<divBackward<T>>(first, second));
    return ret;
}

// ===================== Main =====================
int main() {
    xt::xarray<double> tensor = {1.0, 2.0, 3.0};
    xt::xarray<double> tensor1 = {4.0, 5.0, 6.0};

    auto x = make_shared<gradTensor<double>>(tensor);
    auto y = make_shared<gradTensor<double>>(tensor1);

    x = x * y; // elementwise multiply
    x->backward();

    cout << "x.grad = " << x->getGrad() << endl;
    cout << "y.grad = " << y->getGrad() << endl;
    //cout << "z.data = " << z->getData() << endl;
}
//before, we were creating new objects on the heap inside the operator with new gradTensor<T> and returning a reference to that, which is dangerous and leads to memory leaks or dangling references. Now, we use make_shared to create shared_ptr instances, ensuring proper memory management.