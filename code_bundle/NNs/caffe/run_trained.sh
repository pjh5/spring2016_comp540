#!/usr/bin/env sh

TOOLS=./build/tools

# run trained model from the final step of optimization
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_my_solver_test.prototxt \
    --snapshot=examples/cifar10/cifar10_my_iter_40000.solverstate
