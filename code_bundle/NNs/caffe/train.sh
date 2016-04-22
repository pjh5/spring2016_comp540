#!/usr/bin/env sh

TOOLS=./build/tools


$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_my_solver.prototxt \


# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_my_solver_lr1.prototxt \
    --snapshot=examples/cifar10/cifar10_my_iter_30000.solverstate
