#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

/home/odyssey/project/caffe/.build_release/tools/compute_image_mean ./liuyao_train_lmdb \
  ./imagenet_mean.binaryproto

echo "Done."
