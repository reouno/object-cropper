#!/bin/sh
set -eu

docker run \
    -v `pwd`:/workspace \
    -it \
    --rm \
    --runtime=nvidia \
    yolov3:1.0 \
    bash -c "export PYTHONPATH=/opt/darknet && \
             cd /workspace
             python3 crop_images.py \
                 /workspace/sample_images/images \
                 -o /workspace/sample_images/cropped_images \
                 --conf cfg/yolov3-tiny.cfg \
                 --weights weights/yolov3-tiny.weights \
                 --labels dog"
