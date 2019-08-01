# object-cropper
Object cropping tool

# Requirements

- nvidia-docker
- [yolov3-docker image](https://github.com/reouno/yolov3-docker)

# Description

- need `cfg` directory copied from https://github.com/reouno/darknet/tree/docker/cfg
- need `data` directory copied from https://github.com/reouno/darknet/tree/docker/data
- need to download weight file that is in `weights` directory in this sample project
  - download links: https://github.com/AlexeyAB/darknet#pre-trained-models
- need to export `PYTHONPATH=/opt/darknet` in which directory the darknet module is installed

# How to use

Just run `./run.sh` for sample usage. It will output `sample_images/cropped_images`.
