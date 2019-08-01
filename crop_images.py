import argparse
import cv2
import darknet
# import os
import pathlib
import shutil

from typing import List, Text, Tuple

Bbox = Tuple[float, float, float, float]


class YOLOV3Detector:
    def __init__(self, config: Text, weights: Text, meta: Text):
        '''
        :param config: config file path of YOLO V3 darknet
        :param weights: weights file path of YOLO V3 darknet
        :param meta: meta file path of YOLO V3 darknet
        '''
        self.config = config
        self.weights = weights
        self.meta = meta

    def detect(self, f_path: Text):
        '''detect objects
        :param f_path: image file path
        '''
        detections = darknet.performDetect(
                imagePath=f_path,
                configPath=self.config,
                weightPath=self.weights,
                metaPath=self.meta,
                showImage=False)

        return detections


def main():
    '''detect objects and crop detections from images to export
    '''
    # config
    a = get_args()
    output_dir = pathlib.Path(a.output)
    if output_dir.exists():
        confirmation = 'Delete {}?'.format(str(output_dir))
        refused = 'Cannot delete the file. Delete it or'
        refused += ' specify another file path and try again.'
        if get_user_confirmation(confirmation, refused):
            shutil.rmtree(str(output_dir))
        else:
            exit()
    output_dir.mkdir(parents=True)

    target_labels = a.labels.split(',')
    target_path = pathlib.Path(a.target)
    if target_path.is_file():
        src_files = [target_path]
    elif target_path.is_dir():
        src_files = target_path.glob('**/*.jpg')
    else:
        raise RuntimeError(
                'target must be file or directory, {}'.format(a.target))

    # object detector
    detector = YOLOV3Detector(a.conf, a.weights, a.meta)

    # detect images, crop all, and save
    for f_path in src_files:
        # TODO: support the case that target_path is file
        dest_dir = make_dest_dir_path(f_path, target_path, output_dir)
        detect_and_save_crops(f_path, dest_dir, detector, target_labels)

    # for f_path in src_files:
    #     print(f_path)
    #     detections = darknet.performDetect(
    #             imagePath=str(f_path),
    #             configPath=a.conf,
    #             weightPath=a.weights,
    #             metaPath=a.meta,
    #             showImage=False)
    #     print(detections)
    #     filtered_detections = take_target_labels(detections, target_labels)
    #     crop_all_and_save(f_path, filtered_detections, a.output)


def detect_and_save_crops(
        f_path: pathlib.Path,
        dest_dir: pathlib.Path,
        detector: YOLOV3Detector,
        obj_names: List[Text] = ['dog']):
    '''execute pipeline from reading image file to detect
    to crop to output cropped images
    :param f_path: image file path
    :param dest_dir: output root directory
    :param detector: object detector
    :param obj_names: object names to detect
    '''
    # detect target objects
    print('crop "{}"...'.format(str(f_path)))
    detections = detector.detect(str(f_path))
    filtered_detections = take_target_labels(detections, obj_names)
    # crop all and save
    crop_all_and_save(f_path, filtered_detections, dest_dir)


def get_args():
    psr = argparse.ArgumentParser()
    psr.add_argument('target', help='target directory or file path')
    psr.add_argument('-o', '--output', help='output directory')
    psr.add_argument(
            '--labels',
            help='specify label name that will be only extracted',
            required=False, default='dog,book')
    psr.add_argument(
            '--conf', help='config file of the detection model',
            required=False, default='cfg/yolov3-spp.cfg')
    psr.add_argument(
            '--weights', help='weights file of the detection model',
            required=False, default='weights/yolov3-spp.weights')
    psr.add_argument(
            '--meta', help='meta file of the detection model',
            required=False, default='cfg/coco.data')

    return psr.parse_args()


def get_user_confirmation(confirmation_msg: Text, refused_msg: Text) -> bool:
    '''require YES/NO answer on console
    '''
    while True:
        choice = input('{} [y/N]: '.format(confirmation_msg))
        if choice.upper() in ['Y', 'YES']:
            return True
        elif choice.upper() in ['N', 'NO']:
            print(refused_msg)
            return False


def take_target_labels(
        detections: List[Tuple[Text, float, Bbox]],
        target_labels: List[Text]):
    '''take only target labels from detections
    :param detections: return object from darknet object detection
    :param target_labels: target label name list to crop
    '''
    return [det for det in detections if det[0] in target_labels]


def make_dest_dir_path(
        f_path: pathlib.Path,
        target_root: pathlib.Path,
        dest_root: pathlib.Path):
    '''make destination directory path
    Example inputs:
        f_path = '/home/user/data/images/class10/sample.jpg'
        target_root = '/home/user/data'
        dest_root = '/home/hdd/dataset/cropped'
    The output is '/home/hdd/dataset/cropped/images/class10'
    :param f_path: input file path
    :param target_root: target root directory
    :param dest_root: destination root directory
    '''
    middle_dirs = f_path.parent.parts[len(target_root.parts):]
    middle_dir_path = pathlib.Path('/'.join(middle_dirs))
    dest_dir = dest_root.joinpath(middle_dir_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir


def crop_all_and_save(
        f_path: pathlib.Path,
        detections,
        dest_dir: pathlib.Path):
    '''crop all detections and save cropped images
    :param f_path: file path of the image
    :param detections: list of detections that contains label,
                       confidence score, and bbox coordinates
    :param dest_dir: output directory
    '''

    # open image
    img = cv2.imread(str(f_path))

    # crop and save
    for i, (_, conf, bbox) in enumerate(detections):
        # f_name = '{}_{:03d}.jpg'.format(
        #         os.path.splitext(os.path.split(f_path)[1])[0], i)
        # dest_path = os.path.join(dest_dir, f_name)
        dest_path = make_output_file_path(
                f_path, '{:03d}'.format(i), dest_dir)
        cv2.imwrite(str(dest_path), crop(img, bbox))
        print('saved', dest_path)


def make_output_file_path(
        f_path: pathlib.Path,
        suffix: Text,
        dest_dir: pathlib.Path):
    '''make output file path
    :param f_path: input file path
    :param suffix: suffix to add to the original file name
    :param dest_dir: destination directory
    '''
    if len(suffix):
        file_name = pathlib.Path(
                '{}_{}{}'.format(f_path.stem, suffix, f_path.suffix))
    else:
        file_name = f_path.name

    return dest_dir.joinpath(file_name)


def crop(img, bbox):
    '''crop image with bbox coordinates
    :param img: cv2 image
    :param bbox: bounding box in [x-center, y-center, width, height]
    '''
    imh, imw = img.shape[:2]
    xc, yc, bw, bh = bbox
    x0 = max(0, int(xc - (bw/2)))
    y0 = max(0, int(yc - (bh/2)))
    x1 = min(imw, int(xc + (bw/2)))
    y1 = min(imh, int(yc + (bh/2)))
    return img[y0:y1, x0:x1]


if __name__ == '__main__':
    main()
