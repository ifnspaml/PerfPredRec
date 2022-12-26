import json
import os
from argparse import ArgumentParser
from scipy import stats

import numpy as np

from metrics import transform2array

NOISES = ["fgsm", "pgd", "gaussian", "s&p"]
METRICS = ['img_miou', 'miou', 'psnr', 'strength', 'pearson']

class Regression:
    def __init__(self, p=None, d=2):
        self.p = p
        if self.p is None:
            self.degree = d
        else:
            self.degree = len(self.p) - 1

    def __call__(self, x):
        return self._get_output(x)

    def _get_output(self, x):
        y = 0
        for i, p in enumerate(self.p):
            y += p * x ** (len(self.p) - 1 - i)
        return y

    def set_p(self, p):
        self.p = p
        self.degree = len(self.p) - 1

    def perform_regression(self, x, y):
        p = np.polyfit(x=x, y=y, deg=self.degree)
        self.p = p
        print(f"Regression parameters: {self.p}")

    def check_error(self, x, y):
        deltas = []
        for x_, y_ in zip(x, y):
            deltas += [self._get_output(x_) - y_]
        deltas = np.asarray(deltas)
        abs_= np.abs(deltas).mean()
        rootsqrt_= np.sqrt((deltas ** 2).mean())
        return abs_, rootsqrt_


def collect_data(metrics):
    psnr_ = []
    img_miou_ = []
    for i, noise in enumerate(NOISES):
        psnr_ += list(metrics[noise]["all"]["psnr"])
        img_miou_ += list(metrics[noise]["all"]["img_miou"])
    return np.asarray(psnr_), np.asarray(img_miou_)


def main(args):
    """
    main function

    :param args: parsed arguments
    :return:
    """

    """ Load the metrics json """
    with open(os.path.join('../output/', args.calibration_file)) as f:
        metrics_calibration = json.load(f)
        metrics_calibration = transform2array(metrics_calibration)

    with open(os.path.join('../output/', args.regression_file)) as f:
        metrics_regression = json.load(f)
        metrics_regression = transform2array(metrics_regression)

    """ Collect Data for Calibration """
    psnr, miou = collect_data(metrics_calibration)

    """ Calibrate Regression """
    reg = Regression(d=args.d)
    reg.perform_regression(psnr, miou)

    """ Collect Data for mIoU Prediction """
    psnr, miou = collect_data(metrics_regression)

    """ Compute Error Metrics"""
    abs_, rootsqrt_ = reg.check_error(psnr, miou)
    print(f'{stats.pearsonr(psnr, miou)[0]:.2f} &',
          f'{abs_:.2f} &',
          f'{rootsqrt_:.2f}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--calibration_file',
                        type=str,
                        help="file to load from output folder to calibrate regression",
                        default="cityscapes-val__swiftnet_nospp.txt")
    parser.add_argument('--regression_file',
                        type=str,
                        help="file to load from output folder to predict mIoU from PSNR.",
                        default="cityscapes-test__swiftnet_nospp.txt")
    parser.add_argument('--d',
                       type=int,
                       default=2)
    main(parser.parse_args())
