import os
from argparse import ArgumentParser
import json
import numpy as np
import scipy.stats as stats

from metrics import transform2array

NOISES = ["fgsm", "pgd", "gaussian", "s&p"]
METRICS = ['img_miou', 'miou', 'psnr', 'strength', 'pearson']
EPSILON = ["0.25", "0.5", "0", "1.0", "2.0", "4.0", "8.0", "12.0", "16.0", "20.0", "24.0", "28.0", "32.0"]


def compute_single_noise_single_epsilon(metrics, noise, epsilon):
    dict_ = {}
    psnr = np.asarray(metrics[noise][str(epsilon)]["psnr"])
    img_miou = np.asarray(metrics[noise][str(epsilon)]["img_miou"])
    miou = np.asarray(metrics[noise][str(epsilon)]["miou"])
    dict_["miou"] = np.mean(miou)
    dict_["img_miou"] = np.mean(img_miou)
    dict_["psnr"] = np.mean(psnr)
    dict_["pearson"] = stats.pearsonr(psnr, img_miou)[0]
    return dict_

def compute_all_noises(metrics):
    dict_ = {}
    psnr_ = []
    miou_ = []
    img_miou_ = []
    for i, noise in enumerate(NOISES):
        psnr = metrics[noise]["all"]["psnr"]
        img_miou = metrics[noise]["all"]["img_miou"]
        miou = metrics[noise]["all"]["miou"]
        psnr_ += list(psnr)
        miou_ += list(miou)
        img_miou_ += list(img_miou)
    miou_ = np.asarray(miou_)
    psnr_ = np.asarray(psnr_)
    img_miou_ = np.asarray(img_miou_)
    dict_["miou"] = np.mean(miou_)
    dict_["img_miou"] = np.mean(img_miou_)
    dict_["psnr"] = np.mean(psnr_)
    dict_["pearson"] = stats.pearsonr(psnr_, img_miou_)[0]
    return dict_

def compute_single_noise(metrics, noise):
    dict_ = {}
    psnr = np.asarray(metrics[noise]["all"]["psnr"])
    img_miou = np.asarray(metrics[noise]["all"]["img_miou"])
    miou = np.asarray(metrics[noise]["all"]["miou"])
    dict_["miou"] = np.mean(miou)
    dict_["img_miou"] = np.mean(img_miou)
    dict_["psnr"] = np.mean(psnr)
    dict_["pearson"] = stats.pearsonr(psnr, img_miou)[0]
    return dict_

def main(args):
    """
    main function

    :param args: parsed arguments
    :return:
    """

    """ Load the metrics json """
    file = args.file
    with open(os.path.join('../output/', f"{file}.txt")) as f:
        metrics = json.load(f)

    """ Transform all metrics entries to numpy arrays if necessary """
    metrics = transform2array(metrics)

    """ Compute metrics on a single epsilon value """
    print(f"\n=== Report metrics on a single epsilon value ({args.single_epsilon}) ===")
    dicts = []

    """ Collect metrics for clean inputs (therefore epsilon=0.0). Here 'noise' is arbitrary. """
    dict_ = compute_single_noise_single_epsilon(metrics, noise=args.single_noise, epsilon="0.0")
    dicts.append(dict_)

    """ Collect metrics for corrupted inputs with given noise strength """
    for noise in NOISES:
        dict_ = compute_single_noise_single_epsilon(metrics, noise=noise, epsilon=args.single_epsilon)
        dicts.append(dict_)

    print("mIoU:")
    print(f'Clean: {dicts[0]["miou"]:.2f} ||',
          f'FGSM: {dicts[1]["miou"]:.2f} ||',
          f'PGD: {dicts[2]["miou"]:.2f} ||',
          f'Gaussian: {dicts[3]["miou"]:.2f} ||',
          f'S&P: {dicts[4]["miou"]:.2f}')

    print("Img mIoU:")
    print(f'Clean: {dicts[0]["img_miou"]:.2f} ||',
          f'FGSM: {dicts[1]["img_miou"]:.2f} ||',
          f'PGD: {dicts[2]["img_miou"]:.2f} ||',
          f'Gaussian: {dicts[3]["img_miou"]:.2f} ||',
          f'S&P: {dicts[4]["img_miou"]:.2f}')

    print("PSNR:")
    print(f'Clean: {dicts[0]["psnr"]:.2f} ||',
          f'FGSM: {dicts[1]["psnr"]:.2f} ||',
          f'PGD: {dicts[2]["psnr"]:.2f} ||',
          f'Gaussian: {dicts[3]["psnr"]:.2f} ||',
          f'S&P: {dicts[4]["psnr"]:.2f}')

    print(f"\n\n=== Report metrics and correlations computed on all epsilons of each noise/attack ===\n")
    dicts = []
    dict_ = compute_single_noise_single_epsilon(metrics, noise=args.single_noise, epsilon="0.0")
    dicts.append(dict_)
    print("Clean:", dict_)
    for noise in NOISES:
        dict_ = compute_single_noise(metrics, noise=noise)
        dicts.append(dict_)
        print(f"{noise.upper()}: {dict_}")

    dict_ = compute_all_noises(metrics)
    dicts.append(dict_)
    print("all:", dict_)

    print("\nPearson correlations:")
    print(f'Clean: {dicts[0]["pearson"]:.2f} ||',
          f'FGSM: {dicts[1]["pearson"]:.2f}  ||',
          f'PGD: {dicts[2]["pearson"]:.2f} ||',
          f'Gaussian: {dicts[3]["pearson"]:.2f} ||',
          f'S&P: {dicts[4]["pearson"]:.2f} ||',
          f'all: {dicts[5]["pearson"]:.2f}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file',
                        type=str,
                        default="cityscapes-test__swiftnet_nospp",
                        help="file to load from output folder")
    parser.add_argument('--single_noise',
                        type=str,
                        choices=NOISES + ['all'],
                        default='fgsm',
                        help="Select for single noise statistics.")
    parser.add_argument('--single_epsilon',
                        type=str,
                        choices=EPSILON + ['all'],
                        default='8.0',
                        help="Select for single epsilon statistics.")
    main(parser.parse_args())

