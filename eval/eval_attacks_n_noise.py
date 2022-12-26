from __future__ import absolute_import, division, print_function

import operator
import os

import cv2
# Import dataloader
import dataloader.pt_data_loader.mytransforms as mytransforms
import torch
import torch.nn.functional as func
import torchvision as tv
# Import local dependencies
from attack import Attack
from dataloader.definitions.labels_file import *
from dataloader.eval.metrics import SegmentationRunningScore
from dataloader.file_io.get_path import GetPath
from dataloader.pt_data_loader.specialdatasets import StandardDataset
from eval_options import EvalOptions
from model_architectures import load_model_def, load_model_state
# Import robustness tool
from robustness.helper.metrics import MetricsCollector
from robustness.helper.model import ModelWrapper
from scipy import stats
from torch.utils.data import DataLoader

#cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

# Mean/Var definition
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Metrisc that are used during evaluation
METRICS = ['img_miou', 'miou', 'psnr', 'strength', 'pearson']

ATTACKS = ['fgsm', 'pgd', 'gaussian', 's&p']


class Evaluator:
    def __init__(self, options):
        # --------------------------------------------------------------------------------
        # Setup
        # --------------------------------------------------------------------------------
        self.opt = options

        '''     Use '_verbose_info' as 'print' if flag --verbose is specified.  '''
        _verbose_info = print if self.opt.verbose else lambda *a, **k: None

        '''     Remember device type    '''
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        _verbose_info('Using device: ', self.device)

        # --------------------------------------------------------------------------------
        # Configure path settings
        # --------------------------------------------------------------------------------

        path_getter = GetPath()
        checkpoint_path = path_getter.get_checkpoint_path()
        self.base_path = os.path.join(checkpoint_path, self.opt.model_name, self.opt.model_state_name)
        _verbose_info("Base path for loading parameters:", self.base_path)

        # --------------------------------------------------------------------------------
        # Dataset loading and preparation
        # --------------------------------------------------------------------------------

        '''     Load Label and key definitons   '''
        keys_to_load = ['color', 'segmentation_trainid']
        if self.opt.dataset == 'cityscapes':
            self.trainvaltest_split = "validation"
            print("As cityscapes is used note that 'trainvaltest_split' is set to 'validation' ")
            labels = labels_cityscape_seg.getlabels()
            labels_mode = 'fromtrainid'
            if self.opt.subset == 'val':
                folders_to_load = ['leftimg8bit/val/lindau', 'lindau']
            elif self.opt.subset == 'test':
                folders_to_load = ['leftimg8bit/val/frankfurt', 'frankfurt',
                                   'leftimg8bit/val/munster', 'munster']
            else:
                folders_to_load = None
        elif self.opt.dataset == 'kitti_2015':
            self.trainvaltest_split = "train"
            print("As kitti_2015 is used note that 'trainvaltest_split' is set to 'train'")
            labels = labels_kitti_seg.getlabels()
            labels_mode = 'fromid'
            keys_to_load = ['color', 'segmentation']
        else:
            raise ValueError(f"The dataset {self.opt.dataset} is not supported.")

        train_ids = [labels[i].trainId for i in range(len(labels))]
        self.num_classes_wo_bg = len(set(train_ids)) - 1

        self.interpolated = self.opt.dataset == 'cityscapes' and not (
                self.opt.width == 2048 and self.opt.height == 1024)

        '''     Specifiy which data transforms to use   '''
        test_data_transforms = [mytransforms.CreateScaledImage()]
        if self.interpolated:
            test_data_transforms.append(mytransforms.Resize((self.opt.height, self.opt.width), image_types=['color']))
            test_data_transforms.append(mytransforms.RemoveOriginals())
        test_data_transforms.append(mytransforms.ConvertSegmentation())
        test_data_transforms.append(
            mytransforms.CreateColoraug())  # Adjusts keys so that NormalizeZeroMean() finds them
        test_data_transforms.append(mytransforms.ToTensor())
        test_data_transforms.append(mytransforms.Relabel(255, -100))  # -100 is PyTorch's default ignore label
        test_data_transforms.append(mytransforms.RemapKeys({
            ('segmentation_trainid', 0, 0): ('segmentation', 0, 0)
        }))

        self.test_data_transforms = test_data_transforms

        '''     Configure DataLoader    '''
        test_dataset = StandardDataset(dataset=self.opt.dataset,
                                       trainvaltest_split=self.trainvaltest_split,
                                       labels_mode=labels_mode,
                                       keys_to_load=keys_to_load,
                                       data_transforms=test_data_transforms,
                                       folders_to_load=folders_to_load)

        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=self.opt.num_workers,
                                      pin_memory=True,
                                      drop_last=False)

        print("-> Dataset: ", self.opt.dataset, self.trainvaltest_split, "set.")
        print("   There are", len(test_dataset), "samples for evaluation.")

        # Variables for zero mean normalization
        self.mean = torch.tensor(MEAN).reshape(1, -1, 1, 1).to(self.device)
        self.std = torch.tensor(STD).reshape(1, -1, 1, 1).to(self.device)

        # --------------------------------------------------------------------------------
        # Model definition and state checkpoint recovery
        # --------------------------------------------------------------------------------

        # Model architecture definition is loaded here
        self.model_name = self.opt.model_name
        if self.model_name == "SwiftNet":
            self.model = load_model_def(self.model_name, self.num_classes_wo_bg)
        elif self.model_name == 'SwiftNetRec':
            self.model = load_model_def(self.model_name, self.num_classes_wo_bg, rec_decoder=self.opt.rec_decoder,
                                        lateral=self.opt.lateral)
        else:
            raise ValueError(f"The Model {self.opt.dataset} is not supported.")
        self.model = load_model_state(self.model, self.model_name, self.base_path, self.opt.weights_epoch)
        if self.opt.zeroMean:
            preprocess = tv.transforms.Normalize(MEAN, STD)
        else:
            preprocess = None

        self.model = ModelWrapper(self.model, preprocessing=preprocess, postprocessing=None)
        self.model.to(self.device)
        self.model.eval()
        _verbose_info(self.model.state_dict())

        # Create a model copy which only outputs a semantic segmentation. This is needed for the attack tool.
        if self.model_name == 'SwiftNetRec':
            # add a postprocessing to have only segmentation (important for attacks)
            self.model_seg_only = ModelWrapper(self.model, preprocessing=preprocess,
                                               postprocessing=self._remove_rec_output)
            self.model_seg_only.to(self.device)
            self.model_seg_only.eval()
        else:
            self.model_seg_only = self.model

        # --------------------------------------------------------------------------------
        # Metric definition
        # --------------------------------------------------------------------------------

        self.metric_model = SegmentationRunningScore(self.num_classes_wo_bg)
        self.metric_model_image = SegmentationRunningScore(self.num_classes_wo_bg)
        self.metric_collector = MetricsCollector(corruptions=ATTACKS,
                                                 severities=self.opt.epsilon,
                                                 metrics=METRICS)

    def _dataset_loop(self, attack=None, attack_name=None):

        # Only important for Kitti:
        threshold = 50
        # The first 50 images are val images and the last 150 images are test images
        if self.opt.subset == 'val':
            op = operator.lt  # less than operator
        if self.opt.subset == 'test':
            op = operator.ge  # greater than operator

        for d, data in enumerate(self.test_loader):
            if op(d, threshold) or self.opt.dataset != 'kitti_2015':  # always true if not the kitti_2015 dataset
                with torch.no_grad():
                    """ Prepare groundTruth and InputImages """
                    gt_seg = data[('segmentation', 0, 0)].to(self.device).long()
                    input_color = data[("color_aug", 0, 0)].to(self.device)

                    if attack is not None:
                        input_, s = attack.apply_attack(attack_name, input_color, gt_seg[:, 0, :, :])
                        self.strength.append(s.cpu())
                    else:
                        input_ = input_color

                    """ Infer model """
                    outputs = self.model(input_)

                    """ Check whether the output is a dict or """
                    if isinstance(outputs, dict):
                        # Changes for dict output from swiftnet_rec:
                        seg_logits = outputs['logits']  # Semseg Output from the dict
                        input_rec = outputs['image_reconstruction']  # Autoencoder output from dict
                    else:
                        seg_logits = outputs
                        input_rec = None

                    """ If necessary interpolate predictions to native resolution and process argmax    """
                    if self.interpolated:
                        # Interpolation mode: here is just 'bilinear' an possible choice without errors (not! nearest)
                        interp_log = func.interpolate(seg_logits, gt_seg[0].shape, mode='bilinear', align_corners=False)
                        seg_map = torch.argmax(interp_log, dim=1)
                    else:
                        seg_map = torch.argmax(seg_logits, dim=1)

                    if input_rec is not None:
                        """ Reverse the ZeroMean Normalization """
                        # Map the Image Range from -2,. - +2,. to 0 - +1
                        if input_rec.shape != input_.shape:
                            input_rec = func.interpolate(input_rec, input_[0, 0, :, :].shape, mode='bilinear',
                                                      align_corners=False)
                        if self.opt.zeroMean:
                            # Zero Mean Normalization happens inside the AutoEncoder
                            input_rec = (input_rec * self.std) + self.mean

                        """ Calculate image metrics """
                        self.psnr_values.append(self._compute_psnr(input_, input_rec).item())

                    """ Queue in processed batch for metric evaluation  """
                    pred_seg = seg_map.cpu().numpy()
                    gt_seg = gt_seg[:, 0, :, :].cpu().numpy()
                    self.metric_model.update(gt_seg, pred_seg)
                    self.metric_model_image.update(gt_seg, pred_seg)
                    self.miou_values.append(self.metric_model_image.get_scores()['meaniou'])
                    self.metric_model_image.reset()
            else:
                pass

    def calculate_metrics_all(self):
        for attack_name in ATTACKS:
            print(f"=== {attack_name} ===")
            psnr_values_all = []
            miou_values_all = []
            strengths = []
            mious = []
            for epsilon in self.opt.epsilon:
                self.psnr_values = []
                self.miou_values = []
                self.strength = []
                """ loop through the dataset """
                print(f"Current epsilon: {epsilon}", flush=True)
                if epsilon == 0:
                    attack = None
                else:
                    attack = Attack(epsilon / 255, self.model_seg_only)
                self._dataset_loop(attack=attack, attack_name=attack_name)

                """ Compute Metrics """
                # Image metrics
                meanpsnr = self.psnr_values.copy()
                # Segmentation metrics
                mean_img_miou = self.miou_values.copy()
                meaniou = self.metric_model.get_scores()['meaniou'] * 100
                # Correlation metrics
                pearson = stats.pearsonr(self.miou_values, self.psnr_values)[0] if len(self.psnr_values) > 0 else 0.00
                # Distortion strength
                meanstrength = sum(self.strength) / len(self.strength) if len(self.strength) > 0 else 0

                """ Add to dict """
                # Image metrics
                self.metric_collector.add([attack_name, epsilon, "psnr"], meanpsnr)
                # Segmentation metrics
                self.metric_collector.add([attack_name, epsilon, "img_miou"], mean_img_miou)
                self.metric_collector.add([attack_name, epsilon, "miou"], meaniou)
                # Correlation metrics
                self.metric_collector.add([attack_name, epsilon, "pearson"], pearson)
                # Distortion strength
                self.metric_collector.add([attack_name, epsilon, "strength"], self.strength)

                print('mIoU | PSNR | Pearson | Strength')
                meanpsnr = sum(meanpsnr) / len(meanpsnr) if len(self.psnr_values) > 0 else 0.00
                print(f'{meaniou:8.4f} % | {meanpsnr:8.4f} dB | {pearson:.4f} | {meanstrength:.4f}')

                """ Collect & Reset metrics """
                self.metric_model.reset()
                miou_values_all += self.miou_values
                psnr_values_all += self.psnr_values
                strengths += self.strength
                mious += [meaniou]

            pearson = stats.pearsonr(miou_values_all, psnr_values_all)[0] if len(psnr_values_all) > 0 else 0.00

            # Image metrics
            self.metric_collector.add([attack_name, "all", "psnr"], psnr_values_all)
            # Segmentation metrics
            self.metric_collector.add([attack_name, "all", "img_miou"], miou_values_all)
            self.metric_collector.add([attack_name, "all", "miou"], mious)
            # Correlation metrics
            self.metric_collector.add([attack_name, "all", "pearson"], pearson)
            # Distortion strength
            self.metric_collector.add([attack_name, "all", "strength"], strengths)

        self.metric_collector.print()
        self.metric_collector.save(path='../output/',
                                   name=f"{self.opt.dataset}-{self.opt.subset}_"
                                   f"{self.opt.model_state_name}")
        print("\n-> Done!\n")

    @staticmethod
    def _compute_psnr(img, gen_img):
        # Compute MSE first
        mse = ((img - gen_img) ** 2).mean()

        # Compute PSNR with MSE
        if isinstance(mse, torch.Tensor):
            psnr = 10 * torch.log10(1 / mse)
        elif isinstance(mse, np.float32):
            psnr = 10 * np.log10(1 / mse)
        else:
            print(type(mse))
            exit('CHECK')
        return psnr

    @staticmethod
    def _remove_rec_output(x):
        assert isinstance(x, dict), "x is not a dictionary, which is expected"
        for key in x.keys():
            assert key in ['image_reconstruction', 'logits'], f"x has not the correct keys {x.keys()}"
        return x['logits']


if __name__ == "__main__":
    options = EvalOptions()
    opt = options.parse()
    evaluator = Evaluator(opt)
    interpolated = opt.dataset == 'cityscapes' and not (opt.width == 2048 and opt.height == 1024)
    if opt.dataset == "cityscapes":
        print("-> Computing predictions with image input size {}x{}".format(opt.width, opt.height))
    if interpolated:
        print("   Predictions will be interpolated to original label size before mIoU evaluation.")
    evaluator.calculate_metrics_all()
