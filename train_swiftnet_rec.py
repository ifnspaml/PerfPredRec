from __future__ import absolute_import, division, print_function

import csv
import json
import os
import random
import time

# Common Imports
import dataloader.pt_data_loader.mytransforms as mytransforms
import scipy.stats as stats
import torch.nn.functional as func
from dataloader.definitions.labels_file import labels_cityscape_seg
from dataloader.eval.metrics import SegmentationRunningScore
from dataloader.file_io.get_path import GetPath
from dataloader.pt_data_loader.specialdatasets import StandardDataset

from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from metrics.metrics import *
# pt_models Imports to use common models
from models.swiftnet_rec import SwiftNetRec
from train.plotter import *
# Import local dependencies
from train_options import SwiftNetRecOptions


def _init_fn(worker_id):
    seed_worker = worker_seed + worker_id
    random.seed(seed_worker)
    torch.manual_seed(seed_worker)
    torch.cuda.manual_seed(seed_worker)
    torch.cuda.manual_seed_all(seed_worker)
    np.random.seed(seed_worker)


class Trainer:
    ZM_MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).permute(0, 3, 1, 2)
    ZM_STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).permute(0, 3, 1, 2)

    def __init__(self, options):
        self.opt = options

        '''     Use '_verbose_info' as 'print' if flag --verbose is specified.  '''
        _verbose_info = print if self.opt.verbose else lambda *a, **k: None

        '''     Remember device type    '''
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        _verbose_info('Using device: ', self.device)

        # --------------------------------------------------------------------------------
        # Dataset loading and preparation
        # --------------------------------------------------------------------------------

        '''     Load Label and key definitons   '''
        keys_to_load = ['color']

        if self.opt.dataset == 'cityscapes':
            labels = labels_cityscape_seg.getlabels()  # original labels used by Cityscapes
            self.interpolated = not (self.opt.width == 2048 and self.opt.height == 1024)
        train_ids = [labels[i].trainId for i in range(len(labels))]
        self.num_classes_wo_bg = len(set(train_ids)) - 1

        '''     Specifiy DataLoader and which data transforms to use for training   '''
        train_data_transforms = [mytransforms.RandomHorizontalFlip(), mytransforms.CreateScaledImage()]
        if self.interpolated:
            train_data_transforms.append(
                mytransforms.Resize((self.opt.height, self.opt.width), image_types=['color']))

            train_data_transforms.append(mytransforms.RemoveOriginals())
        train_data_transforms.append(mytransforms.RandomRescale((0.5, 2)))  # Careful when modifying: Scale misleading!
        train_data_transforms.append(
            mytransforms.RandomCrop((self.opt.crop_height, self.opt.crop_width), pad_if_needed=True))
        train_data_transforms.append(mytransforms.CreateColoraug())  # Important otherwise keys can not be found!
        train_data_transforms.append(mytransforms.RemoveOriginals())
        train_data_transforms.append(mytransforms.ConvertSegmentation())
        train_data_transforms.append(mytransforms.ToTensor())

        if self.opt.zeromean:
            train_data_transforms.append(mytransforms.NormalizeZeroMean())

        print("Used Dataset: ", self.opt.dataset, "with split", self.opt.trainvaltest_split)

        train_dataset = StandardDataset(dataset=self.opt.dataset,
                                        trainvaltest_split=self.opt.trainvaltest_split,
                                        keys_to_load=keys_to_load,
                                        data_transforms=train_data_transforms)
        self.train_loader = DataLoader(train_dataset, batch_size=self.opt.batch_size_train, shuffle=True,
                                       num_workers=self.opt.num_workers, worker_init_fn=_init_fn, pin_memory=True)

        '''      Specifiy DataLoader and which data transforms to use for validation   '''
        val_data_transforms = [mytransforms.CreateScaledImage()]
        if self.interpolated:
            val_data_transforms.append(mytransforms.Resize((self.opt.height, self.opt.width), image_types=['color']))
            val_data_transforms.append(mytransforms.RemoveOriginals())
        val_data_transforms.append(mytransforms.CreateColoraug())  # Adjusts keys so that NormalizeZeroMean() finds it
        val_data_transforms.append(mytransforms.ConvertSegmentation())
        val_data_transforms.append(mytransforms.ToTensor())
        if self.opt.zeromean:
            val_data_transforms.append(mytransforms.NormalizeZeroMean())

        val_dataset = StandardDataset(dataset=self.opt.dataset,
                                      trainvaltest_split='validation',
                                      labels=labels,
                                      keys_to_load=keys_to_load,
                                      data_transforms=val_data_transforms,
                                      folders_to_load=['leftimg8bit/val/lindau', 'lindau'])

        print("There are {:d} training samples and {:d} validation samples\n".format(len(train_dataset),
                                                                                     len(val_dataset)))

        dataset_size = len(val_dataset)
        indices = list(range(dataset_size))

        val_indices = indices[:dataset_size]
        valid_sampler = SequentialSampler(val_indices)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.opt.batch_size_val,
                                                      shuffle=False, num_workers=self.opt.num_workers,
                                                      pin_memory=True, drop_last=False, sampler=valid_sampler)

        self.val_iter = iter(self.val_loader)

        # --------------------------------------------------------------------------------
        # Configure path settings
        # --------------------------------------------------------------------------------

        path_getter = GetPath()
        checkpoint_path = path_getter.get_checkpoint_path()
        self.model_states_root_path = os.path.join(checkpoint_path, self.opt.model_name)
        self.base_path = os.path.join(self.model_states_root_path, self.opt.savedir)
        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path)
        self.loss_plotter = LossPlotter(path=self.base_path,
                                        lr_scale=100 / self.opt.learning_rate_random_init,
                                        loss_scale=1000,
                                        mode="swiftnet_rec",
                                        max_epoch=self.opt.num_epochs)

        _verbose_info("Base path for loading parameters and saving outputs: ", self.base_path)
        print("Basepath: Model files are saved to:\n  ", self.base_path)

        # --------------------------------------------------------------------------------
        # Model definition and state checkpoint recovery
        # --------------------------------------------------------------------------------

        # Model architecture definition is loaded here
        self.model_name = self.opt.model_name
        if self.model_name == 'SwiftNetRec':
            self.model = SwiftNetRec(self.num_classes_wo_bg, rec_decoder=self.opt.rec_decoder,
                                     lateral=self.opt.lateral)
        else:
            ValueError(f"The model name {self.model_name} is not supported.")

        _verbose_info(self.model)
        self.model.to(self.device)
        print("Model definition based on", self.model_name, "was loaded into: ", self.device)
        """ Load previous state """
        self.opt.load_model_state_name = None if self.opt.load_model_state_name == 'None' \
            else self.opt.load_model_state_name
        if self.opt.load_model_state_name is not None:
            self.load_model_state()

        # --------------------------------------------------------------------------------
        # Configure Optimization while training
        # --------------------------------------------------------------------------------

        """ Define different parameter sets"""
        random_init_params = self.model.random_init_params()

        criterion = MSELoss()
        print("Using Loss criterion:" + str(type(criterion)), flush=True)
        self.criterion = criterion

        """ 
        Optimizer Setting 
        """
        if self.opt.optimizer == "Adam":
            self.optimizer_random = Adam(random_init_params, lr=self.opt.learning_rate_random_init, weight_decay=1e-4)
            eta_min_random = self.opt.eta_min_random_init
        else:
            ValueError("Currently, no optimizer other than ADAM is supported")

        print("\nInitial Optimizer settings for random_init_params with " + str(type(self.optimizer_random)) + " :")
        for param_group in self.optimizer_random.param_groups:
            print("  Initial Learning rate: " + str(param_group['lr']))
            print("  Weight Decay:  " + str(param_group['weight_decay']))

        """ 
        LR Scheduler 
        """
        if self.opt.LRscheduler == "CosineAnnealing":
            print("Cosine AnnealingLR: Eta_min_random: " + str(eta_min_random))
            self.scheduler_random = lr_scheduler.CosineAnnealingLR(self.optimizer_random, T_max=self.opt.num_epochs,
                                                                   eta_min=eta_min_random)  # eta_min=1e-6)
        else:
            ValueError("Currently, no lr scheduling other than CosineAnnealing is supported")

        """
        For all the logging purposes... 
        """
        self.automated_log_path = self.base_path + "/automated_log.txt"
        if (not os.path.exists(self.automated_log_path)):  # dont add first line if it exists
            with open(self.automated_log_path, "a") as myfile:
                myfile.write("Epoch\t\t\tTrain.-loss\t\tVal.-loss\t\tPSNR(val)[dB]\t\tLR_fine")
        self.csv_log_file = self.base_path + "/" + self.opt.savedir + "_plot_log.csv"
        # CSV for tikz_plots
        if (not os.path.exists(self.csv_log_file)):  # dont add first line if it exists
            with open(self.csv_log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['epoch', 'tr_loss', 'val_loss', 'psnr_val', 'lr_fine', 'lr_random'])

        """
        Metric setup
        """

        self.time_loss = []
        self.average_epoch_loss_train = 0
        self.average_epoch_psnr_train = 0
        self.best_psnr = 0

        """
        Save run options
        """
        self.save_opts()

        """
        Put static variables to device
        """
        self.ZM_MEAN = Trainer.ZM_MEAN.to(self.device)
        self.ZM_STD = Trainer.ZM_STD.to(self.device)

    def set_train(self):
        """Convert all models to training mode
        """
        self.model.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.model.eval()

    def train(self):
        """Run the entire training pipeline
        """
        print("\n========== START TRAINING ===========", flush=True)

        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model_state()
            if ((self.epoch + 1) % self.opt.val_frequency == 0) or (self.epoch < 5) or \
                    (self.epoch > (self.opt.num_epochs - 37)):
                self.full_validation()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("----- TRAINING - EPOCH", self.epoch + 1, "-----", flush=True)

        epoch_loss = []
        epoch_psnr = []
        time_train = []
        time_aug = []
        temp_time_aug = time.time()  # just to initialize with some value
        self.time_loss = []

        for param_group in self.optimizer_random.param_groups:
            self.usedLr_random = float(param_group['lr'])

        print("   LEARNING RATES --> random-init params:", self.usedLr_random, flush=True)

        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            time_aug.append(before_op_time - temp_time_aug)

            # Training using clean images
            outputs, losses = self.process_batch(inputs)

            """ Backpropagate """
            self.optimizer_random.zero_grad()

            losses["loss"].backward()
            epoch_loss.append(losses["loss"].item())
            epoch_psnr.append(losses["psnr"].item())

            """ Do Optimization step """
            self.optimizer_random.step()

            time_train.append(time.time() - before_op_time)

            self.step += 1
            temp_time_aug = time.time()
        """ Train time measures """
        average_aug_delay = sum(time_aug) / len(time_aug)
        average_samples_per_sec = self.opt.batch_size_train * len(time_train) / sum(time_train)
        average_time_loss = sum(self.time_loss) / len(self.time_loss)

        self.average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        self.average_epoch_psnr_train = sum(epoch_psnr) / len(epoch_psnr)

        # Output Printings for the Slurm-Output Script:
        print(
            '   Summary --> Train loss: {:3.4f}  with {:.2f} examples/s just for the inference and backpropagate '
            ' + optim part'.format(self.average_epoch_loss_train, average_samples_per_sec))
        print('               Timing analysis: DataLoader + augmentation per batch: {: 4.4f}s;'
              ' time for calculating loss per batch: {: 4.4f}s'.format(average_aug_delay, average_time_loss))

        """ Do Scheduler epoch step """
        self.scheduler_random.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses"""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        input_color = inputs[("color_aug", 0, 0)]

        model_output = self.model(input_color)
        rec_decoder_output = model_output['image_reconstruction']
        outputs = {'rec_decoder_output': rec_decoder_output}
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def compute_losses(self, inputs, outputs):
        """Compute the losses for a minibatch"""

        rec_decoder_output = outputs['rec_decoder_output']
        input_image = inputs[("color_aug", 0, 0)]

        # Reverse the Zero-Mean Normalization
        if self.opt.zeromean:
            input_image = self.reverse_zero_mean(input_image)
        # The output of the reconstruction decoder is always zero-mean normalized
        # (see e.g. models/swiftnet_rec/resnet/full_network.py
        rec_decoder_output = self.reverse_zero_mean(rec_decoder_output)

        start_time_loss = time.time()
        loss = self.criterion(input_image, rec_decoder_output)
        self.time_loss.append(time.time() - start_time_loss)
        psnr = compute_psnr(input_image, rec_decoder_output)
        losses = {'loss': loss,
                  'psnr': psnr}
        return losses

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.base_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model_state(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.base_path, "models", "weights_{}".format(self.epoch + 1))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))

        to_save = self.model.state_dict()

        # use the zipfile_serialization just for older PyTorch Environments!
        torch.save(to_save, save_path, _use_new_zipfile_serialization=False)

    def load_model_state(self):
        """Load model(s) from disk
        """
        with open(os.path.join(self.model_states_root_path, self.opt.load_model_state_name, 'best.txt'), 'r') as f:
            best_epoch = f.readlines()[0].split(',')[0].split(' ')[-1]
            best_epoch = int(best_epoch)

        checkpoint_path = os.path.join(self.model_states_root_path, self.opt.load_model_state_name, 'models',
                                       'weights_{}'.format(best_epoch))
        assert os.path.isdir(checkpoint_path), "Cannot find folder {}".format(checkpoint_path)
        print("loading model from folder {}".format(checkpoint_path))

        path = os.path.join(checkpoint_path, "{}.pth".format('model'))
        model_dict = self.model.state_dict()

        # Setting for pretrained or random init weights:
        print('Training uses pretrained initialized weights')
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        self.model.load_state_dict(model_dict)

    def full_validation(self):
        # Validate on 500 val images after each epoch of training

        print("----- VALIDATING - EPOCH", self.epoch + 1, " on full val set -----", flush=True)
        epoch_loss_val = []
        epoch_psnr_val = []
        time_val = []
        self.set_eval()
        for batch_idx, inputs_val in enumerate(self.val_loader):
            start_time = time.time()
            with torch.no_grad():
                outputs_val, losses_val = self.process_batch(inputs_val)
                time_val.append(time.time() - start_time)

                epoch_loss_val.append(losses_val["loss"].item())
                epoch_psnr_val.append(losses_val["psnr"].item())

        self.set_train()

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        average_epoch_psnr_val = sum(epoch_psnr_val) / len(epoch_psnr_val)

        # Print a Summary for each epoch in the slurm output file:
        print('   Summary --> Validation loss: {:3.4f}'.format(average_epoch_loss_val))
        print('           --> Training:   PSNR: {:.2f} dB'.format(self.average_epoch_psnr_train))
        print('           --> Validation: PSNR: {:.2f} dB'.format(average_epoch_psnr_val))
        self.save_best_epoch(self.average_epoch_loss_train,
                             average_epoch_loss_val,
                             self.usedLr_random,
                             average_epoch_psnr_val)

    def save_best_epoch(self, average_epoch_loss_train, average_epoch_loss_val, currentLR_random,
                        average_epoch_psnr_val):
        """ Remember best val_PSNR and save checkpoint """

        is_best = average_epoch_psnr_val > self.best_psnr
        self.best_psnr = max(average_epoch_psnr_val, self.best_psnr)

        with open(self.automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t\t%.4f\t\t\t%.4f\t\t\t%.2f\t\t\t%.8f" % (
                self.epoch + 1, average_epoch_loss_train, average_epoch_loss_val, average_epoch_psnr_val, currentLR_random))

        with open(self.csv_log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [self.epoch + 1, average_epoch_loss_train, average_epoch_loss_val, average_epoch_psnr_val, currentLR_random,
                 currentLR_random])

        self.loss_plotter.append_data(epoch_=self.epoch + 1,
                                      train_loss=average_epoch_loss_train,
                                      current_lr=currentLR_random,
                                      val_loss=average_epoch_loss_val,
                                      psnr=average_epoch_psnr_val)

        self.loss_plotter.save_loss_fig()

        if (is_best):
            self.save_model_state()
            print(f'-->Saved epoch {self.epoch + 1} as new best!', flush=True)
            with open(self.base_path + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d with Val.-PSNR = %.2f dB." % (
                    self.epoch + 1, average_epoch_psnr_val))

    def reverse_zero_mean(self, batched_input_original):
        batched_input_zero_mean_reverse = batched_input_original * self.ZM_STD
        batched_input_zero_mean_reverse = batched_input_zero_mean_reverse + self.ZM_MEAN
        return batched_input_zero_mean_reverse


if __name__ == "__main__":
    options = SwiftNetRecOptions()
    opt = options.parse()

    # setting global seed values for determinism
    worker_seed = opt.worker_seed
    seed = opt.global_seed

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print('Set random seed to: ' + str(seed), flush=True)

    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # checking height and width are multiples of 32 --> Actually that shouldn't be a deal for SwiftNet (only ERFNet)
    assert opt.height % 32 == 0, "'height' must be a multiple of 32"
    assert opt.width % 32 == 0, "'width' must be a multiple of 32"

    trainer = Trainer(options=opt)
    trainer.train()
