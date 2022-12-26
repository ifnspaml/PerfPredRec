from __future__ import absolute_import, division, print_function

import csv
import json
import os
import random
import time

# Common Imports
import dataloader.pt_data_loader.mytransforms as mytransforms
from dataloader.definitions.labels_file import labels_cityscape_seg
from dataloader.eval.metrics import SegmentationRunningScore
from dataloader.file_io.get_path import GetPath
from dataloader.pt_data_loader.specialdatasets import StandardDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

# pt_models Imports to use common models
from models.swiftnet import SwiftNet
from train.losses import *
from train.plotter import *
# Import local dependencies
from train_options import SwiftNetOptions


def _init_fn(worker_id):
    seed_worker = worker_seed + worker_id
    random.seed(seed_worker)
    torch.manual_seed(seed_worker)
    torch.cuda.manual_seed(seed_worker)
    torch.cuda.manual_seed_all(seed_worker)
    np.random.seed(seed_worker)


class Trainer:
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
        keys_to_load = ['color', 'segmentation_trainid']
        self.keys_to_load = keys_to_load

        labels = labels_cityscape_seg.getlabels()  # original labels used by Cityscapes
        train_ids = [labels[i].trainId for i in range(len(labels))]
        self.num_classes_wo_bg = len(set(train_ids)) - 1  # 19

        self.interpolated = not (self.opt.width == 2048 and self.opt.height == 1024)

        '''     Specifiy DataLoader and which data transforms to use for training   '''
        train_data_transforms = [mytransforms.RandomHorizontalFlip(), mytransforms.CreateScaledImage()]
        train_data_transforms.append(mytransforms.RandomRescale((0.5, 2)))  # Careful when modifying: Scale misleading!
        train_data_transforms.append(
            mytransforms.RandomCrop((self.opt.crop_height, self.opt.crop_width), pad_if_needed=True))
        train_data_transforms.append(mytransforms.CreateColoraug())  # Important otherwise keys can not be found!
        train_data_transforms.append(mytransforms.RemoveOriginals())
        train_data_transforms.append(mytransforms.ConvertSegmentation())
        train_data_transforms.append(mytransforms.ToTensor())

        if self.opt.zeromean:
            train_data_transforms.append(mytransforms.NormalizeZeroMean())

        train_dataset = StandardDataset(dataset=self.opt.dataset,
                                        trainvaltest_split=self.opt.trainvaltest_split,
                                        labels_mode='fromtrainid',
                                        labels=labels,
                                        keys_to_load=keys_to_load,
                                        data_transforms=train_data_transforms,
                                        )
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
                                      labels_mode='fromtrainid',
                                      labels=labels,
                                      keys_to_load=keys_to_load,
                                      data_transforms=val_data_transforms,
                                      folders_to_load=['leftimg8bit/val/lindau', 'lindau'])
        self.val_loader = DataLoader(val_dataset, batch_size=self.opt.batch_size_val, shuffle=False,
                                     num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        self.val_iter = iter(self.val_loader)

        print("Using dataset:\n  ", self.opt.dataset)
        print("There are {:d} training samples and {:d} validation samples\n".format(len(train_dataset),
                                                                                     len(val_dataset)))

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
                                        lr_scale=100 / self.opt.learning_rate_fine_tune,
                                        loss_scale=100,
                                        mode="swiftnet",
                                        max_epoch=self.opt.num_epochs)
        _verbose_info("Base path for loading parameters and saving outputs: ", self.base_path)
        print("Basepath: Models files are saved to:\n  ", self.base_path)

        # --------------------------------------------------------------------------------
        # Model definition and state checkpoint recovery
        # --------------------------------------------------------------------------------

        # Model architecture definition is loaded here
        self.model_name = self.opt.model_name
        if self.model_name == 'SwiftNet':
            self.model = SwiftNet(self.num_classes_wo_bg)
        else:
            ValueError(f"The model name {self.model_name} is not supported.")

        _verbose_info(self.model)
        self.model.to(self.device)
        print("Model definition based on", self.model_name, "was loaded into: ", self.device)

        # --------------------------------------------------------------------------------
        # Handle deterministic, classweighing, etc. options
        # --------------------------------------------------------------------------------

        """ deterministic behaviour """
        if self.opt.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print('Set random seed to: ' + str(seed), flush=True)

        """ Apply classweights due to imbalances (Cityscapes only!)"""
        if self.opt.classweighing:
            # create instances of classes used in the loss
            weight = torch.ones(self.num_classes_wo_bg, dtype=torch.float)
            weight[0] = 2.8149201869965
            weight[1] = 6.9850029945374
            weight[2] = 3.7890393733978
            weight[3] = 9.9428062438965
            weight[4] = 9.7702074050903
            weight[5] = 9.5110931396484
            weight[6] = 10.311357498169
            weight[7] = 10.026463508606
            weight[8] = 4.6323022842407
            weight[9] = 9.5608062744141
            weight[10] = 7.8698215484619
            weight[11] = 9.5168733596802
            weight[12] = 10.373730659485
            weight[13] = 6.6616044044495
            weight[14] = 10.260489463806
            weight[15] = 10.287888526917
            weight[16] = 10.289801597595
            weight[17] = 10.405355453491
            weight[18] = 10.138095855713
            # weight[19] = 0

            if self.device.type == 'cuda':
                weight = weight.cuda()

        # --------------------------------------------------------------------------------
        # Configure Optimization while training
        # --------------------------------------------------------------------------------

        """ Define different parameter sets"""
        fine_tune_params = self.model.fine_tune_params()
        random_init_params = self.model.random_init_params()

        """ Specify Loss Criterion """
        if self.opt.classweighing:
            criterion = CrossEntropyLoss(weight=weight, ignore_index=255, ignore_background=True,
                                         device=self.device)
        else:
            criterion = CrossEntropyLoss(ignore_index=255, ignore_background=True, device=self.device)

        if self.device.type == 'cuda':
            criterion = criterion.cuda()
        print("Using Loss criterion:" + str(type(criterion)), flush=True)
        self.criterion = criterion

        """ 
        Optimizer Setting 
        """
        if self.opt.optimizer == "Adam":
            self.optimizer_fine_tune = Adam(fine_tune_params, lr=self.opt.learning_rate_fine_tune, weight_decay=0.25e-4)
            self.optimizer_random = Adam(random_init_params, lr=self.opt.learning_rate_random_init, weight_decay=1e-4)
            eta_min_fine = self.opt.eta_min_fine_tune
            eta_min_random = self.opt.eta_min_random_init
        else:
            ValueError("Currently, no optimizer other than ADAM is supported")

        print("\nInitial Optimizer settings for fine_tune_params with " + str(type(self.optimizer_fine_tune)) + " :")
        for param_group in self.optimizer_fine_tune.param_groups:
            print("  Initial Learning rate: " + str(param_group['lr']))
            print("  Weight Decay:  " + str(param_group['weight_decay']))

        print("\nInitial Optimizer settings for random_init_params with " + str(type(self.optimizer_random)) + " :")
        for param_group in self.optimizer_random.param_groups:
            print("  Initial Learning rate: " + str(param_group['lr']))
            print("  Weight Decay:  " + str(param_group['weight_decay']))

        """ 
        LR Scheduler 
        """
        if self.opt.LRscheduler == "CosineAnnealing":
            print("\nCosine AnnealingLR: Eta_min_fine: " + str(eta_min_fine)
                  + " Eta_min_random: " + str(eta_min_random))
            self.scheduler_fine_tune = lr_scheduler.CosineAnnealingLR(self.optimizer_fine_tune,
                                                                      T_max=self.opt.num_epochs, eta_min=eta_min_fine)
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
                myfile.write("Epoch\t\t\tTrain.-loss\t\tVal.-loss\t\tVal.-IoU\t\tLR_fine")
        self.csv_log_file = self.base_path + "/" + self.opt.savedir + "_plot_log.csv"
        # CSV for tikz_plots
        if (not os.path.exists(self.csv_log_file)):  # dont add first line if it exists
            with open(self.csv_log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'tr_loss', 'val_loss', 'val_miou', 'lr_fine', 'lr_random'])

        """ Metric setup """
        self.metric_model = SegmentationRunningScore(self.num_classes_wo_bg)
        self.best_iou = 0
        self.time_loss = []
        self.average_epoch_loss_train = 0

        """ Save run options """
        self.save_opts()

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
        time_train = []
        time_aug = []
        temp_time_aug = time.time()  # just to initialize with some value
        self.time_loss = []

        for param_group in self.optimizer_fine_tune.param_groups:
            self.usedLr_fine_tune = float(param_group['lr'])

        for param_group in self.optimizer_random.param_groups:
            self.usedLr_random = float(param_group['lr'])

        print("   LEARNING RATES --> fine-tune params:", self.usedLr_fine_tune,
              " random-init params:", self.usedLr_random, flush=True)

        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            time_aug.append(before_op_time - temp_time_aug)
            outputs, losses = self.process_batch(inputs)

            """ Backpropagate """
            self.optimizer_fine_tune.zero_grad()
            self.optimizer_random.zero_grad()
            losses["loss"].backward()
            epoch_loss.append(losses["loss"].item())

            """ Do Optimization step """
            self.optimizer_fine_tune.step()
            self.optimizer_random.step()
            time_train.append(time.time() - before_op_time)

            self.step += 1
            temp_time_aug = time.time()
        """ Train time measures """
        average_aug_delay = sum(time_aug) / len(time_aug)
        average_samples_per_sec = self.opt.batch_size_train * len(time_train) / sum(time_train)
        average_time_loss = sum(self.time_loss) / len(self.time_loss)

        self.average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        print('   Summary --> Train loss: {: 3.8f} with {: 4.2f} examples/s just for the inference and backpropagate '
              ' + optim part'.format(self.average_epoch_loss_train, average_samples_per_sec))
        print('               Timing analysis: DataLoader + augmentation per batch: {: 4.4f}s);'
              ' time for calculating loss per batch: {: 4.4f}s'.format(average_aug_delay, average_time_loss))

        self.scheduler_fine_tune.step()
        self.scheduler_random.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        input_color = inputs[("color_aug", 0, 0)]
        seg_logits = self.model(input_color)
        seg_map = torch.argmax(seg_logits, dim=1)
        outputs = {'segmentation_logits': seg_logits, 'segmentation': seg_map}
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def compute_losses(self, inputs, outputs):
        """Compute the losses for a minibatch
        """
        losses = {}
        preds = outputs['segmentation_logits'].float()
        targets = inputs[(self.keys_to_load[1], 0, 0)][:, 0, :, :].long()
        start_time_loss = time.time()
        criterion_loss = self.criterion(preds, targets)
        self.time_loss.append(time.time() - start_time_loss)
        losses["loss"] = criterion_loss
        return losses

    def compute_segmentation_losses(self, inputs, outputs):
        label_true = np.array(inputs[(self.keys_to_load[1], 0, 0)].cpu())[:, 0, :, :]
        label_pred = np.array(outputs['segmentation'].cpu())
        self.metric_model.update(label_true, label_pred)
        metrics = self.metric_model.get_scores()
        self.metric_model.reset()
        return metrics

    def update_val_metric_model(self, inputs, outputs):
        label_true = np.array(inputs[(self.keys_to_load[1], 0, 0)].cpu())[:, 0, :, :]
        label_pred = np.array(outputs['segmentation'].cpu())
        self.metric_model.update(label_true, label_pred)

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
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("optim_fine"))
        torch.save(self.optimizer_fine_tune.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("optim_random"))
        torch.save(self.optimizer_random.state_dict(), save_path)

    def full_validation(self):
        # Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", self.epoch + 1, " on full val set -----", flush=True)
        epoch_loss_val = []
        time_val = []
        self.set_eval()
        for batch_idx, inputs_val in enumerate(self.val_loader):
            start_time = time.time()
            with torch.no_grad():
                outputs_val, losses_val = self.process_batch(inputs_val)
                time_val.append(time.time() - start_time)
                epoch_loss_val.append(losses_val["loss"].item())
                if (self.keys_to_load[1], 0, 0) in inputs_val:
                    self.update_val_metric_model(inputs_val, outputs_val)
        metrics_val = self.metric_model.get_scores()
        self.metric_model.reset()
        self.set_train()
        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        print(
            '   Summary --> Train loss: {: 3.8f} Val loss: {: 3.8f} --> mIoU(val): {: 4.3f}% --> meanAcc(val): {: 4.3f}%'.format(
                self.average_epoch_loss_train, average_epoch_loss_val, metrics_val["meaniou"] * 100,
                                                                       metrics_val["meanacc"] * 100))
        self.save_best_epoch(metrics_val["meaniou"] * 100, self.average_epoch_loss_train, average_epoch_loss_val,
                             self.usedLr_fine_tune, self.usedLr_random)

    def save_best_epoch(self, iou_acc, average_epoch_loss_train, average_epoch_loss_val, currentLR_fine,
                        currentLR_random):
        """ Remember best val_IoU and save checkpoint """
        current_iou_val = iou_acc
        is_best = current_iou_val > self.best_iou
        self.best_iou = max(current_iou_val, self.best_iou)
        with open(self.automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.8f" % (
                self.epoch + 1, average_epoch_loss_train, average_epoch_loss_val, current_iou_val, currentLR_fine))
        with open(self.csv_log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [self.epoch + 1, average_epoch_loss_train, average_epoch_loss_val, current_iou_val, currentLR_fine,
                 currentLR_random])

        self.loss_plotter.append_data(epoch_=self.epoch + 1,
                                      train_loss=average_epoch_loss_train,
                                      current_lr=currentLR_fine,
                                      val_loss=average_epoch_loss_val,
                                      mIoU=current_iou_val)

        self.loss_plotter.save_loss_fig()

        if (is_best):
            self.save_model_state()
            print(f'-->saved: epoch: {self.epoch + 1} as new best)', flush=True)
            with open(self.base_path + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val.-IoU= %.4f" % (self.epoch + 1, current_iou_val))


if __name__ == "__main__":
    options = SwiftNetOptions()
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
