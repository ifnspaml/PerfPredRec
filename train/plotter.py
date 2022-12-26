import matplotlib.pyplot as plt
import numpy as np

MODES = ["swiftnet_rec", "swiftnet"]

class LossPlotter:

    def __init__(self, path='loss_plot.pdf', lr_scale=1e6, loss_scale=1e2, max_epoch=200, mode="swiftnet"):
        super().__init__()
        self.path = path
        # For visualization
        self.lr_scale = lr_scale
        self.loss_scale = loss_scale
        self.x = []
        self.y_train_loss = []
        self.y_val_loss = []
        self.y_learning_rate = []
        self.y_psnr = []
        self.y_mIoU = []
        self.max_epoch = max_epoch
        assert mode in MODES
        self.mode = mode

    def append_data(self, **kwargs):
        if self.mode == 'swiftnet':
            self.append_data_swiftnet(**kwargs)
        elif self.mode == 'swiftnet_rec':
            self.append_data_swiftnet_rec(**kwargs)

    def save_loss_fig(self):
        if self.mode == 'swiftnet':
            self.save_loss_fig_segmentation_only()
        elif self.mode == 'swiftnet_rec':
            self.save_loss_fig_swiftnet_rec_standard()

    def append_data_swiftnet(self, epoch_, train_loss, current_lr, val_loss, mIoU):
        self.x.append(epoch_)
        self.y_train_loss.append(train_loss * self.loss_scale)
        self.y_learning_rate.append(current_lr * self.lr_scale)
        self.y_val_loss.append(val_loss * self.loss_scale)
        self.y_mIoU.append(mIoU)

    def append_data_swiftnet_rec(self, epoch_, train_loss, current_lr, val_loss, psnr):
        self.x.append(epoch_)
        self.y_train_loss.append(train_loss * self.loss_scale)
        self.y_learning_rate.append(current_lr * self.lr_scale)
        self.y_val_loss.append(val_loss * self.loss_scale)
        self.y_psnr.append(psnr)

    def save_loss_fig_swiftnet_rec_standard(self):
        fig = plt.figure(figsize=(3, 3), dpi=300)

        line_psnr, = plt.plot(self.x, self.y_psnr)
        plt.setp(line_psnr, linewidth=1.0, color='g')

        line_tr, = plt.plot(self.x, self.y_train_loss)
        plt.setp(line_tr, linewidth=1.0, color='m')

        line_val, = plt.plot(self.x, self.y_val_loss)
        plt.setp(line_val, linewidth=1.0, color='y')

        line_lr, = plt.plot(self.x, self.y_learning_rate)
        plt.setp(line_lr, linewidth=1.0, color='b')

        plt.xlim(xmin=0, xmax=self.max_epoch)
        plt.ylim(ymin=0, ymax=100)

        plt.xlabel('Epoch')

        plt.legend(['PSNR [dB]',
                    f'Train loss (*{str(self.loss_scale)})',
                    f'Val. loss (*{str(self.loss_scale)})',
                    f'LR (*{str(self.lr_scale)})'],
                   loc='upper right', fontsize='x-small')
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.savefig(self.path + '/Plot_PSNR_loss_Val.pdf', dpi=600, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close(fig)

    def save_loss_fig_segmentation_only(self):
        fig = plt.figure(figsize=(3, 3), dpi=300)

        line_mIoU, = plt.plot(self.x, self.y_mIoU)
        plt.setp(line_mIoU, linewidth=1.0, color='r')

        line_tr, = plt.plot(self.x, self.y_train_loss)
        plt.setp(line_tr, linewidth=1.0, color='m')

        line_val, = plt.plot(self.x, self.y_val_loss)
        plt.setp(line_val, linewidth=1.0, color='y')

        line_lr, = plt.plot(self.x, self.y_learning_rate)
        plt.setp(line_lr, linewidth=1.0, color='b')

        plt.xlim(xmin=0, xmax=self.max_epoch)
        plt.ylim(ymin=0, ymax=100)

        plt.xlabel('Epoch')

        plt.legend(['% mIoU (val)',
                    f'Train loss (*{str(self.loss_scale)})',
                    f'Val. loss (*{str(self.loss_scale)})',
                    f'LR (fine) (*{str(self.lr_scale)})'],
                   loc='upper right', fontsize='x-small')
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.savefig(self.path + '/Plot_mIoU_loss_Val.pdf', dpi=600, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close(fig)
