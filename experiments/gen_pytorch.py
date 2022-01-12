import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from stylegan2_pytorch import Trainer, NanException

import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np


def cast_list(el):
    return el if isinstance(el, list) else [el]


def timestamped_filename(prefix='generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class GUIWindow:

    def modificate(self, ind, value):
        im = self.model.modImage(ind, value)
        image = QImage(im)
        self.imageModLabel.setPixmap(QPixmap.fromImage(image))

    def genetare(self):
        samples_name = timestamped_filename()
        self.leftBtn.setEnabled(True)
        self.rightBtn.setEnabled(True)
        im = self.model.genOneImage(f'{samples_name}')
        image = QImage(im)
        self.imageOrigLabel.setPixmap(QPixmap.fromImage(image))
        for slider in self.sliderList:
            slider.slider.setEnabled(True)
            slider.slider.setValue(0)

    def updateSliderRange(self):
        ind = 0
        begin = self.indexBlock * len(self.sliderList)
        end = (self.indexBlock + 1) * len(self.sliderList) - 1
        self.groupbox.setTitle(f'Settings {begin + 1}..{end + 1}')
        for slider in self.sliderList:
            slInd = self.indexBlock * len(self.sliderList) + ind
            slider.setIndex(slInd)
            slider.slider.setValue( self.model.getStyleKf(slInd) )
            ind = ind + 1

    def left(self):
        if self.indexBlock > 0:
            self.indexBlock = self.indexBlock - 1
            self.updateSliderRange()

    def right(self):
        if self.indexBlock < 50:
            self.indexBlock = self.indexBlock + 1
            self.updateSliderRange()

    def __init__(self, model):
        self.model = model
        self.indexBlock = 0
        self.app = QApplication(sys.argv)
        self.window = QWidget()
        self.window.setWindowTitle('StyleGAN2')
        self.window.setGeometry(300, 200, 490, 450)
        self.window.move(60, 15)

        self.genBtn = QPushButton('Generate', parent=self.window)
        self.genBtn.move(95, 10)
        self.genBtn.clicked.connect(self.genetare)

        self.leftBtn = QPushButton('<<', parent=self.window)
        self.leftBtn.move(10, 10)
        self.leftBtn.resize(25, 25)
        self.leftBtn.setEnabled(False)
        self.leftBtn.clicked.connect(self.left)

        self.rightBtn = QPushButton('>>', parent=self.window)
        self.rightBtn.move(235, 10)
        self.rightBtn.resize(25, 25)
        self.rightBtn.setEnabled(False)
        self.rightBtn.clicked.connect(self.right)

        self.imageOrigLabel = QLabel(parent=self.window)
        self.imageOrigLabel.setBackgroundRole(QPalette.Base)
        self.imageOrigLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageOrigLabel.setScaledContents(True)
        self.imageOrigLabel.move(270, 10)
        self.imageOrigLabel.resize(210, 210)

        self.imageModLabel = QLabel(parent=self.window)
        self.imageModLabel.setBackgroundRole(QPalette.Base)
        self.imageModLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageModLabel.setScaledContents(True)
        self.imageModLabel.move(270, 225)
        self.imageModLabel.resize(210, 210)

        self.groupbox = QGroupBox("Settings 1..10", parent=self.window)

        self.sliderList = []
        vbox = QVBoxLayout()
        self.groupbox.setLayout(vbox)
        self.groupbox.move(10, 35)
        self.groupbox.resize(250, 400)

        for i in range(0, 10):
            self.sliderList.append(SemanticClass(parent=self.window, index=i, func=self.modificate))
            self.sliderList[-1].slider.setEnabled(False)
            vbox.addWidget(self.sliderList[-1].label)
            vbox.addWidget(self.sliderList[-1].slider)

        self.window.show()
        sys.exit(self.app.exec_())

class SemanticClass:
    """A simple example class"""
    def __init__(self, parent, index, func):
        self.label = QLabel(parent=parent)
        self.slider = QSlider(Qt.Horizontal, parent=parent)
        self.slider.setRange( -50, 50 )
        self.slider.sliderReleased.connect(self.valRealesed)
        self.slider.valueChanged.connect(self.valChanged)
        self.setIndex(index)
        self.modFunc = func

    def setIndex(self, index):
        self.ind = index
        self.label.setText(str(f's{self.ind} = {self.slider.value()}'))

    def valChanged(self, value):
        self.label.setText(str(f's{self.ind} = {value}'))

    def valRealesed(self):
        self.modFunc(self.ind, self.slider.value())
        print('semantic ', self.ind, ' changed = ', self.slider.value())

def loadModel(
            results_dir='./generated',
            models_dir='./models',
            name='faces',
            load_from=150,
            image_size=128,
            network_capacity=16,
            fmap_max=512,
            gradient_accumulate_every=6,
            learning_rate=2e-4,
            lr_mlp=0.1,
            ttur_mult=1.5,
            rel_disc_loss=False,
            num_workers=None,
            save_every=1000,
            evaluate_every=1000,
            num_image_tiles=8,
            trunc_psi=0.75,
            mixed_prob=0.9,
            fp16=False,
            no_pl_reg=False,
            cl_reg=False,
            fq_layers=[],
            fq_dict_size=256,
            attn_layers=[],
            no_const=False,
            aug_prob=0.,
            aug_types=['translation', 'cutout'],
            top_k_training=False,
            generator_top_k_gamma=0.99,
            generator_top_k_frac=0.5,
            dual_contrast_loss=False,
            dataset_aug_prob=0.,
            calculate_fid_every=None,
            calculate_fid_num_images=12800,
            clear_fid_cache=False,
            log=False
    ):
        model_args = dict(
            name=name,
            results_dir=results_dir,
            models_dir=models_dir,
            gradient_accumulate_every=gradient_accumulate_every,
            image_size=image_size,
            network_capacity=network_capacity,
            fmap_max=fmap_max,
            lr=learning_rate,
            lr_mlp=lr_mlp,
            ttur_mult=ttur_mult,
            rel_disc_loss=rel_disc_loss,
            num_workers=num_workers,
            save_every=save_every,
            evaluate_every=evaluate_every,
            num_image_tiles=num_image_tiles,
            trunc_psi=trunc_psi,
            fp16=fp16,
            no_pl_reg=no_pl_reg,
            cl_reg=cl_reg,
            fq_layers=fq_layers,
            fq_dict_size=fq_dict_size,
            attn_layers=attn_layers,
            no_const=no_const,
            aug_prob=aug_prob,
            aug_types=cast_list(aug_types),
            top_k_training=top_k_training,
            generator_top_k_gamma=generator_top_k_gamma,
            generator_top_k_frac=generator_top_k_frac,
            dual_contrast_loss=dual_contrast_loss,
            dataset_aug_prob=dataset_aug_prob,
            calculate_fid_every=calculate_fid_every,
            calculate_fid_num_images=calculate_fid_num_images,
            clear_fid_cache=clear_fid_cache,
            mixed_prob=mixed_prob,
            log=log
        )

        model = Trainer(**model_args)
        model.load(load_from)
        return model

def main():
    model = loadModel()
    GUIWindow(model)

if __name__ == '__main__':
    main()
