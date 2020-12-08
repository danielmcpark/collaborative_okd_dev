from __future__ import absolute_import
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import sys
import numpy as np
import time
import logging

from collections import OrderedDict
import json
import subprocess
import sys
import time
import xml.etree.ElementTree

__all__ = ['Logger', 'LoggerMonitor', 'savefig', 'SpeedoMeter', 'GPUMemHooker']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)

def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass

        self.numbers = {}
        seslf.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers

        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend(['(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    def __init__(self, paths):
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
            loc = 'best'
        plt.legend(legend_text, loc='upper right', ncol=2)
        plt.grid(True)

class SpeedoMeter(object):
    """Logs training speed and evaluation metrics periodically.

    Parameters
    ---------
    batch_size: int
        Batch size of data.
    frequent: int
        Specifies how frequently training speed and evaluation metrics
        must be logged. Default behavior is to log once every 50 batches.
    auto_reset: bool
        Reset the evaluation metrics after each log.
    """
    def __init__(self, batch_size, batch_length, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.first_tick = True
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset
        self.speed = 0
        self.batch_length = batch_length

    def __call__(self, batch_idx, epoch, init_tick):
        """Callback to Show speed."""
        if self.first_tick:
            self.tic = init_tick
            self.first_tick = False
        count = batch_idx # # of batch
        if self.last_count < count:
            self.init = True
        self.last_count = count

        if self.init:
            #if count % self.frequent == 0: # if you hope to display by regular samples
            if count == self.batch_length:
                try:
                    self.speed = (self.frequent * self.batch_size) / (time.time() - self.tic)
                except ZeroDivisionError:
                    self.speed = float('inf')
                #logging.info("Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                #             epoch, count, speed)
                print("\nEpoch[%d] Batch [%d]\tSpeed: %.2f samples/sec"
                        %(epoch, count, self.speed))
                self.tic = time.time()
            else:
                pass
        else:
            self.init = False
            self.tic = time.time()
        return self.speed

class GPUMemHooker(object):
    def __init__(self, batch_length, frequent, n_gpu):
        self.d = OrderedDict()
        self.frequent = frequent
        self.n_gpu = n_gpu
        self.idx_gpu = 0
        self.batch_length = batch_length

        if self.n_gpu == 'cuda:0' or self.n_gpu == '0':
            self.idx_gpu = 0
        elif self.n_gpu == 'cuda:1' or self.n_gpu == '1':
            self.idx_gpu = 1
        elif self.n_gpu == 'cuda:2' or self.n_gpu == '2':
            self.idx_gpu = 2
        elif self.n_gpu == 'cuda:3' or self.n_gpu == '3':
            self.idx_gpu = 3
        else:
            raise ValueError('Define obviously gpu number..')

    def __call__(self, batch_idx):
        count = batch_idx
        self.d["time"] = time.time()
        self.d["mem_used"] = 0

        #if count % self.frequent == 0: # if you hope to display by regular samples
        if count == self.batch_length:
            cmd = ['nvidia-smi', '-q', '-x']
            cmd_out = subprocess.check_output(cmd)
            gpu = xml.etree.ElementTree.fromstring(cmd_out).findall("gpu")

            util = gpu[self.idx_gpu].find("utilization")
            self.d["gpu_util"] = self.extract(util, "gpu_util", "%")

            self.d["mem_used"] = self.extract(gpu[self.idx_gpu].find("fb_memory_usage"), "used", "MiB")
            self.d["mem_used_per"] = self.d["mem_used"] * 100 / 11178

            if self.d["gpu_util"] < 13 and self.d["mem_used"] < 2816:
                msg = 'GPU status: Idle \n'
            else:
                msg = 'GPU status: Busy \n'

            now = time.strftime("%c")
            print('\n\nUpdated at %s\n\nGPU utilization: %s %%\nRAM used: %s %%\nRAM allocation: %s MiB\n\n%s\n\n' %(now, self.d["gpu_util"], self.d["mem_used_per"], self.d["mem_used"], msg))
        else:
            pass
        return self.d["mem_used"]

    def extract(self, elem, tag, drop_s):
        text = elem.find(tag).text
        if drop_s not in text: raise Exception(text)
        text = text.replace(drop_s, "")
        try:
            return int(text)
        except ValueError:
            return float(next)

