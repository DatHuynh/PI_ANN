__author__ = 'DatHuynh'

import DataSetGenerator as dg
import WGA
import Network as nw
import PGD
import FileHelper as fh
import PGA
import numpy as np


sizes = [10,8,140]
trainingdata = fh.loadTrainingReal('PsBeta','UsBeta',sizes[0],sizes[2],9,215999,0.1,31)
eta = 0.03
thresholdTerminate = 0.1

net = nw.Network(sizes,threshold= thresholdTerminate,alpha=0.1)

net.GD(trainingdata,None,epoch=100000,eta=eta)