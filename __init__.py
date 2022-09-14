import torch
from torch import nn
import torch.nn.functional as F

import math
from typing import List, Tuple, Dict, OrderedDict, Optional, Union

from einops import *
from einops.layers.torch import *

##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("You are on " + str(torch.cuda.get_device_name(device)))
else:
    print("You are on " + str(device).upper())

##