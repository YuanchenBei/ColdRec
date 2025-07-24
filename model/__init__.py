from .ALDI import ALDI
from .AMR import AMR
from .CCFCRec import CCFCRec
from .CLCRec import CLCRec
from .DeepMusic import DeepMusic
from .DropoutNet import DropoutNet
from .DUIF import DUIF
from .GAR import GAR
from .GoRec import GoRec
from .Heater import Heater
from .KNN import KNN
from .LARA import LARA
from .LightGCN import LightGCN
from .MetaEmbedding import MetaEmbedding
from .MF import MF
from .MTPR import MTPR
from .NCL import NCL
from .NGCF import NGCF
from .SimGCL import SimGCL
from .USIM import USIM
from .VBPR import VBPR
from .XSimGCL import XSimGCL

AVAILABLE_MODELS = {
    'ALDI': ALDI,
    'AMR': AMR,
    'CCFCRec': CCFCRec,
    'CLCRec': CLCRec,
    'DeepMusic': DeepMusic,
    'DropoutNet': DropoutNet,
    'DUIF': DUIF,
    'GAR': GAR,
    'GoRec': GoRec,
    'Heater': Heater,
    'KNN': KNN,
    'LARA': LARA,
    'LightGCN': LightGCN,
    'MetaEmbedding': MetaEmbedding,
    'MF': MF,
    'MTPR': MTPR,
    'NCL': NCL,
    'NGCF': NGCF,
    'SimGCL': SimGCL,
    'USIM': USIM,
    'VBPR': VBPR,
    'XSimGCL': XSimGCL,
}

__all__ = list(AVAILABLE_MODELS.keys()) + ['AVAILABLE_MODELS']
