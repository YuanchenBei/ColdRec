from .AGNN import AGNN
from .ALDI import ALDI
from .AMR import AMR
from .CCFCRec import CCFCRec
from .CGRC import CGRC
from .CLCRec import CLCRec
from .DeepMusic import DeepMusic
from .DropoutNet import DropoutNet
from .FSGNN import FSGNN
from .DUIF import DUIF
from .GAR import GAR
from .GoRec import GoRec
from .Heater import Heater
from .KNN import KNN
from .LARA import LARA
from .LightGCN import LightGCN
from .MetaEmbedding import MetaEmbedding
from .MF import MF
from .M2VAE import M2VAE
from .MTPR import MTPR
from .NCL import NCL
from .NGCF import NGCF
from .SimGCL import SimGCL
from .USIM import USIM
from .VBPR import VBPR
from .XSimGCL import XSimGCL

AVAILABLE_MODELS = {
    'AGNN': AGNN,
    'ALDI': ALDI,
    'AMR': AMR,
    'CCFCRec': CCFCRec,
    'CGRC': CGRC,
    'CLCRec': CLCRec,
    'DeepMusic': DeepMusic,
    'DropoutNet': DropoutNet,
    'FSGNN': FSGNN,
    'DUIF': DUIF,
    'GAR': GAR,
    'GoRec': GoRec,
    'Heater': Heater,
    'KNN': KNN,
    'LARA': LARA,
    'LightGCN': LightGCN,
    'MetaEmbedding': MetaEmbedding,
    'MF': MF,
    'M2VAE': M2VAE,
    'MTPR': MTPR,
    'NCL': NCL,
    'NGCF': NGCF,
    'SimGCL': SimGCL,
    'USIM': USIM,
    'VBPR': VBPR,
    'XSimGCL': XSimGCL,
}

__all__ = list(AVAILABLE_MODELS.keys()) + ['AVAILABLE_MODELS']
