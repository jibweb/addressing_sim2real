from enum import Enum
#from EFA_CoolPool import Model as EFA_CoolPool_Model
from SPH_EFA_MaxPool import Model as SPH_EFA_MaxPool_Model
from SPH_TConv_MaxPool import Model as SPH_TConv_MaxPool_Model
from COORDSSET_TConv_MaxPool import Model as COORDSSET_TConv_MaxPool_Model
from COORDSSET_GAT_MaxPool import Model as COORDSSET_GAT_MaxPool_Model
from COORDSSET_GCN_MaxPool import Model as COORDSSET_GCN_MaxPool_Model
from COORDSSET_GAT_SingleNodePred import Model as COORDSSET_GAT_SingleNodePred_Model
from COORDSSET_GAT_Disconnect_SingleNodePred import Model as COORDSSET_GAT_Disconnect_SingleNodePred_Model


MODELS = Enum("MODELS", "EFA_CoolPool SPH_EFA_MaxPool SPH_TConv_MaxPool COORDSSET_TConv_MaxPool COORDSSET_GAT_MaxPool COORDSSET_GCN_MaxPool COORDSSET_GAT_SingleNodePred COORDSSET_GAT_Disconnect_SingleNodePred")


def get_model(model_name):
    if type(model_name) == str:
        try:
            model_name = MODELS[model_name]
        except KeyError:
            raise Exception("Unknown model ! Check the name again")

    # if model_name is MODELS.EFA_CoolPool:
    #     return EFA_CoolPool_Model
    if model_name is MODELS.SPH_EFA_MaxPool:
        return SPH_EFA_MaxPool_Model
    if model_name is MODELS.SPH_TConv_MaxPool:
        return SPH_TConv_MaxPool_Model
    if model_name is MODELS.COORDSSET_TConv_MaxPool:
        return COORDSSET_TConv_MaxPool_Model
    if model_name is MODELS.COORDSSET_GAT_MaxPool:
        return COORDSSET_GAT_MaxPool_Model
    if model_name is MODELS.COORDSSET_GCN_MaxPool:
        return COORDSSET_GCN_MaxPool_Model
    if model_name is MODELS.COORDSSET_GAT_SingleNodePred:
        return COORDSSET_GAT_SingleNodePred_Model
    if model_name is MODELS.COORDSSET_GAT_Disconnect_SingleNodePred:
        return COORDSSET_GAT_Disconnect_SingleNodePred_Model
    # elif model_name is DATASETS.ModelNet10OFF:
    #     return ModelNet10OFF, MN10_CLASS_DICT
    else:
        raise Exception("Unknown model ! Check the name again")
