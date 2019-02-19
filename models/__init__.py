from enum import Enum
#from EFA_CoolPool import Model as EFA_CoolPool_Model
from SPH_EFA_MaxPool import Model as SPH_EFA_MaxPool_Model

MODELS = Enum("MODELS", "EFA_CoolPool SPH_EFA_MaxPool")


def get_model(model_name):
    if type(model_name) == str:
        try:
            model_name = MODELS[model_name]
        except KeyError:
            raise Exception("Unknown model ! Check the name again")

    if model_name is MODELS.EFA_CoolPool:
        return EFA_CoolPool_Model
    if model_name is MODELS.SPH_EFA_MaxPool:
        return SPH_EFA_MaxPool_Model
    # elif model_name is DATASETS.ModelNet10OFF:
    #     return ModelNet10OFF, MN10_CLASS_DICT
    else:
        raise Exception("Unknown model ! Check the name again")
