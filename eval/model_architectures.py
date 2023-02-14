import os
import importlib
import torch
from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a model and all meta information
Model = namedtuple( 'Model' , [

    'name'        , # The identifier of this model architecture variant, e.g. 'erfnet', ...
                    # We use them to uniquely name a model of a specific architecture.

    'id'          , # An integer ID that is associated with each model.

    'ModuleName',   # Name of file/module that includes 'cooking recipe' to create new instance
                    # of the model architecture as a pytorch nn.Module.

    'instantClass', # The name of the class that needs to be called for instantiation.

    'modelState'  , # The name of the file which includes parameters (weights etc.) of
                    # a previous state (e.g. after training) which is needed to load
                    # a default checkpoint.
    ] )

#--------------------------------------------------------------------------------
# A list of all models
#--------------------------------------------------------------------------------

models = [
    #       name                        id   ModuleName                                         instanceClass       modelState(not considered right now)
    Model(  'sample',                   0,  'ModuleName',                                       'ClassName',        'none'),
    Model(  'SwiftNet',                 1,  'models.swiftnet.fw_adaptor',           'SwiftNet',         'none'),
    Model(  'SwiftNetRec',              2,  'models.swiftnet_rec.fw_adaptor',       'SwiftNetRec',      'none'),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to modul object
name2model      = { model.name    : model for model in models           }
# id to model object
id2model        = { model.id      : model for model in models           }

# --------------------------------------------------------------------------------
# Offer generic 'load model definition' function
# --------------------------------------------------------------------------------
def load_model_def(model_name, num_classes_wo_bg, ignoreBG=True, **kwargs):
    # load the module which includes model def; will raise ImportError if module cannot be loaded
    module = importlib.import_module(name2model[model_name].ModuleName)
    # get the class; will raise AttributeError if class cannot be found
    class_ = getattr(module, name2model[model_name].instantClass)
    # Call an instance of the class to build the model
    if ignoreBG:
        model = class_(num_classes_wo_bg, **kwargs)
    else:
        model = class_(num_classes_wo_bg + 1)
    print("Model definition based on", model_name, "was loaded.")
    return model

# --------------------------------------------------------------------------------
# Call custom 'load model state' function
# --------------------------------------------------------------------------------
def load_model_state(model, model_name, model_base_path, epoch):
    """Load model state checkpoint from disk"""
    model_state_path = os.path.join(model_base_path, 'models', 'weights_{}'.format(epoch))
    assert os.path.isdir(model_state_path), \
        "Cannot find folder {}".format(model_state_path)
    print("Loading 'model.pth' state checkpoint from folder {}".format(model_state_path))
    path = os.path.join(model_state_path, "{}.pth".format('model'))
    pretrained_dict = torch.load(path)

    # load the module, will raise ImportError if module cannot be loaded
    module = importlib.import_module(name2model[model_name].ModuleName)
    # get the class, will raise AttributeError if class cannot be found
    function_ = getattr(module, 'load_state_dict_into_model')
    # Call custom function to load pretrained state dict for the specific model architecture
    function_(model, pretrained_dict)
    return model
