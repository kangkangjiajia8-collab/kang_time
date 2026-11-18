from autoapex.tabular.models.tabicl.tabicl_model import TabICLModel
from autoapex.tabular.testing import FitHelper

toy_model_params = {}


def test_tabicl():
    model_cls = TabICLModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
