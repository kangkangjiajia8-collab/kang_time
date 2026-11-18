from autoapex.tabular.models.tabm.tabm_model import TabMModel
from autoapex.tabular.testing import FitHelper

toy_model_params = {}


def test_tabm():
    model_cls = TabMModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
