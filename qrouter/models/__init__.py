__all__ = ["QRouterModel", "build_model"]


def __getattr__(name):
    if name == "build_model":
        from qrouter.models.materialize import build_model

        return build_model
    if name == "QRouterModel":
        from qrouter.models.vlms.qrouter import QRouterModel

        return QRouterModel
    raise AttributeError(name)
