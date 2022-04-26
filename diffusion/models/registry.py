_MODELS = dict()


def register(fn):
    global _MODELS
    _MODELS[fn.__name__] = fn
    return fn


def get_model(args=None, device=None):
    if args.model is None:
        return _MODELS
    return _MODELS[args.model](args).to(device)
