_TRAINERS = dict()


def register(fn):
    global _TRAINERS
    _TRAINERS[fn.__name__] = fn
    return fn


def get_trainer(args=None):
    if args.trainer is None:
        return _TRAINERS
    return _TRAINERS[args.trainer](args)
