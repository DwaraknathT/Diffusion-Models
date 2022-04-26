_SDES = dict()


def register(fn):
    global _SDES
    _SDES[fn.__name__] = fn
    return fn


def get_sde(args=None, device=None):
    if args.sde is None:
        return _SDES
    return _SDES[args.sde](args, device)
