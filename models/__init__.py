from .ShowAttTellModel import ShowAttTellModel


def setup(opt):
    if opt.model_name == 'vatt':
        model = ShowAttTellModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
