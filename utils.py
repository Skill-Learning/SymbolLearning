def init_weights_and_freeze(model_tgt, model_src, freeze_layers = []):
    '''
        copy weights to existing layers in model_tgt from model_src
    '''
    for w_tgt, w_src in zip(model_tgt.named_parameters(), model_src.named_parameters()):
        if 'layer' in w_tgt[0]:
            w_tgt[1].data = w_src[1].data
            w_tgt[1].requires_grad = True

        # check for layer freeze 
        for unfreeze_layer in freeze_layers:
            if unfreeze_layer in w_tgt[0]:
                w_tgt[1].requires_grad = False