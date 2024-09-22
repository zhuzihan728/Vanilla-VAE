import matplotlib.pyplot as plt
import torch
import json
import yaml

def show_image_grid(images, labels = None, n_img=8):
    fig = plt.figure(figsize=(8, n_img/10))
    gs = plt.GridSpec(int(n_img/10)+1, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, image in enumerate(images[:n_img]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if labels is not None:
            ax.set_title(str(labels[i]), fontsize=8)
        plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=0.8)
    # plt.show()
    return fig
    
    
def to_json(path_params, path_json):
    params = torch.load(path_params, map_location=torch.device('cpu'))
    raw_state_dict = {}
    for k, v in params.items():
        val = v.flatten().numpy().tolist()
        raw_state_dict[k] = val

    with open(path_json, 'w') as outfile:
        json.dump(raw_state_dict, outfile,indent= "\t")
        
class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
        
def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    return args
        
def get_network_paras_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info[model_name] = trainable_params
    return info