import yaml 
import os 


def read_config(config_path):

    '''
    arg : 
    config_path : path to the config file
    
    return : 
    config : config file in the form of dictionary    
    
    
    '''
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config 


def create_dir(path): 

    """
    arg : 
    path : path to the directory 

    return : 
    path : path to the directory
    
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path 



def lr_sheduler(optimizer, epoch, lr, decay_rate, decay_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer