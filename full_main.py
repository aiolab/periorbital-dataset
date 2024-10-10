from full_parameter import *
from full_trainer import Trainer
# from full_tester import Tester
from full_data_loader import Data_Loader_Split
from torch.backends import cudnn
from full_utils import make_folder
import torch
import wandb
from full_tester import Tester

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    print('starting run in main')

    cudnn.benchmark = True

    torch.cuda.empty_cache()
    set_seed(42)
    
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    if config.train:
        make_folder(config.model_save_path, config.version)
        
        data_loader = Data_Loader_Split(config.img_path, config.label_path, config.imsize,
                            config.batch_size, config.train)            

        if config.hp_tune:
            sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'Dice_score_class_1',
                    'goal': 'maximize'
                },
                'parameters': {
                    'learning_rate': {
                        'values': [ 1e-5, 1e-4, 1e-3, 1e-2]
                    },
                    'batch_size': {
                        'values': [8]
                    },
                    'beta1': {
                        'values': [.8, .99]
                    },
                    'beta2': {
                        'values': [ .8, 0.99]
                    }
                }
            }
            
            # dont pass dataloader because will dynamically load data based on batch size hp 
            sweep_id = wandb.sweep(sweep=sweep_config, project="DLV3_celeb")

            def train_wrapper():
                # wandb.init(project='unet_celeb', config=sweep_config)
                trainer = Trainer(None, config,True, device=device)
                trainer.train()
            wandb.agent(sweep_id, function=train_wrapper, count=50)

        else:
            trainer = Trainer(data_loader.loader(), config, device=device)
            trainer.train()
            
    if config.test:
        tester = Tester(config, device)
        tester.test()
        

if __name__ == '__main__':

    config = get_parameters()


    main(config)


