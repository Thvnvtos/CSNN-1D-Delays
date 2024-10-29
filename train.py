import torch

import data, utils
from csnnext_delays import CSnnNext_delays
from config import Config


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"\n=====> Device = {device} \n\n")

config = Config()


if config.model_type == 'csnnext-delays':
    model = CSnnNext_delays(config).to(device)



print(f"===> Dataset    = {config.dataset}")
print(f"===> Model type = {config.model_type}")
print(f"===> Model size = {utils.count_parameters(model)}\n\n")


if config.dataset == 'shd':
    train_loader, valid_loader = data.SHD_dataloaders(config)
    test_loader = None
elif config.dataset == 'ssc':
    train_loader, valid_loader, test_loader = data.SSC_dataloaders(config)
elif config.dataset == 'gsc':
    train_loader, valid_loader, test_loader = data.GSC_dataloaders(config)
else:
    raise Exception(f'dataset {config.dataset} not implemented')


model.train_model(train_loader, valid_loader, test_loader, device)