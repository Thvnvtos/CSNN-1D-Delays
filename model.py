import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_seed


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.build_model()
        self.init_parameters()
    


    def train_model(self, train_loader, valid_loader, test_loader, device):
        ######################################################################################
        #                                                                                    #
        #                               Main Training Loop for all models                    #
        #                                                                                    #
        ##################################    Initializations    #############################
    

        set_seed(self.config.seed)

        # returns a list of optimizers for different groups of parameters
        optimizers = self.optimizers()
 
        ##################################    Train Loop    ##################################

        for epoch in range(self.config.epochs):
            self.train()

            # _ is the length of unpadded x
            for i, (x, labels, _ ) in enumerate(train_loader):

                # x for shd is: (batch_size, time_steps, neurons)
                labels = F.one_hot(labels, self.config.n_outputs).float()

                x = x.to(device)
                labels = labels.to(device)

                for opt in optimizers:  opt.zero_grad()

                output = self.forward(x)
                loss = self.calc_loss(output, labels)

                loss.backward()
                for opt in optimizers: opt.step()

                self.reset_model()
        
            ##################################    Eval Loop    ##################################

            loss_valid, metric_valid = self.eval_model(valid_loader, device)


            ##########################     Logging and Plotting  ################################

            print(f'====> Epoch {epoch} : ', loss.detach().cpu().item())

                

    def eval_model(self, loader, device):
        
        self.eval()
        with torch.no_grad():
            # _ is the length of unpadded x
            for i, (x, labels, _ ) in enumerate(loader):

                # x for shd is: (batch_size, time_steps, neurons)
                labels = F.one_hot(labels, self.config.n_outputs).float()

                x = x.to(device)
                labels = labels.to(device)

                output = self.forward(x)
                loss = self.calc_loss(output, labels)

                self.reset_model()

        
        return loss.detach().cpu().item(), 0
    

    def calc_loss(self, output, y):

        if self.config.loss == 'mean': m = torch.mean(output, 0)
        elif self.config.loss == 'max': m, _ = torch.max(output, 0)
        elif self.config.loss == 'spike_count': m = torch.sum(output, 0)
        elif self.config.loss == 'sum':
            softmax_fn = nn.Softmax(dim=2)
            m = torch.sum(softmax_fn(output), 0)

        
        if self.config.loss_fn == 'CEloss':
            CEloss = nn.CrossEntropyLoss()
            loss = CEloss(m, y)

        
        return loss