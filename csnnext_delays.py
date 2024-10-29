import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

import numpy as np
import matplotlib.pyplot as plt

from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional

from DCLS.construct.modules import Dcls2_1d

from model import Model
from utils import set_seed




# To include the permutation inside the Sequential block
class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)






class CSNNext_delays(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config


    def build_model(self):

        self.blocks = []          

        ################################################   Stem    #######################################################
        self.stem = [
            
            layer.Conv1d(in_channels = 1 , out_channels = self.config.channels[0], kernel_size = self.config.stem_kernel_size,
                         stride = self.config.stem_stride, bias = self.config.bias, step_mode='m')
        ]


        if self.config.batchnorm_type == 'SJ_bn1d':
            self.stem.append(layer.BatchNorm1d(num_features = self.config.channels[0], step_mode='m'))
        

        ################################################   Hidden Layers - Stages    #######################################################

        for i in range(self.config.n_stages):
            for N in self.config.n_blocks:
            
                block = [
                    
                    Permute(1, 2, 3, 0),
                    #(batch, channels, neurons, time)

                    Dcls2_1d(in_channels = self.config.channels[i], out_channels = self.config.channels[i], kernel_count = self.config.kernel_count,
                             stride = (self.config.strides[i], 1), dense_kernel_size = self.config.kernel_sizes[i], 
                             dilated_kernel_size = self.config.max_delays[i], bias = self.config.bias, version = self.config.DCLSversion,
                             groups = self.config.channels[i]),

                    nn.Conv2d(in_channels=self.config.channels[i], out_channels=self.config.channels[i], kernel_size=(1,1), stride=1),

                    Permute(3, 0, 1, 2)
                    #(time, batch, channels, neurons)
                ]


                # Add BN and Neuron model according to config
                
                if self.config.batchnorm_type == 'SJ_bn1d':
                    block.append(layer.BatchNorm1d(num_features = self.config.channels[i], step_mode='m'))
                

                if self.config.spiking_neuron_type == 'lif': 
                    block.append(neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                        surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                        step_mode='m', decay_input=False, store_v_seq = True))
                elif self.config.spiking_neuron_type == 'plif': 
                    block.append(neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                        surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                        step_mode='m', decay_input=False, store_v_seq = True))
                
                self.blocks += block
            

            # Downsampling block

            if i < self.config.n_stages-1:          # No downsampling after Final block
                downsampling_block = [
                
                layer.Conv1d(in_channels = self.config.channels[i] , out_channels = self.config.channels[i+1], kernel_size = self.config.downsampling_kernel_sizes[i],
                            stride = self.config.downsampling_strides[i], bias = self.config.bias, step_mode='m')
                ]

                if self.config.batchnorm_type == 'SJ_bn1d':
                    downsampling_block.append(layer.BatchNorm1d(num_features = self.config.channels[i+1], step_mode='m'))


                self.blocks += downsampling_block

        ################################################   Final FC Layer    #######################################################

        self.final_FC = [
            
            layer.Dropout(p = self.config.dropout_p, step_mode='m'),
            layer.Linear(in_features = self.config.channels[-1], out_features = self.config.n_outputs, bias = self.config.bias, step_mode='m')
        ]
        

        if self.config.spiking_neuron_type == 'lif': 
            self.final_FC.append(neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True))
        elif self.config.spiking_neuron_type == 'plif': 
            self.final_FC.append(neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True))


        ################################################   Registering parameter groups   #########################################
        # Register parameter groups to have different learning rates and/or optimizer/scheduler fo each one, potentially.

        self.stem_seq = nn.Sequential(*self.stem)
        self.blocks_seq = nn.Sequential(*self.blocks)
        self.final_FC_seq = nn.Sequential(*self.final_FC)

        self.model = nn.Sequential(*(self.stem + self.blocks + self.final_FC))



        self.weights_conv = []
        self.delay_positions = []
        self.weights_fc = []

        self.weights_bn = []
        self.weights_plif = []
        
        for m in self.model.modules():
            if isinstance(m, Dcls2_1d):
                self.delay_positions.append(m.P)
                self.weights_conv.append(m.weight)
                if self.config.bias:
                    self.weights_conv.append(m.bias)
            elif isinstance(m, layer.Conv1d) or isinstance(m, nn.Conv2d):
                self.weights_conv.append(m.weight)
                if self.config.bias:
                    self.weights_conv.append(m.bias)

            elif isinstance(m, layer.Linear):
                self.weights_fc.append(m.weight)
                if self.config.bias:
                    self.weights_fc.append(m.bias)

            elif isinstance(m, layer.BatchNorm1d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)

            elif isinstance(m, neuron.ParametricLIFNode):
                self.weights_plif.append(m.w)





    def forward(self, x):
        
        # Input x = (batch, time, freqs)
        x = x.permute(1,0,2)                    # permute from (batch, time, neurons) to  (time, batch, neurons) for multi-step processing
        x = x.unsqueeze(2)                      # add channels dimension  (time, batch, channels, neurons)

        # Input x = (time, batch, channels, neurons)

        x = self.stem_seq(x)
        x = self.blocks_seq(x)

        # x size is (Time, Batch, Channels, Neurons)
        x = x.mean(dim=3)                     # GlobalAvgPooling on Neurons/Freqs

        out = self.final_FC_seq(x)

        if self.config.loss != 'spike_count':   
            out = self.blocks[-1][2].v_seq   # Return output neurons membrane potentials (Threshold should be infinite) if loss is not about spike counts      

        return out






    def init_parameters(self):
        set_seed(self.config.seed)

        ###################################   Weights Init   ######################################

        # TO - DO


        ###################################   Delay positions Init   ##############################
        if self.config.init_pos_method == 'uniform':
            for i in range(self.config.n_layers):
                torch.nn.init.uniform_(self.blocks[i][0].P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                self.blocks[i][0].clamp_parameters()


        ##################################   SIG Init   ############################

        if self.config.DCLSversion in ['gauss', 'max']:
            for i in range(self.config.n_layers):
                torch.nn.init.constant_(self.blocks[i][0].SIG, self.config.sigInit)
                self.blocks[i][0].SIG.requires_grad = False






    def reset_model(self, train=True):
        # Reset Spiking neurons dynamics
        functional.reset_net(self)

        #Clamp parameters of DCLS2-1D modules
        if train:
            for i in range(self.config.n_layers):
                self.blocks[i][0].clamp_parameters()





    def get_sigma(self):
        if self.config.DCLSversion in ['gauss', 'max']:
            return self.blocks[0][0].SIG[0,0,0,0].detach().cpu().item()
        else: return 0    





    def decrease_sig(self, epoch):
        
        with torch.no_grad():
            if self.config.DCLSversion in ['gauss', 'max']:
                
                if self.config.decrease_sig_method == 'exp':
                    if epoch < self.config.final_epoch:
                        for i in range(self.config.n_layers):
                            self.blocks[i][0].SIG *= self.config.alpha
                    
                    elif epoch == self.config.final_epoch:
                        sig = self.get_sigma()                                                        
                        alpha_final = 0 if self.config.DCLSversion == 'max' else self.config.sig_final_gauss/sig                            # Make sig 0 or final_gauss_sig which is 0.23
                        for i in range(self.config.n_layers):
                            self.blocks[i][0].SIG *= alpha_final






    def optimizers(self):
        # weight_decay for positions and for batchnorm should be 0
        opts = []
        if self.config.optimizer_w == 'adam':
            opts.append(optim.Adam([{'params':self.weights_conv, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_fc, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_plif, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_bn, 'lr':self.config.lr_w, 'weight_decay':0}]))
            
            opts.append(optim.Adam([{'params':self.delay_positions, 'lr':self.config.lr_pos, 'weight_decay': 0}]))

        return opts
    





    def schedulers(self, optimizers):
        #  returns a list of schedulers
        #  Fro now using one cycle for weights and cosine annealing for delay positions

        schedulers = []

        schedulers.append(optim.lr_scheduler.OneCycleLR(optimizers[0], max_lr=self.config.max_lr_w, total_steps=self.config.epochs))
        schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizers[1], T_max = self.config.t_max_pos))

        return schedulers





    def round_pos(self):
        with torch.no_grad():
            for i in range(self.config.n_layers):
                self.blocks[i][0].P.round_()
                self.blocks[i][0].clamp_parameters()



    def delay_eval_mode(self, temp_id):

        torch.save(self.state_dict(), temp_id + '.pt')                                      # Save state of model

        if self.config.DCLSversion in ['gauss', 'max']: 
            for i in range(self.config.n_layers):                                           # Change each DCLS conv to discrete vmax and round positions
                self.blocks[i][0].version = 'max'
                self.blocks[i][0].DCK.version = 'max'
                self.blocks[i][0].SIG *= 0
        
        self.round_pos()






    def delay_train_mode(self, temp_id):                                                    
        if self.config.DCLSversion == 'gauss':                                              
            for i in range(self.config.n_layers):
                self.blocks[i][0].version = 'gauss'
                self.blocks[i][0].DCK.version = 'gauss'

        self.load_state_dict(torch.load(temp_id + '.pt'), strict=True)
        if os.path.exists(temp_id + '.pt'):
            os.remove(temp_id + '.pt')
        else:
            print(f"File '{temp_id + '.pt'}' does not exist.")






    def save_pos_distribution(self, path):
        with torch.no_grad():
            #dcls.P size is (1, n_C_out, n_C_in, kernel_size, 1)
            for i in range(self.config.n_layers):
                
                pos_tensor = self.blocks[i][0].P
                fig, axes = plt.subplots(self.config.kernel_sizes[i], 1, figsize = (10, self.config.kernel_sizes[i]*3))

                bin_edges = np.linspace(-self.config.max_delay//2 + 1, self.config.max_delay//2, 50)

                for j in range(self.config.kernel_sizes[i]):                    
                    axes[j].hist(pos_tensor[:, :, :, j].flatten().cpu().detach().numpy(), bins =  bin_edges, color='lightgreen', edgecolor='black')
                    axes[j].set_title(f'Kernel row {j}')
                    axes[j].set_ylabel('Frequency')
                    axes[j].set_xlim(-self.config.max_delay//2, self.config.max_delay//2 + 1)
                axes[self.config.kernel_sizes[i]-1].set_xlabel('Position')
            
                plt.savefig(f'Layer_{i}.jpg')
                #plt.clf()