import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional

from DCLS.construct.modules import Dcls2_1d

from model import Model
from utils import set_seed

class CSNN1d_Delays(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config


    def build_model(self):

        self.blocks = []          

        ################################################   First Layer    #######################################################

        block = [
                    Dcls2_1d(in_channels = 1, out_channels = self.config.channels[0], kernel_count  = self.config.kernel_count,
                            stride = (self.config.strides[0], 1), dense_kernel_size = self.config.kernel_sizes[0], 
                            dilated_kernel_size = self.config.max_delay, bias = self.config.bias, version = self.config.DCLSversion),
                    
                    nn.BatchNorm2d(num_features = self.config.channels[0])
                    ]
        if self.config.spiking_neuron_type == 'lif': 
            block.append(neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
        elif self.config.spiking_neuron_type == 'plif': 
            block.append(neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))


        self.blocks.append(block)

        ################################################   Hidden Layers    #######################################################

        for i in range(1, self.config.n_layers):
            block = [
                        Dcls2_1d(in_channels = self.config.channels[i-1], out_channels = self.config.channels[i], kernel_count  = self.config.kernel_count,
                                stride = (self.config.strides[i], 1), dense_kernel_size = self.config.kernel_sizes[i], 
                                dilated_kernel_size = self.config.max_delay, bias = self.config.bias, version = self.config.DCLSversion),
                            
                        nn.BatchNorm2d(num_features = self.config.channels[i]) 
                    ]
            
            if self.config.spiking_neuron_type == 'lif': 
                block.append(neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            elif self.config.spiking_neuron_type == 'plif': 
                block.append(neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
                
            self.blocks.append(block)

        ################################################   Final Layer    #######################################################

        self.final_block = [
                            layer.Linear(in_features = self.config.channels[-1], out_features = self.config.n_outputs, bias = self.config.bias, step_mode='m')
                            ]
        
        if self.config.spiking_neuron_type == 'lif': 
            self.final_block.append(neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True))
        elif self.config.spiking_neuron_type == 'plif': 
            self.final_block.append(neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True))

        
        self.blocks.append(self.final_block)

        ################################################   Registering parameter groups   #########################################
        # Register parameter groups to have different learning rates and/or optimizer/scheduler fo each one, potentially.

        self.model = nn.Sequential(*[m for b in self.blocks for m in b])


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
            elif isinstance(m, layer.Linear):
                self.weights_fc.append(m.weight)
                if self.config.bias:
                    self.weights_fc.append(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)
            elif isinstance(m, neuron.ParametricLIFNode):
                self.weights_plif.append(m.w)


    def forward(self, x):
        # Neurons is same as Freqs

        x = x.permute(0,2,1)                    # permute from (batch, time, neurons) to  (batch, neurons, time) for dcls2-1d strides
        x = x.unsqueeze(1)                      # add channels dimension  (batch, channels, neurons, time)

        

        for i in range(self.config.n_layers):
            l = self.blocks[i]
            x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)          # add 0 padding following the time dimension
            x = l[0](x)                         # Apply the conv x size = (Batch, Channels, Neurons, Time)
            x = l[1](x)                         # Apply Batchnorm

            x = x.permute(3, 0, 1, 2)           # permute to (Time, Batch, *) for multi-step mode in SJ 
            x = l[2](x)                         # Apply spiking neuron
            x = x.permute(1, 2, 3, 0)           # permute back 

        # x size is (Batch, Channels, Neurons, Time)
        out = x.mean(dim=2)                     # GlobalAvgPooling on Neurons/Freqs

        out = out.permute(2, 0, 1)              # permute to (Time, Batch, Channels)
        out = self.blocks[-1][0](out)           # Apply final FC+LIF block
        out = self.blocks[-1][1](out)   

        if self.config.loss != 'spike_count':   
            out = self.blocks[-1][1].v_seq   # Return output neurons membrane potentials (Threshold should be infinite) if loss is not about spike counts      

        return out




    def init_parameters(self):
        set_seed(self.config.seed)


        ###################################   Weights Init   ######################################



        ###################################   Delay positions Init   ##############################
        if self.config.init_pos_method == 'uniform':
            for i in range(self.config.n_layers):
                torch.nn.init.uniform_(self.blocks[i][0].P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                self.blocks[i][0].clamp_parameters()


        ##################################   SIG Init   ############################

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


    def decrease_sig(self, epoch):
        # Decreasing to 0.23 instead of 0.5
                                                          
        sig = self.blocks[0][0].SIG[0,0,0,0].detach().cpu().item()                      # Get current SIG value
        if self.config.decrease_sig_method == 'exp':
            if epoch < self.config.final_epoch and sig > 0.23:
                alpha = 0 
                if self.config.DCLSversion == 'gauss':
                    alpha = (0.23/self.config.sigInit)**(1/(self.config.final_epoch))

                for i in range(self.config.n_layers):
                    self.blocks[i][0].SIG *= alpha



    def optimizers(self):
        # weight_decay for positions and for batchnorm should be 0
        opts = []
        if self.config.optimizer_w == 'adam':
            opts.append(optim.Adam([{'params':self.weights_conv, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_fc, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.delay_positions, 'lr':self.config.lr_pos, 'weight_decay': 0},
                                    {'params':self.weights_plif, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_bn, 'lr':self.config.lr_w, 'weight_decay':0}]))

        return opts