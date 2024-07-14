import torch.nn as nn
import torch.optim as optim

from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional


from model import Model
from utils import set_seed

class CSNN1d(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config


    def build_model(self):

        self.blocks = []          

        ################################################   First Layer    #######################################################

        block = [
                            layer.Conv1d(in_channels = 1 , out_channels = self.config.channels[0], kernel_size = self.config.kernel_sizes[0],
                                         stride = self.config.strides[0], bias = self.config.bias, step_mode='m'),
                            layer.BatchNorm1d(num_features = self.config.channels[0], step_mode='m')
                        
                        ]
        if self.config.spiking_neuron_type == 'lif': 
            block.append(neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
        elif self.config.spiking_neuron_type == 'plif': 
            block.append(neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))


        self.blocks.append(nn.Sequential(*block))

        ################################################   Hidden Layers    #######################################################

        for i in range(1, self.config.n_layers):
            block = [
                            layer.Conv1d(in_channels = self.config.channels[i-1] , out_channels = self.config.channels[i], kernel_size = self.config.kernel_sizes[i],
                                         stride = self.config.strides[i], bias = self.config.bias, step_mode='m'),
                            layer.BatchNorm1d(num_features = self.config.channels[i], step_mode='m')
                            ]
            
            if self.config.spiking_neuron_type == 'lif': 
                block.append(neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            elif self.config.spiking_neuron_type == 'plif': 
                block.append(neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
                
            self.blocks.append(nn.Sequential(*block))

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

        self.final_block = nn.Sequential(*self.final_block)
        self.blocks.append(self.final_block)

        ################################################   Registering parameter groups   #########################################
        # Register parameter groups to have different learning rates and/or optimizer/scheduler fo each one, potentially.

        self.model = nn.Sequential(*self.blocks)

        self.weights_conv = []
        self.weights_fc = []
        self.weights_bn = []
        self.weights_plif = []
        for m in self.model.modules():
            if isinstance(m, layer.Conv1d):
                self.weights_conv.append(m.weight)
                if self.config.bias:
                    self.weights_conv.append(m.bias)
            if isinstance(m, layer.Linear):
                self.weights_fc.append(m.weight)
                if self.config.bias:
                    self.weights_fc.append(m.bias)
            elif isinstance(m, layer.BatchNorm1d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)
            elif isinstance(m, neuron.ParametricLIFNode):
                self.weights_plif.append(m.w)


    def forward(self, x):
        # Neurons is same as Freqs

        x = x.permute(1,0,2)                    # permute from (batch, time, neurons) to  (time, batch, neurons) for multi-step processing
        x = x.unsqueeze(2)                      # add channels dimension  (time, batch, channels, neurons)

        for i in range(self.config.n_layers):
            x = self.blocks[i](x)

        # x size is (time, Batch, Channels, Neurons)
        out = x.mean(dim=3)                     # GlobalAvgPooling on Neurons/Freqs

        out = self.final_block(out)             # Apply final FC+LIF block

        if self.config.loss != 'spike_count':   
            out = self.final_block[-1].v_seq    # Return output neurons membrane potentials (Threshold should be infinite) if loss is not about spike counts      

        return out



    def init_parameters(self):
        # Should initialize with kaiming uniform ?
        pass


    def reset_model(self):
        # you can add sparsity mask in here
        functional.reset_net(self)



    def optimizers(self):
        opts = []
        if self.config.optimizer_w == 'adam':
            opts.append(optim.Adam([{'params':self.weights_conv, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_fc, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_plif, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_bn, 'lr':self.config.lr_w, 'weight_decay':0}]))

        return opts