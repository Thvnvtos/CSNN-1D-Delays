import torch.nn as nn
import torch.optim as optim

from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional


from model import Model
from utils import set_seed

class SNN(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

    
    def build_model(self):
        
        ################################################   First Layer    #######################################################

        self.blocks = [[[layer.Linear(self.config.n_inputs, self.config.n_hidden_neurons, bias = self.config.bias, step_mode='m')],
                        [layer.Dropout(self.config.dropout_p, step_mode='m')]]]
        
        if self.config.use_batchnorm: self.blocks[0][0].insert(1, layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m'))
        if self.config.spiking_neuron_type == 'lif': 
            self.blocks[0][1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))

        elif self.config.spiking_neuron_type == 'plif': 
            self.blocks[0][1].insert(0, neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))


        if self.config.stateful_synapse:
            self.blocks[0][1].append(layer.SynapseFilter(tau=self.config.stateful_synapse_tau, learnable=self.config.stateful_synapse_learnable, 
                                                         step_mode='m'))


        ################################################   Hidden Layers    #######################################################

        for i in range(self.config.n_hidden_layers-1):
            self.block = [[layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias = self.config.bias, step_mode='m')],
                            [layer.Dropout(self.config.dropout_p, step_mode='m')]]
        
            if self.config.use_batchnorm: self.block[0].insert(1, layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m'))
            if self.config.spiking_neuron_type == 'lif': 
                self.block[1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            elif self.config.spiking_neuron_type == 'plif': 
                self.block[1].insert(0, neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            
            if self.config.stateful_synapse:
                self.block[1].append(layer.SynapseFilter(tau=self.config.stateful_synapse_tau, learnable=self.config.stateful_synapse_learnable, 
                                                             step_mode='m'))

            self.blocks.append(self.block)


        ################################################   Final Layer    #######################################################


        self.final_block = [[layer.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias = self.config.bias, step_mode='m')]]
        if self.config.spiking_neuron_type == 'lif':
            self.final_block.append([neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True)])
        elif self.config.spiking_neuron_type == 'plif': 
            self.final_block.append([neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True)])

        self.blocks.append(self.final_block)

        ################################################   Registering parameter groups   ###################################################

        
        self.model = [l for block in self.blocks for sub_block in block for l in sub_block]
        self.model = nn.Sequential(*self.model)

        self.positions = []
        self.weights = []
        self.weights_bn = []
        self.weights_plif = []
        for m in self.model.modules():
            if isinstance(m, layer.Linear):
                self.weights.append(m.weight)
                if self.config.bias:
                    self.weights_bn.append(m.bias)
            elif isinstance(m, layer.BatchNorm1d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)
            elif isinstance(m, neuron.ParametricLIFNode):
                self.weights_plif.append(m.w)



    def forward(self, x):
        

        # permute from (batch, time, neurons) to  (time, batch, neurons) for multi-step processing
        x= x.permute(1,0,2)

        for block_id in range(self.config.n_hidden_layers):
            x = self.blocks[block_id][0][0](x)

            if self.config.use_batchnorm:
                # we use x.unsqueeze(3) to respect the expected shape to batchnorm which is (time, batch, channels, length)
                # we do batch norm on the channels since length is the time dimension
                # we use squeeze to get rid of the channels dimension 
                x = self.blocks[block_id][0][1](x.unsqueeze(3)).squeeze()
            
            # we use our spiking neuron filter
            spikes = self.blocks[block_id][1][0](x)
            
            # we use dropout on generated spikes tensor
            x = self.blocks[block_id][1][1](spikes)

            # we apply synapse filter
            if self.config.stateful_synapse:
                x = self.blocks[block_id][1][2](x)

        
        ################################### Apply final layer
        out = self.blocks[-1][0][0](x)
        out = self.blocks[-1][1][0](out)

        if self.config.loss != 'spike_count':
            out = self.blocks[-1][1][0].v_seq

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
            opts.append(optim.Adam([{'params':self.weights, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_plif, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                    {'params':self.weights_bn, 'lr':self.config.lr_w, 'weight_decay':0}]))

        return opts