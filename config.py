from spikingjelly.activation_based import surrogate

class Config:
    
    ################################################
    #            General configuration             #
    ################################################
    
    debug = False

    seed = 0

    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    # ONLY SHD FOR NOW  
    dataset = 'shd'                    
    datasets_path = 'Datasets/SHD'

    time_step = 10

    epochs = 20
    batch_size = 128
    ################################################
    #               Model Achitecture              #
    ################################################
    
    model_type = 'csnn-1d'           # model type could be set to : 'csnn-1d', 'csnn-1d-delays'


    spiking_neuron_type = 'lif'             # plif, lif
    init_tau = 10.05                        # in ms, can't be < time_step

    stateful_synapse_tau = 10.0             # in ms, can't be < time_step
    stateful_synapse = False
    stateful_synapse_learnable = False

    
    n_inputs = 700
    n_C = 16                                # base number of conv channels

    n_layers = 4
    kernel_sizes = [5, 5, 2, 2]
    strides = [5, 5, 2, 2]
    channels = [n_C, 2*n_C, 4*n_C, 8*n_C]

    n_outputs = 20 if dataset == 'shd' else 35

    use_batchnorm = True
    bias = True
    detach_reset = True


    v_threshold = 1.0
    alpha = 5.0
    surrogate_function = surrogate.ATan(alpha = alpha)


    loss = 'sum'                                                # 'mean', 'max', 'spike_count', 'sum
    loss_fn = 'CEloss'
    output_v_threshold = 2.0 if loss == 'spike_count' else 1e9  #  use 1e9 for loss = 'mean' or 'max'


    ################################################
    #                Optimization                  #
    ################################################
    
    optimizer_w = 'adam'

    lr_w = 1e-3

    weight_decay = 1e-5


    ################################################
    #                    Delays                    #
    ################################################
    DCLSversion = 'gauss' if model_type =='csnn-1d-delays' else 'max'
    decrease_sig_method = 'exp'
    kernel_count = 1

    max_delay = 200//time_step
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number
    
    # For constant sigma without the decreasing policy, set model_type == 'snn_delays' and sigInit = 0.23 and final_epoch = 0
    sigInit = max_delay // 2        if model_type == 'csnn-1d-delays' else 0
    final_epoch = (1*epochs)//4     if model_type == 'csnn-1d-delays' else 0


    left_padding = max_delay-1
    right_padding = 0#(max_delay-1) // 2

    init_pos_method = 'uniform'
    init_pos_a = -max_delay//2
    init_pos_b = max_delay//2


    ################################################
    #                 Fine-tuning                  #
    ################################################
    

    


    ################################################
    #               Data-Augmentation              #
    ################################################

   


    #############################################
    #                      Wandb                #
    #############################################