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
    datasets_path = '../Datasets/SHD'

    time_step = 10
    n_bins = 5

    epochs = 100
    batch_size = 64
    ################################################
    #               Model Achitecture              #
    ################################################
    
    model_type = 'dwsep-csnn-1d-delays'                               # model type could be set to : 'csnn-1d', 'csnn-1d-delays' or 'dwsep-csnn-1d-delays'


    spiking_neuron_type = 'lif'                                 # plif, lif
    init_tau = 30                                              # in ms, can't be < time_step
    init_tau = (init_tau  +  1e-9) / time_step

    
    n_inputs = 700
    n_C = 64                                                   # base number of conv channels

    n_layers = 4
    kernel_sizes =  [5, 2, 2, 2]
    strides =       [1, 2, 2, 2]
    channels = [n_C, 2*n_C, 4*n_C, 8*n_C]

    n_outputs = 20 if dataset == 'shd' else 35

    batchnorm_type = 'bn1'                                      # 'bn1' = 1D BN ignoring time, 'bn2' = 2D BN considering (Freqs, Time) as the 2 dimensions (Maybe add SNN specific BNs next)

    dropout_p = 0.75
    bias = False
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
    lr_pos = 100*lr_w   if model_type =='csnn-1d-delays' or model_type =='dwsep-csnn-1d-delays' else 0

    weight_decay = 1e-5


    max_lr_w = 5 * lr_w
    t_max_pos = epochs

    ################################################
    #                    Delays                    #
    ################################################
    DCLSversion = 'max'                                     # 'gauss',  'max',   'v1'
    decrease_sig_method = 'exp'

    kernel_count = 1

    max_delay = 100//time_step
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number
    

    sigInit = max_delay // 2        
    final_epoch = (1*epochs)//4     
    

    left_padding = max_delay-1
    right_padding = 0#(max_delay-1) // 2

    init_pos_method = 'uniform'
    init_pos_a = max_delay//2#-max_delay//2 + 1
    init_pos_b = max_delay//2

    sig_final_vmax = 1e-6
    sig_final_gauss = 0.23                                      # Remember why it's specifically 0.23 for vgauss in dcls

    if DCLSversion == 'gauss' and final_epoch > 0:
        if decrease_sig_method == 'exp':
            alpha = (sig_final_gauss/sigInit)**(1/final_epoch)
    elif DCLSversion == 'max' and final_epoch > 0:
        if decrease_sig_method == 'exp':
            alpha = (sig_final_vmax/sigInit)**(1/final_epoch)

    #############################################
    #                      Wandb                #
    #############################################
    # If use_wandb is True, specify your wandb API key in wandb_API_key and the project and run names.

    use_wandb = True
    wandb_API_key = '25f19d79982fd7c29f092981a100f187f2c706b4'
    wandb_project_name = 'CSNN-1D-Delays'

    run_name = 'Max|DW-sep'


    run_info = f'||{model_type}||{dataset}'

    wandb_run_name = run_name  + run_info + f'||seed={seed}'
    wandb_group_name = run_name + run_info

    ################################################
    #                 Fine-tuning                  #
    ################################################
    

    ################################################
    #               Data-Augmentation              #
    ################################################
