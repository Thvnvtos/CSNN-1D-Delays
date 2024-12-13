from spikingjelly.activation_based import surrogate

class Config:
    
    ################################################
    #            General configuration             #
    ################################################
    
    debug = False

    seed = 0

    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    # ONLY SHD FOR NOW  
    dataset = 'ssc'                    
    datasets_path = '../Datasets/SSC'

    time_step = 10
    n_bins = 1

    epochs = 50
    batch_size = 32
    ################################################
    #               Model Achitecture              #
    ################################################
    
    model_type = 'csnnext-delays'                               # model type could be set to : 'csnnext-delays', 'csnn-1d', 'csnn-1d-delays' or 'dwsep-csnn-1d-delays'


    spiking_neuron_type = 'lif'                                 # plif, lif
    init_tau = 20                                               # in ms, can't be < time_step
    init_tau = (init_tau  +  1e-9) / time_step

    
    n_inputs = 700
    n_C = 16                                                   # base number of conv channels


    stem_kernel_size = 7
    stem_stride = 7

    n_stages = 4
    n_blocks = [1, 1, 3, 1] #[1,3,1]

    channels = [n_C, 2*n_C, 4*n_C, 8*n_C]

    kernel_sizes =   [7, 7, 7, 5] #[7, 7, 7] 
    strides =       [1, 1, 1, 1] 
    downsampling_kernel_sizes = [2, 2, 2]
    downsampling_strides = [2, 2, 2]
    

    n_outputs = 20 if dataset == 'shd' else 35

    batchnorm_type = 'SJ_bn1d'                                      # 'bn1' = 1D BN ignoring time, 'bn2' = 2D BN considering (Freqs, Time) as the 2 dimensions (Maybe add SNN specific BNs next)

    dropout_p = 0.5  # 0.75
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
    lr_pos = 100*lr_w

    weight_decay = 2e-5

    max_lr_w = 5 * lr_w
    t_max_pos = epochs

    ################################################
    #                    Delays                    #
    ################################################
    DCLSversion = 'max'                                     # 'gauss',  'max',   'v1'
    decrease_sig_method = 'exp'

    kernel_count = 1
    
    max_delays = [70, 90, 110, 110] # [70, 90, 110, 110]

    sigInits = [1/3, 1/3, 1/2, 1/2]       #[1/3, 1/3, 1/2, 1/2] 
    final_epoch = (1*epochs)//3     
    

    left_paddings = [None] * len(max_delays)    #defined in init
    right_paddings = [0] * len(max_delays)      #(max_delay-1) // 2

    init_pos_method = 'uniform'
    init_pos_mode = 'random'            # 'random' or 'rm' (right-most)
    init_pos_a = [None] * len(max_delays) 
    init_pos_b = [None] * len(max_delays) 

    sig_final_vmax = 1e-6
    sig_final_gauss = 0.23                                      # Remember why it's specifically 0.23 for vgauss in dcls

    # Exponential decreasing coefficient, defined in init
    alpha = [None] * len(max_delays)     

    #############################################
    #                      Wandb                #
    #############################################
    # If use_wandb is True, specify your wandb API key in wandb_API_key and the project and run names.

    use_wandb = True
    wandb_API_key = '25f19d79982fd7c29f092981a100f187f2c706b4'
    wandb_project_name = 'CSNN-1D-Delays'

    run_name = 'Inverted_Bottleneck|CSnnNext-Delays|stem7|bins=1|4-stages|n_C=16'

    run_info = f'||{model_type}||{dataset}'

    wandb_run_name = run_name  + run_info + f'||seed={seed}'
    wandb_group_name = run_name + run_info

    

    def __init__(self):
        self.max_delays = [md//self.time_step for md in self.max_delays]
        self.max_delays = [md if md%2==1 else md+1 for md in self.max_delays]


        self.sigInits = [md*self.sigInits[i] for i,md in enumerate(self.max_delays)]


        self.left_paddings = [md - 1 for md in self.max_delays]

        for i, md in enumerate(self.max_delays):
            if self.init_pos_mode == 'random':
                self.init_pos_a[i] = -md//2 + 1
            elif self.init_pos_mode == 'rm':
                self.init_pos_a[i] = md//2
            self.init_pos_b[i] = md//2

        for i, sig in enumerate(self.sigInits):
            if self.DCLSversion == 'gauss' and self.final_epoch > 0:
                if self.decrease_sig_method == 'exp':
                    self.alpha[i] = (self.sig_final_gauss/sig)**(1/self.final_epoch)
            elif self.DCLSversion == 'max' and self.final_epoch > 0:
                if self.decrease_sig_method == 'exp':
                    self.alpha[i] = (self.sig_final_vmax/sig)**(1/self.final_epoch)
