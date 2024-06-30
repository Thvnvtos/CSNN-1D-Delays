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

    epochs = 150
    batch_size = 64
    ################################################
    #               Model Achitecture              #
    ################################################
    # model type could be set to : 'snn'
    model_type = 'snn'

    n_outputs = 20 if dataset == 'shd' else 35


    ################################################
    #                Optimization                  #
    ################################################
    
    optimizer_w = 'adam'

    lr_w = 1e-3


    ################################################
    #                    Delays                    #
    ################################################
    



    ################################################
    #                 Fine-tuning                  #
    ################################################
    

    


    ################################################
    #               Data-Augmentation              #
    ################################################

   


    #############################################
    #                      Wandb                #
    #############################################