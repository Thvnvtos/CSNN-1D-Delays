# CSNN-1D-Delays

This is mostly for my offline usage, to take notes and remember what to do next. Also, for Tim to follow the progress of the project.

## To-Do: 

### Priority 1:

- Do some Tests for CSnnNext (for potential bugs)
- Launch multiple runs with a fast-to-run config to get some first results with CSnnNext

### Priority 2: 

- Change sigmas wandb plotting in model.py to all sigmas instead of just the first stage sigma 


### Notes:
- Add delays to the FC layer  ?
- Add weights initialization for Conv layers and for final FC layer
- Decrease Sig during batches instead of epochs
- Log more information like (weight parameters, position changes, gradients, all sigmas, etc...) to catch potential bugs


###

# First set of runs:

All have loss converge to 0.00x
bins=1 best_config_shd(3stages) => 96.45% (consistently > 95%)
bins=2 best_config_shd(3stages) => 96.05% (also consistently > 95%)
bins=5 best_config_shd(3stages) => 91.96% (around 90-91%)


# Second set of runs:     (loss converge to 0.000x)
As I've seen that smaller bins gave better accuracy I added a 4th stage to the network with 1 block:
All below are 4 stages:

bins=2 => 92.83% (consistently around 92%) (time = 18h)
bins=1 =>  95.95% (consistently around 94-95%) (time = 24h)
bins=1 (bigKS in last stages 5->7, smaller MD and smaller sigInit) => 95.66% (consistently around 94-95%)  (time = 24h 22mins)

notes: 
- distribution of delays (taken on first block of every stage) is focused on the extremeties with layer 2 and layer 2 having much more negative (left) delays and layer 3 having a tiny bit more on the positive extremity than the left.
- Maybe 4 stages would be better for SSC (expect really high run time), for shd it seems that it's better to stick to 3 stages (or 4 stages with half n_C)



# Third set of runs:

- bins = 1, 4 stages, n_C/2             =>      96.73%  (3 stages seems to have a better generalization on average, best_valid_acc is better with this)  (10h30mins)
- bins = 1, 3 stages, timestep/2        =>      95.16%  (valid acc worse than ts, consistently around 94.5%) (7h)
- bins = 2, 3 stages, timestep/2        =>      95.35%  (valid perf seems lower than bins=1 on average) (5h30mins)

notes:
- loss -> 0.00x 
- reducing n_C from 64 to 32 reduces time from 24h to 10h30mins
- more binning also lowers time
- More (also smaller timesteps) bins seem to make train acc and loss converge faster (but to similar final value) thus more overfitting (maybe)



# 4th set of runs (testing RM)

- RM|bins=1|4-stages|n_C=32         =>  96.45%  (very similar to random init with same config, need more tests to decide, delay distribution also very similar) (17h30mins)
- RM|bins=1|3-stages|n_C=64         =>  96.64%  (generalization on average seems a little better than other runs but very little)   (20h)
                                                (Delay distribution is slighly different but not significative)
- SmallMD|bins=1|3-stages|n_C=64    =>  95.67%  (performance is slighly worse than other runs BUT in both train and valid)  (20h)
                                                (delay distribution: mostly on extremitites apart last layer which is 70 instead of 50)


- Need to understand why RM|bins=1|4-stages|n_C=32 took more time than the random one (is there a bug) also in 2nd run
    No for stem7 RM vs RM same time, maybe just diff GPU



# 5th set of runs (testing 7,7 stem):


- stem7|RM|bins=1|4-stages|n_C=32         =>    94.67%      (has the most overfitting on average, average valid around 93.5%)  (13h30mins)
- stem7|RM|bins=1|3-stages|n_C=64         =>    95.91%      (Lower on average than not using stem, but maybe need to give it more epochs ?) (14h20mins)
- stem7|RM|bins=1|3-stages|n_C=32         =>    96.60%      (Seems to be one of the best in terms of average generalization)  (11h15mins)


- Stem7 doesnt seem to help, and causes more generalization than not using it, need more testing.
- Layer 0 delay distribution for 4-stages and 3-stages different, more on left side for 3-stages ?



stem 7 no RM

- stem7|bins=1|4-stages|n_C=32          =>  96.20%  (better generalization than RM version, but worse than stem5 versions) (13h 35mins)
- stem7|bins=1|3-stages|n_C=64          =>  95.10%  (Bad generalization)    (14h22mins)
- stem7|bins=1|3-stages|n_C=32          =>  96.76%  (Good generalization but similar to RM)      (11h16mins)



Test also stem5 smallest:

- stem5|bins=1|3-stages|n_C=32          =>  96.39%  (Average generalization very similar to other best runs, but stem7 seems more stable and higher best)   (12h43mins)

Then test not using decreasing sigma:

- stem5|bins=1|3-stages|n_C=32|sig=0    =>  96.02%  (Lower Average generalization than similar with decreasing_sig - Look graphs)   (12h43mins)
- stem7|bins=1|3-stages|n_C=32|sig=0    =>  94.91%  (Lower average generalization similar with decreasing sig)  (11h)


Notes: decreasing sig seems to have more delays in the middle whereas sig=0 have more delays on extremities


     

# 6th set of runs (testing SSC):


- stem7|bins=1|4-stages|n_C=32          =>  82.08%  (3days 11h)
- stem7|bins=1|3-stages|n_C=32          =>  80.07%  (2days 23h)


- stem10|bins=1|3-stages|n_C=32         =>  96.10% (average performance lower than stem5 and stem7) (9h45mins)

 
## Test blocks = [1, 1, 3]
## Test plif (Add plif wandb logging)


SSC: 
- stem7|bins=1|4-stages|n_C=64          =>  82.06%
- stem7|bins=1|3-stages|n_C=64          =>  82.73%
SHD:
- stem7|bins=1|3-stages|n_C=32|plif         => 95.55% (average performance lower than only LIF)
- stem7|bins=1|3-stages|n_C=32|No-Delays    => 91.59%





- Add an inverted bottleneck
- Try [1, 1, 3]
- Delays in the FC layer

- Add logging tau and positions maybe

- Test no-delays same network


GSC-Tests:
- train1: bins=1|3-stages|nc=64                 < 95.06%
- train2: bins=2|windwsize/2|3-stages|nc=64     = 95.25%

Inverted bottleneck:
- train3: nc=32|shd         = 94.48%
- train4: nc=32|ssc         = Not working






- inverted bottleneck nc=32 better train perf than normal with n_C=64 ,   but similar test perf (best acc test = 82.62%)
- 1 BN between inverted conv 4*nc and LIF  >  2 BNs, (BN after 2nd conv)



- train1: Baseline-Light
- train2: Baseline + 4-stages
- train3: Baseline + V1
- train4: Baseline + RM_init
- train5: Baseline + PLIF