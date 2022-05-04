# Computer-vision based Long time parked vehicle detection

## Set parameters
Set parameters at `utils\others.py`

Best paramters for `ISLab` dataset: ISLab_event_car_0.3_True_0.8_True_0.1_1000_0.6_True_5__DS_0.5_70

Best parameters for `xd_full` dataset: xd_full_event_car_0.3_True_0.9_True_0.1_1000_0.6_True_1__DS_0.5_70

Best parameters for `Sussex` dataset: Sussex_event_car_0.3_False_0.9_False_0.1_1000_0.6_False_1__DS_0.5_5

## Run experiments

Run `python demo_with_mask.py -n "ISLab"`

Add `--debug` to show full yolox + deepsort results

Add `--mask` to show masked unauthorized parking area

## Evaluate results

Run `python evaluator.py -n "ISLab"  -m "event"`