# Plan and todo's

## Silvilaser

General outline:

Occlusion mapping and completion: demo on CLS data  
    - First show occlusion mapping, talk about empty pulses modelling and talk about usefullnes and occpy
    - Then show point cloud completion experiment and discuss why could be useful (as compared to just foundational model)
    - Talk about potential future work: integrating the two steps, pretraining, other types of data: winter vs summer, UAV+TLS

Note: only discuss Peru 1 and 2 and COL for co-author issue

### Part one: occlusion mapping on CLS data: already done

maybe rerun with empty pulses modelling? as we can use this already as big part of presentation

### Part two: point cloud completion on CLS data

Goal: initial experiment

Requirements:
 - Process input data:  
         - get cubes of CLS data where there is at least some points not present in the TLS data  DONE
         - divide into testing, training and validation cubes DONE
 - Train a model: start from Aline's weights, retrain on CLS cubes DONE
 - Get some qualitative results, maybe on independent trees or just on test cubes. TODO

+ Discuss potential of integrating occlusion mapping and point cloud completion

## Paper

Goal: expand on silvilaser demo by applying to more practical data

IDEA: TLS + UAV-LS of deciduous forest: try to get more accurate tree structural parameters
    - completing TLS data: better crown cover, better branch length from qsm (?), better PA and alpha volume
    - completing UAV data: better DBH and height -> better inventory at larger scale
IDEA: completing summer data from winter data: upper branches -> can be done with Wytham

If possible, integrate occlusion mapping (might be seperate work)