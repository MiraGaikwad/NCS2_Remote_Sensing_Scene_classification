# NCS2_Remote_Sensing_Scene_classification
Remote Sensing Scene Classification on Intel Neural Compute Stick 2 

## Files in each model folder

1. Baseline and pruned models                   - <Model name>_Baseline_Pruned.ipynb  
2. Quantised models                             - <Model name>_Quantization.ipynb
3. Freeze the model                             - Convert_to_PB_<Model name>.ipynb
4. Inference pipeline and time calculation      - hello_UCMerced_time.py 
5. Load model to device plugin time calculation - hello_UCMerced_load_time.py
6. Batch Inference on 120 images                - Batch_inference_120.ipynb
7. IR_model_FP16/                               - IR Files for FP16 models
8. IR_model_FP32/                               - IR Files for FP32 models 

## H5 models are frozen using following code 

import tensorflow as tf

model = tf.keras.models.load_model('<Model Name>.h5')
tf.saved_model.save(model,'saved_model')

or 

Using the code in file "01_MobilenetV2/Convert_to_PB_MobileNetV2.ipynb"


## Frozen model are converted into Intermediate Representation (IR) 

Before running following command OPENVINO environment needs to be initialised

python mo_tf.py --data_type FP16 --input_model tf_model.pb --model_name IR_model --output_dir IR_model/ --input_shape [1,224,224,3]


or 

MO command -

mo --saved_model_dir saved_model --data_type FP16 --output_dir IR_model/ --input_shape [1,224,224,3]

