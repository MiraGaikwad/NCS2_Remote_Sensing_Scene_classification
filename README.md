# NCS2_Remote_Sensing_Scene_classification
Remote Sensing Scene Classification on Intel Neural Compute Stick 2 

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

