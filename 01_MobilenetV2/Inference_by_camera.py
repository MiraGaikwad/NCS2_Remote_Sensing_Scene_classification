import time
import cv2
import time
from pathlib import Path
import numpy as np
from openvino.runtime import Core

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
(h, w) = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('Video_output.mp4', fourcc, 15.0, (w, h), True)


pb_path = Path("MobileNetV2_New_baseline.pb")
ie = Core()
model = ie.read_model(model="IR_model_FP32/IR_model.xml", weights="IR_model_FP32/IR_model.bin")
compiled_model = ie.compile_model(model=model, device_name="CPU")

import json
with open("label_map.json") as jsonFile:
    data = json.load(jsonFile)
#     jsonData = data["label_map"]
    
while cap.isOpened():
    ret, frame = cap.read()
    dim = (224, 224)
    rescaled_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    rescaled_frame1 = cv2.resize(frame, (500,500), interpolation=cv2.INTER_AREA)
    writer.write(rescaled_frame1)
    # write the output frame to file

#     image = cv2.cvtColor(cv2.imread(filename="airplane00_r.tif"), code=cv2.COLOR_BGR2RGB)
    input_key = next(iter(compiled_model.inputs))
    output_key = next(iter(compiled_model.outputs))
    network_input_shape = input_key.shape

    input_image = np.expand_dims(rescaled_frame, 0)
    result = compiled_model([input_image])[output_key]
    result_index = np.argmax(result)
    print(result_index)
    num_images = 122
    start = time.perf_counter()
    request = compiled_model.create_infer_request()
#     x = [request.infer(inputs={input_key.any_name: input_image}) for _ in range(num_images) ]  
    out = request.infer(inputs={input_key.any_name: input_image})

    end = time.perf_counter()
    time_ir = 1/(end - start)

    for key, value in out.items():
        val_list =value.tolist()
        class_val = val_list[0].index(max(val_list[0]))
        str_val = "This is " + str(class_val)
#         print(data)
        for keys, vals in data.items():
            if vals == class_val:
#                     keys = x.keys()
               fin_cls =keys
        
#         prt = str_val + keys

    fps = "FPS:" + str(round(time_ir, 2))

    cv2.putText(img=rescaled_frame1, text="This is:  " + fin_cls, org=(100, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
    cv2.putText(img=rescaled_frame1, text=fps, org=(20, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
    cv2.resize(rescaled_frame1, (500, 500) , interpolation=cv2.INTER_AREA)
    cv2.imshow("Output", rescaled_frame1)

#     print(f"IR model in Inference Engine/CPU: {time_ir/num_images:.4f} seconds per image, FPS: {num_images/time_ir:.2f}")
#     print("Inference time is:", time_ir)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
writer.release()