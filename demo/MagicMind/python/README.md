# YOLOX-CPP-MagicMind

## Tutorial

### Step1
Install MagicMind relative software and python wheel. Please following the reference https://developer.cambricon.com.

### Step2
Use provided tools to generate onnx file.
For example, if you want to generate onnx file of yolox-m, please run the following command:
```shell
cd <path of yolox>
python3 tools/export_onnx.py --output-name yolox_m.onnx -n yolox-m -c yolox_m.pth
```
Then, a yolox_m.onnx file is generated.

### Step3
Generate MagicMind model.
```shell
cd demo/MagicMind/python/
python onnx2mm.py --onnx  ../../../yolox_m.onnx
```
Then, a yolox_m_int8fp16.model file is generated.

### Step4
Use MagicMind model to infer.
```shell
python mm_infer.py
```

### Step5
Evaluate model on COCO dataset.
```shell
python mm_eval.py -b 1 -d 1 --conf 0.001 -n yolox-m --mm-file-name yolox_m_int8fp16.model
```

### Step6
Profile MagicMind model.
```shell
python mm_perf.py --mm_file_name yolox_m_int8fp32.model
```
Use tensorboard to view profiling results.
```shell
tensorboard --port 8833 --logdir profile_data_output_dir/ --bind_all
```
