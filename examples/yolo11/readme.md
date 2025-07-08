# YOLO11 Pruning

## 0. Requirements

```bash
conda create -n pruning python=3.9
conda activate pruning
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
Tested environment:
```
Pytorch==1.12.1
Torchvision==0.13.1
```

## 1. Pruning

#### Ultralytics
```bash
git clone https://github.com/ultralytics/ultralytics.git 
cp yolo11_pruning.py ultralytics/
cd ultralytics 
git checkout 6dffa0cef3e30e765b1622d5cc02d57b04d7ee8b # for compatibility
```
## 2. Usage
### Basic Pruning
Run the pruning script with default parameters:
```bash
python yolo11_pruning.py --model yolo11n.pt --cfg default.yaml
```
### Advanced Pruning
Run the pruning script with advanced parameters:
```bash
python yolo11_pruning.py \
    --model yolo11s.pt \
    --cfg custom_config.yaml \
    --iterative-steps 5 \
    --target-prune-rate 0.3 \
    --max-map-drop 0.15
```
### Parameters
- --model : Path to the pre-trained YOLOv11 model (default: yolo11n.pt )
- --cfg : Configuration file for training parameters (default: default.yaml )
- --iterative-steps : Number of pruning iterations (default: 10)
- --target-prune-rate : Target pruning ratio (default: 0.5)
- --max-map-drop : Maximum allowed mAP drop threshold (default: 0.2)

## 3. Pruning Process
### Key Features
1. C3k2 Module Replacement : Automatically replaces C3k2 modules with pruning-compatible C3k2_v2 versions
2. Iterative Pruning : Performs gradual pruning over multiple steps to maintain model performance
3. Fine-tuning : After each pruning step, the model is fine-tuned to recover performance
4. Early Stopping : Stops pruning if mAP drop exceeds the specified threshold
5. Performance Tracking : Generates performance graphs showing mAP and MACs changes
### Pruning Algorithm
- Uses GroupNormPruner with GroupMagnitudeImportance (L2 norm based)
- Ignores detection heads ( Detect , C2PSA , PSABlock ) during pruning
- Applies uniform pruning ratio across all iterations

## 5. Outputs
### Console Output
The script provides detailed information during execution:
```bash
=== Model Structure Before Pruning ===
Model: DetectionModel
Total parameters: 2,624,080
Trainable parameters: 2,624,080
Before Pruning: MACs= 3.27583 G, #Params= 2.62408 M, mAP= 0.50788

=== Model Structure After Pruning Step 1 ===
Total parameters: 2,374,149
Trainable parameters: 2,374,149
After pruning iter 1: MACs=2.9445236 G, #Params=2.374149 M, mAP=0.0, speed up=1.1125151790259042
```
### Generated Files
1. Pruned Models : Saved in runs/detect/stepX_finetune/weights/
   
   - best.pt : Best performing model after fine-tuning
   - last.pt : Last checkpoint
2. Performance Graph : yolo11_pruning_perf_change.png
   
   - Shows mAP recovery and MACs reduction over pruning steps
   - Displays both pruned mAP (before fine-tuning) and recovered mAP (after fine-tuning)
3. ONNX Export : Final pruned model exported to ONNX format