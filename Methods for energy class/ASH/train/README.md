# ResNet50
* Models are trained using PyTorch's image classification reference training script [git://pytorch/vision/references/classification](https://github.com/pytorch/vision/tree/main/references/classification)

### resnet50.V1
`torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`

From the official [recipe](https://github.com/pytorch/vision/tree/main/references/classification#resnet).
Download using TorchVision.
```sh
torchrun --nproc_per_node=8 train.py --model resnet50
```

### resnet50.V2
`torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`

From the official [recipe](https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621).
Download using TorchVision.
```sh
torchrun --nproc_per_node=8 train.py --model resnet50 --batch-size 128 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4
```

### resnet50.A
* resnet50.V1 with labal smoothing, mixup, cutmix
* trained on 4x A6000 GPUs
```sh
torchrun --nproc_per_node=4 train.py --model resnet50 --batch-size 64 --lr 0.1 \
--label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--data-path=/data/Imagenet1K --output-dir=./resnet50/A
```
```
Test:  Acc@1 75.996 Acc@5 93.044
Training time 18:18:55
```

### resnet50.B
* resnet50.V2 without labal smoothing, mixup, cutmix
* trained on 4x A6000 GPUs
```sh
torchrun --nproc_per_node=4 train.py --model resnet50 --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps 4 \
--data-path=/data/Imagenet1K --output-dir=./resnet50/B
```
```
Test: EMA Acc@1 78.104 Acc@5 93.828
Training time 3 days, 6:33:54
```

### resnet50.C
* resnet50.V1 with labal smoothing, mixup, cutmix, batch_size, long_training, cosineannealinglr
* trained on 4x A6000 GPUs
```sh
torchrun --nproc_per_node=4 train.py --model resnet50 --batch-size 256 --lr 0.5 --epochs 600 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--data-path=/data/Imagenet1K --output-dir=./resnet50/C
```
```
Test:  Acc@1 79.022 Acc@5 94.376
Training time 4 days, 15:45:29
```