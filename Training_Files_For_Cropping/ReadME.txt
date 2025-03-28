      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      50/50      4.78G     0.3183     0.2525     0.9608         12        640: 100%|██████████| 101/101 [00:20<00:00,  4.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 15/15 [00:02<00:00,  5.17it/s]
                   all        452        526      0.914       0.96      0.976       0.91

50 epochs completed in 0.332 hours.
Optimizer stripped from runs\detect\train6\weights\last.pt, 22.5MB
Optimizer stripped from runs\detect\train6\weights\best.pt, 22.5MB

Validating runs\detect\train6\weights\best.pt...
Ultralytics 8.3.94 🚀 Python-3.10.0 torch-2.6.0+cu118 CUDA:0 (NVIDIA GeForce RTX 3070, 8192MiB)
Model summary (fused): 72 layers, 11,127,519 parameters, 0 gradients, 28.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 15/15 [00:07<00:00,  1.99it/s]
                   all        452        526      0.936      0.964      0.982      0.914
                   b_a        224        224      0.937      0.997      0.972      0.889
                 big_a         13         13      0.888          1      0.984      0.941
                   f_a        237        237      0.906       0.97      0.964      0.884
              half_b_a         16         16       0.95          1      0.995       0.95
              half_f_a         36         36          1      0.853      0.995      0.906
Speed: 0.4ms preprocess, 6.5ms inference, 0.0ms loss, 2.0ms postprocess per image
Results saved to runs\detect\train6