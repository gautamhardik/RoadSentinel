import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from ultralytics import YOLO

model = YOLO('weights/bdd100k_yolo11s.pt')
print("Model Task:", model.task)
print("Model Names:", model.names)
print("Model Parameters Count:", sum(p.numel() for p in model.model.parameters()))

# Inspect detection head
head = model.model.model[-1]
print("Head class:", head.__class__.__name__)
print("Number of detection layers / strides:", getattr(head, 'stride', None))
print("Number of classes in head:", getattr(head, 'nc', None))
print("Number of channels per head layer:", getattr(head, 'ch', None))
