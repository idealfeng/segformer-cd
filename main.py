# test_model_structure.py
from transformers import SegformerForSemanticSegmentation
from config import cfg

model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/mit-b1',
    num_labels=cfg.NUM_CLASSES,
    ignore_mismatched_sizes=True
)

print("decode_head的属性:")
print(dir(model.decode_head))

print("\ndecode_head的子模块:")
for name, module in model.decode_head.named_children():
    print(f"  {name}: {type(module)}")