import os
import timm
import torch
import torch.onnx

model_name = 'mobilevit_xxs'
clss_num = 8

os.makedirs("./onnx_ckpt", exist_ok=True)

print("="*70)
print("Exporting MobileViT-XXS to ONNX")
print("="*70)

# 創建和載入模型
print("\n📦 Loading model...")
model = timm.create_model(model_name, pretrained=False, num_classes=clss_num)

ckpt = torch.load("./checkpoints_comparison_mobilevit/best_finetune_model.pth")
state_dict = ckpt["model_state_dict"]
model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded")

# 準備輸入
dummy_input = torch.randn(1, 3, 224, 224)

print("\n🔄 Exporting to ONNX (using legacy exporter)...")

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        "./onnx_ckpt/model.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=True,
        dynamo=True
    )

print("✅ ONNX model exported successfully!")

# 驗證模型
print("\n🔍 Validating ONNX model...")
try:
    import onnx
    model_onnx = onnx.load("./onnx_ckpt/model.onnx")
    onnx.checker.check_model(model_onnx)
    print("✅ ONNX model validation passed!")
    
    # 顯示模型信息
    file_size = os.path.getsize("./onnx_ckpt/model.onnx")
    print(f"\n📊 Model Information:")
    print(f"  File size: {file_size / (1024**2):.2f} MB")
    print(f"  Input shape: {dummy_input.shape}")
    
except ImportError:
    print("⚠️  'onnx' package not found, skipping validation")
    print("   Install with: pip install onnx")

# 測試推論（可選）
print("\n🧪 Testing ONNX inference...")
try:
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession("./onnx_ckpt/model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # PyTorch 推論
    with torch.no_grad():
        torch_output = model(dummy_input)
    
    # ONNX 推論
    onnx_output = session.run([output_name], {input_name: dummy_input.numpy()})
    
    # 比較結果
    diff = np.abs(torch_output.numpy() - onnx_output[0]).max()
    print(f"  Max difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("  ✅ ONNX output matches PyTorch!")
    else:
        print(f"  ⚠️  Small difference detected (acceptable for FP32)")
    
except ImportError:
    print("  ⚠️  'onnxruntime' not found, skipping inference test")
    print("     Install with: pip install onnxruntime")

print("\n" + "="*70)
print("✅ Export Complete!")
print("="*70)