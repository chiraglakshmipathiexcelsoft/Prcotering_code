python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat  # Windows
```
```
### 2. Install PyTorch with CUDA 12.8
```
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


```
pip install onnxruntime-gpu
```


```bash
python -c "import torch; print('GPU Ready!' if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 12 else 'GPU Not Ready')"
```

```python
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Test GPU
x = torch.randn(1000, 1000).cuda()
y = torch.matmul(x, x.t())
print("GPU Test: PASSED")
```



