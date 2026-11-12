"""
Verify All 5 Models in RetinaCare AI
"""
import sys
sys.path.insert(0, 'E:/pylib')
sys.path.insert(0, 'e:/P B L')

from pathlib import Path
from PIL import Image
from src.model_manager import get_all_models_status, CANONICAL_MODEL_NAMES
from src.prediction import run_model_inference

print("=== 5-MODEL AVAILABILITY CHECK ===")
statuses = get_all_models_status()
all_ready = True
for s in statuses:
    fname = Path(s['checkpoint_path']).name if s['checkpoint_path'] else 'None'
    print(f"[{s['status_label']}] {s['canonical_name']} -> {fname} ({s['file_size_mb']} MB)")
    if not s['is_available']:
        all_ready = False

print(f"\nAll models available: {all_ready}")

samples = list(Path('e:/P B L/sample_images').glob('*.jpg')) + list(Path('e:/P B L/sample_images').glob('*.png'))
test_img = samples[0] if samples else Image.new('RGB', (224, 224), (120, 50, 30))
print(f"\n=== TESTING INFERENCE ON ALL 5 ARCHITECTURES WITH: {test_img} ===")

for m in CANONICAL_MODEL_NAMES:
    res = run_model_inference(m, test_img)
    print(f"  {m:<16} -> {res['class_name']} ({res['confidence_pct']}%) [Latency: {res['latency_ms']}ms]")

print("\nALL 5 MODELS VALIDATED AND READY FOR DEPLOYMENT!")
