import shutil
import os
import sys

# Step 1: Run the training script
training_script = os.path.join(os.path.dirname(__file__), 'Training_py.py')
print("🚀 Running training script...")
exit_code = os.system(f'python "{training_script}"')

if exit_code != 0:
    print("❌ Training script failed.")
    sys.exit(1)

# Step 2: Move model and vectorizer to backend/
src_dir = os.path.dirname(__file__)
dst_dir = os.path.join(src_dir, '..', 'backend')

src_model = os.path.join(src_dir, 'model.pkl')
src_vectorizer = os.path.join(src_dir, 'vectorizer.pkl')

dst_model = os.path.join(dst_dir, 'model.pkl')
dst_vectorizer = os.path.join(dst_dir, 'vectorizer.pkl')

print("📦 Copying model and vectorizer to backend...")
shutil.copy(src_model, dst_model)
shutil.copy(src_vectorizer, dst_vectorizer)

print("✅ Model and vectorizer updated in backend successfully.")
