import os
import string
from PIL import Image

input_folder = "raw_az_photos"       # Jahan Member 3 ki original photos rakhi hain
output_folder = "static/signs/az"    # Jahan final photos save hongi
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        
        # 1. Center-Square Crop (removes side green edge & centers hands)
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        
        # 2. Resize to clean 500x500 px
        img = img.resize((500, 500), Image.Resampling.LANCZOS)
        
        # 3. Save as uppercase letter .jpg under 100 KB
        letter_name = os.path.splitext(filename)[0].strip().upper() + ".jpg"
        out_path = os.path.join(output_folder, letter_name)
        img.save(out_path, "JPEG", quality=80, optimize=True)
        
        size_kb = os.path.getsize(out_path) / 1024
        print(f"Saved {letter_name}: {size_kb:.1f} KB")

# 4. Verify if any letter from A-Z is missing
existing = {os.path.splitext(f)[0] for f in os.listdir(output_folder)}
missing = [ch for ch in string.ascii_uppercase if ch not in existing]
print("\nMissing letters:", missing if missing else "None! All A-Z ready.")
