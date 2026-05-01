from PIL import Image, ImageOps
from pathlib import Path
 
def compress_images(input_folder, output_folder=None, quality=85, max_size=(1920, 1920)):
    src = Path(input_folder)
    dst = Path(output_folder) if output_folder else src.parent / f"{src.name}_webp"
    dst.mkdir(parents=True, exist_ok=True)
 
    for file in src.iterdir():
        if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            with Image.open(file) as img:
                img = ImageOps.exif_transpose(img)
                img.thumbnail(max_size)
                img = img.convert("RGB")
                img.save(dst / f"{file.stem}.webp", format="WEBP", quality=quality, method=6)
            print(f"✓ {file.name}")
        except Exception as e:
            print(f"✗ {file.name}: {e}")
 
base = r"D:\Project\Nguyen_Trong_Anh"
compress_images(f"{base}/Abnormal")
compress_images(f"{base}/Normal")
print("Hoàn tất!")