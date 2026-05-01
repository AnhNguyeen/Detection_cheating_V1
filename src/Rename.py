import os
def rename_images_in_folder(folder_path, prefix="image_"):
    files = os.listdir(folder_path) 
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')  
    count = 0  
    print(f"Bắt đầu đổi tên ảnh trong thư mục: {folder_path}")
    for filename in files:
        name, ext = os.path.splitext(filename)
        if ext.lower() in valid_extensions:  
            count += 1
            new_name = f"{prefix}{count}{ext.lower()}" 
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_name)
            os.rename(old_file_path, new_file_path) 
            print(f"Đã đổi tên: {filename}  -->  {new_name}")
    print(f"Đã đổi tên tổng cộng {count} ảnh.")
thu_muc_cua_anh = r"D:\Project\Nguyen_Trong_Anh\Abnormal"
rename_images_in_folder(thu_muc_cua_anh)