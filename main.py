import cv2
import os
import shutil
from findRectangles import find_rectangles_and_x

def clear_dir_safe(dirpath):
    abs_dir = os.path.abspath(dirpath)
    cwd = os.path.abspath(os.getcwd())

    if not abs_dir.startswith(cwd):
        raise ValueError(f"Refusing to remove dir outside working dir: {abs_dir}")

    if os.path.exists(abs_dir):
        shutil.rmtree(abs_dir)
        print(f"Deleted directory: {abs_dir}")
    
    os.makedirs(abs_dir, exist_ok=True)
    print(f"Created empty directory: {abs_dir}")

if __name__ == "__main__":
    rectangles_dir = "rectangles_full_image"
    clear_dir_safe(rectangles_dir)

    image_path = "/home/gegham/Screenshots/n2.png"
    input_ext = os.path.splitext(image_path)[1]

    os.makedirs(rectangles_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not open image: {image_path}")
        exit(1)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    output_path = os.path.join(rectangles_dir, f"annotated_full_image{input_ext}")

    result = find_rectangles_and_x(
        img,
        gray,
        output_path=output_path,
        min_area=200,
        max_overlap=0.1,
        save_crops=True,
        crops_dir=rectangles_dir,
        save_only_with_x=False, 
        save_thresh=False 
    )
    
    # Ստուգում ենք քանի արժեք է վերադարձվել
    if len(result) == 4:
        rectangles, x_marks, result_img, blocks = result
        has_blocks = True
    else:
        rectangles, x_marks, result_img = result
        blocks = []
        has_blocks = False

    cv2.imwrite(output_path, result_img)
    
    print(f"✅ Full image processed. Annotated: {output_path}. Crops dir: {rectangles_dir}")
    print(f"📊 Found rectangles: {len(rectangles)}, X-marks: {len(x_marks)}")
    
    if has_blocks:
        print(f"🔢 Number of blocks detected: {len(blocks)}")
        
        # Մանրամասն ինֆորմացիա բլոկների մասին
        for block_idx, block in enumerate(blocks):
            print(f"\n   Block {block_idx + 1}: {len(block)} rectangles")
            rows = {}
            for rect in block:
                row_num = rect.get('row', 0)
                if row_num not in rows:
                    rows[row_num] = 0
                rows[row_num] += 1
            print(f"   Structure: {len(rows)} rows")
            for row_num in sorted(rows.keys()):
                print(f"      Row {row_num}: {rows[row_num]} cells")
    else:
        print("⚠️  Old version detected - please update findRectangles.py for matrix grouping")