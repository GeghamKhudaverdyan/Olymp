import cv2
import os
import shutil
import time
from findRectangles import find_rectangles_and_x
from cutExcessives import test_smart_cutting
from export_results import export_results_to_formats
from compare_results import compare_with_correct 

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
    start_time = time.time()
    
    rectangles_dir = "rectangles_full_image"
    clear_dir_safe(rectangles_dir)
    
    image_path = test_smart_cutting("/home/gegham/Screenshots/student.png", save_path="/home/gegham/Screenshots/num1.png")
    input_ext = os.path.splitext(image_path)[1]
    os.makedirs(rectangles_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not open image: {image_path}")
        exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_path = os.path.join(rectangles_dir, f"annotated_full_image{input_ext}")

    result = find_rectangles_and_x(
        img, gray,
        output_path=output_path,
        min_area=200,
        max_overlap=0.1,
        save_crops=True,
        crops_dir=rectangles_dir,
        save_only_with_x=False, 
        save_thresh=False 
    )
    
    if len(result) == 4:
        rectangles, x_marks, result_img, blocks = result
        has_blocks = True
    else:
        rectangles, x_marks, result_img = result
        blocks = []
        has_blocks = False

    cv2.imwrite(output_path, result_img)

    if has_blocks:
        export_results_to_formats(rectangles, x_marks, blocks, 
                                 output_prefix=os.path.join(rectangles_dir, "result"))
    
    print(f"✅ Full image processed. Annotated: {output_path}")
    print(f"📊 Found rectangles: {len(rectangles)}, X-marks: {len(x_marks)}")
    
    # ՀԱՄԵՄԱՏՈՒՄ ՃԻՇՏ ՊԱՏԱՍԽԱՆՆԵՐԻ ՀԵՏ
    if os.path.exists("correct_answers.txt"):
        print(f"\n{'='*50}")
        print("🔍 ՍՏՈՒԳՈՒՄ ԵՄ ՊԱՏԱՍԽԱՆՆԵՐԸ...")
        print(f"{'='*50}\n")
        score, total, errors = compare_with_correct(
            os.path.join(rectangles_dir, "result_simple.txt")
        )
    else:
        print("\n⚠️  'correct_answers.txt' ֆայլը չկա")
        print("💡 Եթե սա ճիշտ պատասխանների թեստն է, պահիր այն:")
        print(f"   shutil.copy('{os.path.join(rectangles_dir, 'result_simple.txt')}', 'correct_answers.txt')")
    
    end_time = time.time()
    print(f"\n⏱️  Total execution time: {end_time - start_time:.2f} seconds")




















# import cv2
# import os
# import shutil
# import time  # Add this import
# from findRectangles import find_rectangles_and_x
# from cutExcessives import test_smart_cutting
# from export_results import export_results_to_formats

# def clear_dir_safe(dirpath):
#     abs_dir = os.path.abspath(dirpath)
#     cwd = os.path.abspath(os.getcwd())

#     if not abs_dir.startswith(cwd):
#         raise ValueError(f"Refusing to remove dir outside working dir: {abs_dir}")

#     if os.path.exists(abs_dir):
#         shutil.rmtree(abs_dir)
#         print(f"Deleted directory: {abs_dir}")
    
#     os.makedirs(abs_dir, exist_ok=True)
#     print(f"Created empty directory: {abs_dir}")

# if __name__ == "__main__":
#     start_time = time.time()
    
#     rectangles_dir = "rectangles_full_image"
#     clear_dir_safe(rectangles_dir)
    
#     image_path = test_smart_cutting("/home/gegham/Screenshots/n2.png", save_path="/home/gegham/Screenshots/num1.png")

#     # image_path = "/home/gegham/Screenshots/n5.png"
#     input_ext = os.path.splitext(image_path)[1]

#     os.makedirs(rectangles_dir, exist_ok=True)

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"❌ Could not open image: {image_path}")
#         exit(1)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     output_path = os.path.join(rectangles_dir, f"annotated_full_image{input_ext}")

#     result = find_rectangles_and_x(
#         img,
#         gray,
#         output_path=output_path,
#         min_area=200,
#         max_overlap=0.1,
#         save_crops=True,
#         crops_dir=rectangles_dir,
#         save_only_with_x=False, 
#         save_thresh=False 
#     )
    
#     if len(result) == 4:
#         rectangles, x_marks, result_img, blocks = result
#         has_blocks = True
#     else:
#         rectangles, x_marks, result_img = result
#         blocks = []
#         has_blocks = False

#     cv2.imwrite(output_path, result_img)

#     if has_blocks:
#         export_results_to_formats(rectangles, x_marks, blocks, 
#                                  output_prefix=os.path.join(rectangles_dir, "result"))
    
    
#     print(f"✅ Full image processed. Annotated: {output_path}. Crops dir: {rectangles_dir}")
#     print(f"📊 Found rectangles: {len(rectangles)}, X-marks: {len(x_marks)}")
    
#     if has_blocks:
#         print(f"🔢 Number of blocks detected: {len(blocks)}")
        
#         for block_idx, block in enumerate(blocks):
#             print(f"\n   Block {block_idx + 1}: {len(block)} rectangles")
#             rows = {}
#             for rect in block:
#                 row_num = rect.get('row', 0)
#                 if row_num not in rows:
#                     rows[row_num] = 0
#                 rows[row_num] += 1
#             print(f"   Structure: {len(rows)} rows")
#             for row_num in sorted(rows.keys()):
#                 print(f"      Row {row_num}: {rows[row_num]} cells")
#     else:
#         print("⚠️  Old version detected - please update findRectangles.py for matrix grouping")
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds")