import cv2
import numpy as np
import os

def find_all_rectangles(thresh, min_area=500):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        area = cv2.contourArea(contour)
        
        if len(approx) == 4 and area > min_area:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            if 0.5 < aspect_ratio < 2.0:
                rectangles.append({
                    'contour': approx,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area,
                    'sides': 4
                })
    
    return rectangles


def find_elongated_shapes(thresh, min_area=500, elongation_threshold=1.7):

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    elongated_shapes = []
    ignored_shapes = []
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(contour)
        num_sides = len(approx)
        
        if area > min_area and num_sides != 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            
            shape_info = {
                'contour': approx,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'sides': num_sides,
                'aspect_ratio': aspect_ratio
            }
            
            if aspect_ratio >= elongation_threshold:
                elongated_shapes.append(shape_info)
            else:
                ignored_shapes.append(shape_info)
    
    return elongated_shapes, ignored_shapes


def cut_shape_in_half(thresh, shape, orientation='auto'):

    x, y, w, h = shape['x'], shape['y'], shape['w'], shape['h']
    
    if orientation == 'auto':
        orientation = 'horizontal' if w > h else 'vertical'
    
    cut_thickness = max(3, int(min(w, h) * 0.15))
    
    if orientation == 'horizontal':
        center_x = x + w // 2
        cv2.rectangle(thresh, 
                     (center_x - cut_thickness//2, y), 
                     (center_x + cut_thickness//2, y + h), 
                     0, -1)
        cut_line = ((center_x, y), (center_x, y + h))
    else:
        center_y = y + h // 2
        cv2.rectangle(thresh, 
                     (x, center_y - cut_thickness//2), 
                     (x + w, center_y + cut_thickness//2), 
                     0, -1)
        cut_line = ((x, center_y), (x + w, center_y))
    
    return cut_line


def test_smart_cutting(image_path, output_dir='smart_cut_output', elongation_threshold=1.7, save_path=None):
    """
    Խելացի կտրում՝ միայն երկարավուն պատկերները
    
    Args:
        save_path: Լրիվ path որտեղ պահպանել 5_cuts_visualization.png-ը (օր՝ /home/gegham/Screenshots/num1.png)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ================================
    # ՔԱՅԼ 1: Կարդալ պատկերը
    # ================================
    print(f"📖 Reading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"✅ Image loaded: {img.shape}")
    
    # ================================
    # ՔԱՅԼ 2: Ստեղծել threshold
    # ================================
    print("🔧 Creating threshold...")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(os.path.join(output_dir, '1_original_thresh.png'), thresh)
    print(f"   💾 Saved: 1_original_thresh.png")
    
    # ================================
    # ՔԱՅԼ 3: Գտնել սկզբնական քառակուսիները
    # ================================
    print("🔍 Phase 1: Finding initial rectangles...")
    
    rectangles_before = find_all_rectangles(thresh, min_area=500)
    print(f"   ✅ Found {len(rectangles_before)} proper rectangles (4 sides)")
    
    img_rects_before = img.copy()
    for rect in rectangles_before:
        cv2.drawContours(img_rects_before, [rect['contour']], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, '2_rectangles_before.png'), img_rects_before)
    
    # ================================
    # ՔԱՅԼ 4: Գտնել երկարավուն պատկերները
    # ================================
    print(f"🔎 Phase 2: Finding ELONGATED shapes (aspect ratio >= {elongation_threshold})...")
    
    elongated_shapes, ignored_shapes = find_elongated_shapes(
        thresh, min_area=500, elongation_threshold=elongation_threshold
    )
    
    print(f"   ✂️  Found {len(elongated_shapes)} ELONGATED shapes (will be cut)")
    print(f"   ⏭️  Ignored {len(ignored_shapes)} square-ish shapes (will NOT be cut)")
    
    # Վիզուալիզացիա - երկարավուն պատկերներ (կարմիր)
    img_elongated = img.copy()
    for shape in elongated_shapes:
        cv2.drawContours(img_elongated, [shape['contour']], -1, (0, 0, 255), 2)
        x, y, w, h = shape['x'], shape['y'], shape['w'], shape['h']
        cv2.putText(img_elongated, f"AR:{shape['aspect_ratio']:.1f}", 
                   (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, '3_elongated_shapes.png'), img_elongated)
    
    # Վիզուալիզացիա - բաց թողած պատկերներ (կանաչ)
    img_ignored = img.copy()
    for shape in ignored_shapes:
        cv2.drawContours(img_ignored, [shape['contour']], -1, (0, 255, 0), 2)
        x, y, w, h = shape['x'], shape['y'], shape['w'], shape['h']
        cv2.putText(img_ignored, f"AR:{shape['aspect_ratio']:.1f}", 
                   (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, '4_ignored_shapes.png'), img_ignored)
    
    # ================================
    # ՔԱՅԼ 5: Կտրել ՄԻԱՅՆ երկարավուն պատկերները
    # ================================
    print("✂️  Phase 3: Cutting ONLY elongated shapes...")
    
    cleaned_thresh = thresh.copy()
    img_cuts = img.copy()
    cut_lines = []
    
    for i, shape in enumerate(elongated_shapes):
        cut_line = cut_shape_in_half(cleaned_thresh, shape, orientation='auto')
        cut_lines.append(cut_line)
        
        cv2.line(img_cuts, cut_line[0], cut_line[1], (255, 255, 255), 3)
        
        orientation = 'horizontal' if shape['w'] > shape['h'] else 'vertical'
        print(f"   ✂️  Cut #{i+1}: {shape['sides']} sides, "
              f"AR={shape['aspect_ratio']:.2f} → {orientation} cut")
    
    cv2.imwrite(os.path.join(output_dir, '5_cuts_visualization.png'), img_cuts)
    cv2.imwrite(os.path.join(output_dir, '6_thresh_after_cuts.png'), cleaned_thresh)
    
    # Եթե save_path տրված է, պահպանել այնտեղ էլ
    final_save_path = None
    if save_path:
        # Ստեղծել directory-ն եթե գոյություն չունի
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        cv2.imwrite(save_path, img_cuts)
        final_save_path = save_path
        print(f"   💾 Saved cuts visualization to: {save_path}")
    
    # ================================
    # ՔԱՅԼ 6: Նորից գտնել քառակուսիները
    # ================================
    print("🔍 Phase 4: Re-detecting rectangles after cuts...")
    
    rectangles_after = find_all_rectangles(cleaned_thresh, min_area=500)
    print(f"   ✅ Found {len(rectangles_after)} rectangles after cutting")
    print(f"   📊 Difference: +{len(rectangles_after) - len(rectangles_before)} new rectangles")
    
    img_rects_after = img.copy()
    for rect in rectangles_after:
        cv2.drawContours(img_rects_after, [rect['contour']], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, '7_rectangles_after.png'), img_rects_after)
    
    # ================================
    # ՔԱՅԼ 7: Համեմատություն
    # ================================
    comparison = np.hstack([thresh, cleaned_thresh])
    cv2.imwrite(os.path.join(output_dir, '8_comparison_before_after.png'), comparison)
    
    # ================================
    # Ամփոփում
    # ================================
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ԱՄՓՈՓՈՒՄ (Խելացի կտրում)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Սկզբնական քառակուսիներ: {len(rectangles_before)}\n")
        f.write(f"Երկարավուն պատկերներ (կտրված): {len(elongated_shapes)}\n")
        f.write(f"Քառակուսի պատկերներ (բաց թողած): {len(ignored_shapes)}\n")
        f.write(f"Վերջնական քառակուսիներ: {len(rectangles_after)}\n")
        f.write(f"Նոր քառակուսիներ: +{len(rectangles_after) - len(rectangles_before)}\n\n")
        
        f.write("Կտրված երկարավուն պատկերներ:\n")
        for i, shape in enumerate(elongated_shapes):
            f.write(f"  #{i+1}: {shape['sides']} կողմ, "
                   f"AR={shape['aspect_ratio']:.2f}, "
                   f"չափսեր={shape['w']}x{shape['h']}\n")
        
        f.write("\nԲաց թողած քառակուսի պատկերներ:\n")
        for i, shape in enumerate(ignored_shapes):
            f.write(f"  #{i+1}: {shape['sides']} կողմ, "
                   f"AR={shape['aspect_ratio']:.2f}, "
                   f"չափսեր={shape['w']}x{shape['h']}\n")
    
    print("\n" + "=" * 60)
    print("✅ ԱՎԱՐՏՎԱԾ!")
    print("=" * 60)
    print(f"📊 Սկզբնական քառակուսիներ: {len(rectangles_before)}")
    print(f"✂️  Կտրված (երկարավուն): {len(elongated_shapes)}")
    print(f"⏭️  Բաց թողած (քառակուսի): {len(ignored_shapes)}")
    print(f"✅ Վերջնական քառակուսիներ: {len(rectangles_after)}")
    print(f"🎯 Նոր քառակուսիներ: +{len(rectangles_after) - len(rectangles_before)}")
    print("=" * 60)
    
    # Վերադարձնել path-ը եթե save_path տրված է, հակառակ դեպքում img_cuts-ը
    if final_save_path:
        return final_save_path
    else:
        return img_cuts


# if __name__ == "__main__":
#     IMAGE_PATH = "/home/gegham/Screenshots/n5.png"
#     SAVE_PATH = "/home/gegham/Screenshots/num1.png"
    
#     result = test_smart_cutting(IMAGE_PATH, 
#                                 output_dir='smart_cut_output',
#                                 elongation_threshold=1.7,
#                                 save_path=SAVE_PATH)
    
#     if result is not None:
#         if isinstance(result, str):
#             print(f"\n✅ Պատկերը պահպանված է: {result}")
#         else:
#             print("\n✅ Վերադարձված է img_cuts պատկերը որպես numpy array")
#             print("   Օգտագործի result փոփոխականը հետագա մշակման համար")
    
#     print("\n📋 Ստուգիր հետևյալ ֆայլերը:")
#     print("   3_elongated_shapes.png   - երկարավուն (կարմիր)")
#     print("   4_ignored_shapes.png     - քառակուսի/բաց թողած (կանաչ)")
#     print("   5_cuts_visualization.png - կտրված տեղերը")
#     print("   7_rectangles_after.png   - վերջնական արդյունք")













































# # 2test.py
# import cv2
# import numpy as np
# import os


# def find_all_rectangles(thresh, min_area=500):
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     rectangles = []
#     for contour in contours:
#         perimeter = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
#         area = cv2.contourArea(contour)
        
#         if len(approx) == 4 and area > min_area:
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
            
#             if 0.5 < aspect_ratio < 2.0:
#                 rectangles.append({
#                     'contour': approx,
#                     'x': x, 'y': y, 'w': w, 'h': h,
#                     'area': area,
#                     'sides': 4
#                 })
    
#     return rectangles


# def find_elongated_shapes(thresh, min_area=500, elongation_threshold=1.7):
#     """
#     Գտնում է ՈՉ-քառակուսի պատկերները, բայց միայն երկարավուն (elongated) տարբերակները
    
#     elongation_threshold: Եթե w/h >= 1.7 կամ h/w >= 1.7 → երկարավուն է
#     """
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     elongated_shapes = []
#     ignored_shapes = []
    
#     for contour in contours:
#         perimeter = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#         area = cv2.contourArea(contour)
#         num_sides = len(approx)
        
#         if area > min_area and num_sides != 4:
#             x, y, w, h = cv2.boundingRect(contour)
#             aspect_ratio = max(w, h) / max(min(w, h), 1)  # Ամենից մեծը / ամենափոքրը
            
#             shape_info = {
#                 'contour': approx,
#                 'x': x, 'y': y, 'w': w, 'h': h,
#                 'area': area,
#                 'sides': num_sides,
#                 'aspect_ratio': aspect_ratio
#             }
            
#             # Ստուգում՝ արդյո՞ք երկարավուն է
#             if aspect_ratio >= elongation_threshold:
#                 elongated_shapes.append(shape_info)
#             else:
#                 ignored_shapes.append(shape_info)
    
#     return elongated_shapes, ignored_shapes


# def cut_shape_in_half(thresh, shape, orientation='auto'):
#     """
#     Կտրում է պատկերը կիսով
#     """
#     x, y, w, h = shape['x'], shape['y'], shape['w'], shape['h']
    
#     if orientation == 'auto':
#         orientation = 'horizontal' if w > h else 'vertical'
    
#     cut_thickness = max(3, int(min(w, h) * 0.15))
    
#     if orientation == 'horizontal':
#         center_x = x + w // 2
#         cv2.rectangle(thresh, 
#                      (center_x - cut_thickness//2, y), 
#                      (center_x + cut_thickness//2, y + h), 
#                      0, -1)
#         cut_line = ((center_x, y), (center_x, y + h))
#     else:
#         center_y = y + h // 2
#         cv2.rectangle(thresh, 
#                      (x, center_y - cut_thickness//2), 
#                      (x + w, center_y + cut_thickness//2), 
#                      0, -1)
#         cut_line = ((x, center_y), (x + w, center_y))
    
#     return cut_line


# def test_smart_cutting(image_path, output_dir='smart_cut_output', elongation_threshold=1.7):
#     """
#     Խելացի կտրում՝ միայն երկարավուն պատկերները
#     """
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # ================================
#     # ՔԱՅԼ 1: Կարդալ պատկերը
#     # ================================
#     print(f"📖 Reading image: {image_path}")
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"❌ Could not read image: {image_path}")
#         return
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(f"✅ Image loaded: {img.shape}")
    
#     # ================================
#     # ՔԱՅԼ 2: Ստեղծել threshold
#     # ================================
#     print("🔧 Creating threshold...")
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                     cv2.THRESH_BINARY_INV, 11, 2)
#     kernel = np.ones((3,3), np.uint8)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
#     cv2.imwrite(os.path.join(output_dir, '1_original_thresh.png'), thresh)
#     print(f"   💾 Saved: 1_original_thresh.png")
    
#     # ================================
#     # ՔԱՅԼ 3: Գտնել սկզբնական քառակուսիները
#     # ================================
#     print("🔍 Phase 1: Finding initial rectangles...")
    
#     rectangles_before = find_all_rectangles(thresh, min_area=500)
#     print(f"   ✅ Found {len(rectangles_before)} proper rectangles (4 sides)")
    
#     img_rects_before = img.copy()
#     for rect in rectangles_before:
#         cv2.drawContours(img_rects_before, [rect['contour']], -1, (0, 255, 0), 2)
#     cv2.imwrite(os.path.join(output_dir, '2_rectangles_before.png'), img_rects_before)
    
#     # ================================
#     # ՔԱՅԼ 4: Գտնել երկարավուն պատկերները
#     # ================================
#     print(f"🔎 Phase 2: Finding ELONGATED shapes (aspect ratio >= {elongation_threshold})...")
    
#     elongated_shapes, ignored_shapes = find_elongated_shapes(
#         thresh, min_area=500, elongation_threshold=elongation_threshold
#     )
    
#     print(f"   ✂️  Found {len(elongated_shapes)} ELONGATED shapes (will be cut)")
#     print(f"   ⏭️  Ignored {len(ignored_shapes)} square-ish shapes (will NOT be cut)")
    
#     # Վիզուալիզացիա - երկարավուն պատկերներ (կարմիր)
#     img_elongated = img.copy()
#     for shape in elongated_shapes:
#         cv2.drawContours(img_elongated, [shape['contour']], -1, (0, 0, 255), 2)
#         x, y, w, h = shape['x'], shape['y'], shape['w'], shape['h']
#         cv2.putText(img_elongated, f"AR:{shape['aspect_ratio']:.1f}", 
#                    (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     cv2.imwrite(os.path.join(output_dir, '3_elongated_shapes.png'), img_elongated)
    
#     # Վիզուալիզացիա - բաց թողած պատկերներ (կանաչ)
#     img_ignored = img.copy()
#     for shape in ignored_shapes:
#         cv2.drawContours(img_ignored, [shape['contour']], -1, (0, 255, 0), 2)
#         x, y, w, h = shape['x'], shape['y'], shape['w'], shape['h']
#         cv2.putText(img_ignored, f"AR:{shape['aspect_ratio']:.1f}", 
#                    (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     cv2.imwrite(os.path.join(output_dir, '4_ignored_shapes.png'), img_ignored)
    
#     # ================================
#     # ՔԱՅԼ 5: Կտրել ՄԻԱՅՆ երկարավուն պատկերները
#     # ================================
#     print("✂️  Phase 3: Cutting ONLY elongated shapes...")
    
#     cleaned_thresh = thresh.copy()
#     img_cuts = img.copy()
#     cut_lines = []
    
#     for i, shape in enumerate(elongated_shapes):
#         cut_line = cut_shape_in_half(cleaned_thresh, shape, orientation='auto')
#         cut_lines.append(cut_line)
        
#         cv2.line(img_cuts, cut_line[0], cut_line[1], (255, 255, 255), 3)
        
#         orientation = 'horizontal' if shape['w'] > shape['h'] else 'vertical'
#         print(f"   ✂️  Cut #{i+1}: {shape['sides']} sides, "
#               f"AR={shape['aspect_ratio']:.2f} → {orientation} cut")
    
#     cv2.imwrite(os.path.join(output_dir, '5_cuts_visualization.png'), img_cuts)
#     cv2.imwrite(os.path.join(output_dir, '6_thresh_after_cuts.png'), cleaned_thresh)
    
#     # ================================
#     # ՔԱՅԼ 6: Նորից գտնել քառակուսիները
#     # ================================
#     print("🔍 Phase 4: Re-detecting rectangles after cuts...")
    
#     rectangles_after = find_all_rectangles(cleaned_thresh, min_area=500)
#     print(f"   ✅ Found {len(rectangles_after)} rectangles after cutting")
#     print(f"   📊 Difference: +{len(rectangles_after) - len(rectangles_before)} new rectangles")
    
#     img_rects_after = img.copy()
#     for rect in rectangles_after:
#         cv2.drawContours(img_rects_after, [rect['contour']], -1, (0, 255, 0), 2)
#     cv2.imwrite(os.path.join(output_dir, '7_rectangles_after.png'), img_rects_after)
    
#     # ================================
#     # ՔԱՅԼ 7: Համեմատություն
#     # ================================
#     comparison = np.hstack([thresh, cleaned_thresh])
#     cv2.imwrite(os.path.join(output_dir, '8_comparison_before_after.png'), comparison)
    
#     # ================================
#     # Ամփոփում
#     # ================================
#     with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
#         f.write("=" * 60 + "\n")
#         f.write("ԱՄՓՈՓՈՒՄ (Խելացի կտրում)\n")
#         f.write("=" * 60 + "\n\n")
#         f.write(f"Սկզբնական քառակուսիներ: {len(rectangles_before)}\n")
#         f.write(f"Երկարավուն պատկերներ (կտրված): {len(elongated_shapes)}\n")
#         f.write(f"Քառակուսի պատկերներ (բաց թողած): {len(ignored_shapes)}\n")
#         f.write(f"Վերջնական քառակուսիներ: {len(rectangles_after)}\n")
#         f.write(f"Նոր քառակուսիներ: +{len(rectangles_after) - len(rectangles_before)}\n\n")
        
#         f.write("Կտրված երկարավուն պատկերներ:\n")
#         for i, shape in enumerate(elongated_shapes):
#             f.write(f"  #{i+1}: {shape['sides']} կողմ, "
#                    f"AR={shape['aspect_ratio']:.2f}, "
#                    f"չափսեր={shape['w']}x{shape['h']}\n")
        
#         f.write("\nԲաց թողած քառակուսի պատկերներ:\n")
#         for i, shape in enumerate(ignored_shapes):
#             f.write(f"  #{i+1}: {shape['sides']} կողմ, "
#                    f"AR={shape['aspect_ratio']:.2f}, "
#                    f"չափսեր={shape['w']}x{shape['h']}\n")
    
#     print("\n" + "=" * 60)
#     print("✅ ԱՎԱՐՏՎԱԾ!")
#     print("=" * 60)
#     print(f"📊 Սկզբնական քառակուսիներ: {len(rectangles_before)}")
#     print(f"✂️  Կտրված (երկարավուն): {len(elongated_shapes)}")
#     print(f"⏭️  Բաց թողած (քառակուսի): {len(ignored_shapes)}")
#     print(f"✅ Վերջնական քառակուսիներ: {len(rectangles_after)}")
#     print(f"🎯 Նոր քառակուսիներ: +{len(rectangles_after) - len(rectangles_before)}")
#     print("=" * 60)


# if __name__ == "__main__":
#     IMAGE_PATH = "/home/gegham/Screenshots/n5.png"
    
#     test_smart_cutting(IMAGE_PATH, 
#                        output_dir='smart_cut_output',
#                        elongation_threshold=1.7)
    
#     print("\n📋 Ստուգիր հետևյալ ֆայլերը:")
#     print("   3_elongated_shapes.png   - երկարավուն (կարմիր)")
#     print("   4_ignored_shapes.png     - քառակուսի/բաց թողած (կանաչ)")
#     print("   5_cuts_visualization.png - կտրված տեղերը")
#     print("   7_rectangles_after.png   - վերջնական արդյունք")