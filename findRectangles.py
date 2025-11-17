import cv2
import numpy as np
import os
from x_detector import detect_x_in_roi


def find_rectangles_and_x(img, gray, output_path='output.jpg', min_area=1000, max_overlap=0.3,
                          save_crops=False, crops_dir='crops', save_only_with_x=False, save_thresh=False,
                          debug=False, debug_dir='debug'):
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    h, w = img.shape[:2]
    min_short_side = min(h, w)
    target_short = 600 
    if min_short_side < target_short:
        scale = target_short / min_short_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    dyn_min_area = int( (img.shape[0] * img.shape[1]) * 0.0004 )
    min_area = min_area if min_area > dyn_min_area else dyn_min_area

    original = img.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        area = cv2.contourArea(contour)
        if 4 <= len(approx) <= 8 and area > min_area:
            x, y, w_r, h_r = cv2.boundingRect(approx)
            aspect_ratio = float(w_r) / h_r
            if 0.5 < aspect_ratio < 2.0:
                rectangles.append({
                    'contour': approx,
                    'area': area,
                    'x': x, 'y': y, 'w': w_r, 'h': h_r,
                    'aspect_ratio': aspect_ratio,
                    'center': (x + w_r//2, y + h_r//2),
                    'has_x': False
                })

    def boxes_overlap(box1, box2, threshold=0.3):
        x1, y1, w1, h1 = box1['x'], box1['y'], box1['w'], box1['h']
        x2, y2, w2, h2 = box2['x'], box2['y'], box2['w'], box2['h']
        x_left = max(x1, x2); y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2); y_bottom = min(y1 + h1, y2 + h2)
        if x_right < x_left or y_bottom < y_top:
            return False
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        overlap1 = intersection_area / (w1 * h1)
        overlap2 = intersection_area / (w2 * h2)
        return max(overlap1, overlap2) > threshold

    filtered_rectangles = []
    rectangles.sort(key=lambda r: r['area'], reverse=True)
    for rect in rectangles:
        should_add = True
        for existing in filtered_rectangles:
            if boxes_overlap(rect, existing, max_overlap):
                should_add = False
                break
        if should_add:
            filtered_rectangles.append(rect)
    
    GAP_THRESHOLD = 8
    
    sorted_rects = sorted(filtered_rectangles, key=lambda r: (r['y'], r['x']))
    
    blocks = []
    used = set()
    
    for i, rect in enumerate(sorted_rects):
        if i in used:
            continue
            
        current_block = [rect]
        used.add(i)
        
        queue = [rect]
        while queue:
            current = queue.pop(0)
            
            for j, other in enumerate(sorted_rects):
                if j in used:
                    continue
                
                cx, cy = current['center']
                ox, oy = other['center']
                
                h_dist = abs(ox - cx) - (current['w'] + other['w']) / 2
                v_dist = abs(oy - cy) - (current['h'] + other['h']) / 2
                
                if (h_dist <= GAP_THRESHOLD and abs(oy - cy) < current['h'] * 1.5) or \
                   (v_dist <= GAP_THRESHOLD and abs(ox - cx) < current['w'] * 1.5):
                    current_block.append(other)
                    used.add(j)
                    queue.append(other)
        
        if current_block:
            blocks.append(current_block)
    
    all_rectangles_with_labels = []
    
    for block_idx, block in enumerate(blocks):
        block_sorted = sorted(block, key=lambda r: r['y'])
        
        if not block_sorted:
            continue
            
        median_height = np.median([r['h'] for r in block_sorted])
        tolerance = median_height * 0.5
        
        rows = []
        for rect in block_sorted:
            placed = False
            for row in rows:
                if abs(rect['y'] - row[0]['y']) < tolerance:
                    row.append(rect)
                    placed = True
                    break
            if not placed:
                rows.append([rect])
        
        for row_idx, row in enumerate(rows):
            row.sort(key=lambda r: r['x'])
            for col_idx, rect in enumerate(row):
                rect['block_id'] = block_idx + 1
                rect['matrix_label'] = f"{row_idx + 1}.{col_idx + 1}"
                rect['row'] = row_idx + 1
                rect['col'] = col_idx + 1
                all_rectangles_with_labels.append(rect)
    
    for rect_idx, rect in enumerate(all_rectangles_with_labels):
        x, y, w_r, h_r = rect['x'], rect['y'], rect['w'], rect['h']
        margin = max(5, int(min(w_r, h_r) * 0.18))
        roi = thresh[y+margin:y+h_r-margin, x+margin:x+w_r-margin]
        
        has_x = detect_x_in_roi(roi, rect_idx, debug=debug, debug_dir=debug_dir)
        rect['has_x'] = has_x

    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    x_marks = [r for r in all_rectangles_with_labels if r['has_x']]

    for rect in all_rectangles_with_labels:
        if rect['has_x']:
            cv2.drawContours(img, [rect['contour']], -1, (0, 0, 255), 3)
        else:
            cv2.drawContours(img, [rect['contour']], -1, (0, 255, 0), 2)

        x, y = rect['x'], rect['y']
        
        label_text = rect.get('matrix_label', '?')
        label_bg_color = (0, 0, 255) if rect['has_x'] else (0, 255, 0)
        
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_w = text_size[0] + 12
        label_h = text_size[1] + 12
        
        cv2.rectangle(img, (x+2, y+2), (x+label_w, y+label_h), label_bg_color, -1)
        cv2.putText(img, label_text, (x+6, y+label_h-6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        if save_crops:
            w_r, h_r = rect['w'], rect['h']
            margin = max(5, int(min(w_r, h_r) * 0.18))
            x1 = max(0, x + margin); y1 = max(0, y + margin)
            x2 = min(original.shape[1], x + w_r - margin)
            y2 = min(original.shape[0], y + h_r - margin)
            crop_color = original[y1:y2, x1:x2]
            
            if crop_color.size != 0:
                if (not save_only_with_x) or (save_only_with_x and rect['has_x']):
                    tag = 'X' if rect['has_x'] else 'noX'
                    block_id = rect.get('block_id', 0)
                    matrix_label = rect.get('matrix_label', '0.0')
                    fname = os.path.join(crops_dir, 
                                       f"block{block_id}_{matrix_label}_{tag}.png")
                    cv2.imwrite(fname, crop_color)
                    
                    if save_thresh:
                        crop_thresh = thresh[y1:y2, x1:x2]
                        fname_t = os.path.join(crops_dir, 
                                             f"block{block_id}_{matrix_label}_{tag}_thresh.png")
                        cv2.imwrite(fname_t, crop_thresh)

    cv2.imwrite(output_path, img)
    
    return all_rectangles_with_labels, x_marks, img, blocks














# import cv2
# import numpy as np
# import os
# from x_detector import detect_x_in_roi


# def find_rectangles_and_x(img, gray, output_path='output.jpg', min_area=1000, max_overlap=0.3,
#                           save_crops=False, crops_dir='crops', save_only_with_x=False, save_thresh=False,
#                           debug=False, debug_dir='debug'):
#     if debug:
#         os.makedirs(debug_dir, exist_ok=True)

#     h, w = img.shape[:2]
#     min_short_side = min(h, w)
#     target_short = 600 
#     if min_short_side < target_short:
#         scale = target_short / min_short_side
#         img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#         gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     gray = clahe.apply(gray)

#     dyn_min_area = int( (img.shape[0] * img.shape[1]) * 0.0004 )
#     min_area = min_area if min_area > dyn_min_area else dyn_min_area

#     original = img.copy()
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                     cv2.THRESH_BINARY_INV, 11, 2)
#     kernel = np.ones((3,3), np.uint8)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     rectangles = []
#     for contour in contours:
#         perimeter = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
#         area = cv2.contourArea(contour)
#         if 4 <= len(approx) <= 8 and area > min_area:
#             x, y, w_r, h_r = cv2.boundingRect(approx)
#             aspect_ratio = float(w_r) / h_r
#             if 0.5 < aspect_ratio < 2.0:
#                 rectangles.append({
#                     'contour': approx,
#                     'area': area,
#                     'x': x, 'y': y, 'w': w_r, 'h': h_r,
#                     'aspect_ratio': aspect_ratio,
#                     'center': (x + w_r//2, y + h_r//2),
#                     'has_x': False
#                 })

#     def boxes_overlap(box1, box2, threshold=0.3):
#         x1, y1, w1, h1 = box1['x'], box1['y'], box1['w'], box1['h']
#         x2, y2, w2, h2 = box2['x'], box2['y'], box2['w'], box2['h']
#         x_left = max(x1, x2); y_top = max(y1, y2)
#         x_right = min(x1 + w1, x2 + w2); y_bottom = min(y1 + h1, y2 + h2)
#         if x_right < x_left or y_bottom < y_top:
#             return False
#         intersection_area = (x_right - x_left) * (y_bottom - y_top)
#         overlap1 = intersection_area / (w1 * h1)
#         overlap2 = intersection_area / (w2 * h2)
#         return max(overlap1, overlap2) > threshold

#     filtered_rectangles = []
#     rectangles.sort(key=lambda r: r['area'], reverse=True)
#     for rect in rectangles:
#         should_add = True
#         for existing in filtered_rectangles:
#             if boxes_overlap(rect, existing, max_overlap):
#                 should_add = False
#                 break
#         if should_add:
#             filtered_rectangles.append(rect)
#     # Ավելի ճիշտ տողերի խմբավորում
#     median_height = np.median([r['h'] for r in filtered_rectangles])
#     tolerance = median_height * 0.4  # 40% հանդուրժողականություն

#     # Տողերով խմբավորում
#     rows = []
#     sorted_by_y = sorted(filtered_rectangles, key=lambda r: r['y'])

#     for rect in sorted_by_y:
#         placed = False
#         for row in rows:
#             # Եթե y դիրքերը մոտ են միմյանց՝ նույն տողն են
#             if abs(rect['y'] - row[0]['y']) < tolerance:
#                 row.append(rect)
#                 placed = True
#                 break
#         if not placed:
#             rows.append([rect])

#     # Յուրաքանչյուր տողում ձախից-աջ սորտավորում
#     filtered_rectangles = []
#     for row in rows:
#         row.sort(key=lambda r: r['x'])
#         filtered_rectangles.extend(row)

#     # helper: line intersection
#     def intersect_lines(a, b):
#         # a, b are (x1,y1,x2,y2)
#         x1,y1,x2,y2 = a; x3,y3,x4,y4 = b
#         denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
#         if abs(denom) < 1e-6:
#             return None
#         px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
#         py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
#         return (px, py)

#     for rect_idx, rect in enumerate(filtered_rectangles):
#         x, y, w_r, h_r = rect['x'], rect['y'], rect['w'], rect['h']
#         margin = max(5, int(min(w_r, h_r) * 0.18))
#         roi = thresh[y+margin:y+h_r-margin, x+margin:x+w_r-margin]
        
#         # Հայտնաբերում ենք X-ը
#         has_x = detect_x_in_roi(roi, rect_idx, debug=debug, debug_dir=debug_dir)
#         rect['has_x'] = has_x

#     # saving crops and annotated image (same as before)...
#     if save_crops:
#         os.makedirs(crops_dir, exist_ok=True)

#     x_marks = [r for r in filtered_rectangles if r['has_x']]

#     for i, rect in enumerate(filtered_rectangles):
#         if rect['has_x']:
#             cv2.drawContours(img, [rect['contour']], -0, (0, 0, 255), 3)
#         else:
#             cv2.drawContours(img, [rect['contour']], -0, (0, 255, 0), 2)

#         x, y = rect['x'], rect['y']
#         label_bg_color = (0, 0, 255) if rect['has_x'] else (0, 255, 0)
#         cv2.rectangle(img, (x+2, y+2), (x+35, y+25), label_bg_color, -1)
#         cv2.putText(img, f"{i+1}", (x+6, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

#         if save_crops:
#             w_r, h_r = rect['w'], rect['h']
#             margin = max(5, int(min(w_r, h_r) * 0.18))
#             x1 = max(0, x + margin); y1 = max(0, y + margin)
#             x2 = min(original.shape[1], x + w_r - margin); y2 = min(original.shape[0], y + h_r - margin)
#             crop_color = original[y1:y2, x1:x2]
#             if crop_color.size != 0:
#                 if (not save_only_with_x) or (save_only_with_x and rect['has_x']):
#                     tag = 'X' if rect['has_x'] else 'noX'
#                     fname = os.path.join(crops_dir, f"crop_{i+1:03d}_{tag}.png")
#                     cv2.imwrite(fname, crop_color)
#                     if save_thresh:
#                         crop_thresh = thresh[y1:y2, x1:x2]
#                         fname_t = os.path.join(crops_dir, f"crop_{i+1:03d}_{tag}_thresh.png")
#                         cv2.imwrite(fname_t, crop_thresh)

#     cv2.imwrite(output_path, img)
#     return filtered_rectangles, x_marks, img