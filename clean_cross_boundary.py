import cv2
import numpy as np
import os


def remove_cross_boundary_lines(thresh, rectangles, debug=False, debug_dir='debug'):
    """
    Հեռացնում է գծերը, որոնք հատում են քառակուսիների սահմանները։
    
    Տարբերակ 3: Connected Components Filtering
    
    Args:
        thresh: Binary threshold պատկեր (255=սպիտակ գծեր, 0=սև ֆոն)
        rectangles: Քառակուսիների list (յուրաքանչյուրը dict է contour-ով)
        debug: Եթե True, պահպանում է debug պատկերներ
        debug_dir: Debug պատկերների պանակ
    
    Returns:
        cleaned_thresh: Մաքրված threshold պատկեր
    """
    
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
    
    # 1. Ստեղծում ենք mask-եր յուրաքանչյուր քառակուսու համար
    h, w = thresh.shape
    individual_masks = []
    
    for idx, rect in enumerate(rectangles):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [rect['contour']], -1, 255, -1)
        individual_masks.append({
            'mask': mask,
            'rect_idx': idx,
            'rect': rect
        })
    
    if debug:
        # Պահպանում ենք բոլոր mask-երի համադրումը
        all_masks = np.zeros((h, w), dtype=np.uint8)
        for item in individual_masks:
            all_masks = cv2.bitwise_or(all_masks, item['mask'])
        cv2.imwrite(os.path.join(debug_dir, 'all_rect_masks.png'), all_masks)
    
    # 2. Գտնում ենք բոլոր connected components-ները thresh-ում
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )
    
    # 3. Ստեղծում ենք մաքրված պատկեր
    cleaned_thresh = thresh.copy()
    
    components_to_remove = []
    
    # 4. Ստուգում ենք յուրաքանչյուր component
    for label_id in range(1, num_labels):  # 0-ը background է
        # Ստեղծում ենք mask այս component-ի համար
        component_mask = (labels == label_id).astype(np.uint8) * 255
        
        # Հաշվում ենք քանի քառակուսու հետ է հատվում
        intersecting_rects = []
        
        for item in individual_masks:
            # Ստուգում ենք overlap
            overlap = cv2.bitwise_and(component_mask, item['mask'])
            overlap_pixels = np.count_nonzero(overlap)
            
            if overlap_pixels > 0:
                intersecting_rects.append(item['rect_idx'])
        
        # 5. Եթե component-ը հատում է 2+ քառակուսի → ջնջել
        if len(intersecting_rects) >= 2:
            components_to_remove.append(label_id)
            
            # Ջնջում ենք այս component-ը
            cleaned_thresh[labels == label_id] = 0
            
            if debug:
                print(f"   🗑️  Removed component {label_id}: crosses {len(intersecting_rects)} rectangles "
                      f"(indices: {intersecting_rects})")
    
    if debug:
        # Պահպանում ենք removed components-ները
        removed_mask = np.zeros((h, w), dtype=np.uint8)
        for label_id in components_to_remove:
            removed_mask[labels == label_id] = 255
        
        cv2.imwrite(os.path.join(debug_dir, 'removed_components.png'), removed_mask)
        cv2.imwrite(os.path.join(debug_dir, 'cleaned_thresh.png'), cleaned_thresh)
        
        print(f"✅ Cleaned {len(components_to_remove)} cross-boundary components")
    
    return cleaned_thresh


def remove_cross_boundary_lines_aggressive(thresh, rectangles, border_width=10, 
                                           debug=False, debug_dir='debug'):
    """
    Ավելի aggressive տարբերակ՝ հեռացնում է նաև եզրերին մոտ գծերը։
    
    Համադրում է:
    - Connected Components Filtering (Տարբերակ 3)
    - Border Cleaning (Տարբերակ 1)
    
    Args:
        thresh: Binary threshold պատկեր
        rectangles: Քառակուսիների list
        border_width: Քանի պիքսել հեռացնել եզրերից (default: 10)
        debug: Debug ռեժիմ
        debug_dir: Debug պատկերների պանակ
    
    Returns:
        cleaned_thresh: Մաքրված threshold պատկեր
    """
    
    # Քայլ 1: Connected Components Filtering
    cleaned = remove_cross_boundary_lines(thresh, rectangles, debug=debug, debug_dir=debug_dir)
    
    # Քայլ 2: Border Cleaning
    h, w = cleaned.shape
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    for rect in rectangles:
        # Ստեղծում ենք mask
        rect_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(rect_mask, [rect['contour']], -1, 255, -1)
        
        # Erode անում ենք (փոքրացնում դեպի կենտրոն)
        kernel = np.ones((border_width, border_width), np.uint8)
        eroded_mask = cv2.erode(rect_mask, kernel, iterations=1)
        
        # Ավելացնում ենք վերջնական mask-ին
        final_mask = cv2.bitwise_or(final_mask, eroded_mask)
    
    # Կիրառում ենք երկու mask-երը միաժամանակ
    result = cv2.bitwise_and(cleaned, final_mask)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, 'final_cleaned_aggressive.png'), result)
        print(f"✅ Applied aggressive border cleaning (border_width={border_width}px)")
    
    return result