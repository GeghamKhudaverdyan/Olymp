import cv2
import numpy as np
import os

def intersect_lines(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)


def detect_x_in_roi(roi, rect_idx=0, debug=False, debug_dir='debug'):

    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return False
    
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Ստուգում ենք լցվածության գործակիցը
    black_pixels = np.sum(roi == 255)
    roi_area = roi.shape[0] * roi.shape[1]
    fill_ratio = black_pixels / roi_area if roi_area > 0 else 0
    
    has_x = False
    
    # Արագ ֆիլտր՝ լցվածության գործակիցով
    if 0.06 < fill_ratio < 0.55:
        # Նախապատրաստում՝ փակում + ընդլայնում
        kernel_small = np.ones((3,3), np.uint8)
        roi_proc = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        iter_dilate = 1 if min(roi.shape) < 50 else 2
        roi_proc = cv2.dilate(roi_proc, kernel_small, iterations=iter_dilate)
        
        # Եզրերի հայտնաբերում Canny-ով
        edges = cv2.Canny(roi_proc, 
                         max(20, int(min(roi.shape)*0.03)), 
                         max(80, int(min(roi.shape)*0.12)))
        
        if debug:
            cv2.imwrite(os.path.join(debug_dir, f"roi_{rect_idx}.png"), roi)
            cv2.imwrite(os.path.join(debug_dir, f"roi_proc_{rect_idx}.png"), roi_proc)
            cv2.imwrite(os.path.join(debug_dir, f"edges_{rect_idx}.png"), edges)
        
        # Hough պարամետրեր
        short_side = min(roi.shape[:2])
        hough_thresh = max(6, int(short_side * 0.10))
        minLineLen = max(6, int(short_side * 0.20))
        maxGap = max(4, int(short_side * 0.12))
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=hough_thresh,
                               minLineLength=minLineLen, 
                               maxLineGap=maxGap)
        
        if lines is not None and len(lines) >= 2:
            # Ընտրում ենք երկար գծերը
            lines_long = []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                length = np.hypot(x2-x1, y2-y1)
                angle = np.degrees(np.arctan2((y2-y1), (x2-x1)))
                if angle < 0:
                    angle += 180
                if length > min(roi.shape[:2]) * 0.25:
                    lines_long.append((x1, y1, x2, y2, length, angle))
            
            # Եթե քիչ են, թուլացնում ենք պահանջը
            if len(lines_long) < 2:
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    length = np.hypot(x2-x1, y2-y1)
                    angle = np.degrees(np.arctan2((y2-y1), (x2-x1)))
                    if angle < 0:
                        angle += 180
                    if length > min(roi.shape[:2]) * 0.18:
                        lines_long.append((x1, y1, x2, y2, length, angle))
            
            # Սորտավորում ենք երկարության պրակ
            lines_long.sort(key=lambda t: t[4], reverse=True)
            
            # Կենտրոնի կոորդինատներ
            center_x = roi.shape[1] / 2.0
            center_y = roi.shape[0] / 2.0
            center_thresh = min(roi.shape) / 2.0
            
            found_pair = None
            
            # Ստրատեգիա A: գծեր ~45° և ~135° անկյուններով
            top_k = lines_long[:8]
            for i in range(len(top_k)):
                for j in range(i+1, len(top_k)):
                    a = top_k[i]
                    b = top_k[j]
                    ang1 = a[5]
                    ang2 = b[5]
                    
                    cond1 = (20 <= ang1 <= 70 and 110 <= ang2 <= 160)
                    cond2 = (20 <= ang2 <= 70 and 110 <= ang1 <= 160)
                    
                    if cond1 or cond2:
                        pt = intersect_lines((a[0], a[1], a[2], a[3]), 
                                            (b[0], b[1], b[2], b[3]))
                        if pt is not None:
                            distc = np.hypot(pt[0] - center_x, pt[1] - center_y)
                            if distc <= center_thresh:
                                found_pair = (a, b, pt)
                                break
                if found_pair:
                    break
            
            # Ստրատեգիա B: անկյունների տարբերություն
            if found_pair is None:
                L = len(lines_long)
                for i in range(L):
                    for j in range(i+1, L):
                        a = lines_long[i]
                        b = lines_long[j]
                        ang1 = a[5]
                        ang2 = b[5]
                        diff = abs(ang1 - ang2)
                        diff_wrap = min(diff, 180 - diff)
                        
                        if 40 <= diff_wrap <= 140:
                            pt = intersect_lines((a[0], a[1], a[2], a[3]), 
                                                (b[0], b[1], b[2], b[3]))
                            if pt is not None:
                                distc = np.hypot(pt[0] - center_x, pt[1] - center_y)
                                if distc <= center_thresh:
                                    found_pair = (a, b, pt)
                                    break
                    if found_pair:
                        break
            
            if found_pair is not None:
                has_x = True
                
                # Debug պատկեր
                if debug:
                    debug_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    a, b, pt = found_pair
                    cv2.line(debug_rgb, (int(a[0]), int(a[1])), 
                            (int(a[2]), int(a[3])), (0, 255, 0), 2)
                    cv2.line(debug_rgb, (int(b[0]), int(b[1])), 
                            (int(b[2]), int(b[3])), (0, 255, 0), 2)
                    cv2.circle(debug_rgb, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(debug_dir, f"debug_lines_{rect_idx}.png"), 
                               debug_rgb)
    
    return has_x


# Օգտագործման օրինակ:
if __name__ == "__main__":
    # Ստեղծում ենք թեստային պատկեր X-ով
    test_roi = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(test_roi, (20, 20), (80, 80), 255, 3)
    cv2.line(test_roi, (80, 20), (20, 80), 255, 3)
    
    result = detect_x_in_roi(test_roi, debug=True)
    print(f"X հայտնաբերվել է: {result}")