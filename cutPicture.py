import cv2
import numpy as np

def detect_grids(img, gray):

    print("✅✅ detect_grids ֆունկցիան կանչվեց")

    height, width = gray.shape
    
    # Blur և threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Հորիզոնական projection (տողերի հայտնաբերում)
    horizontal_projection = np.sum(binary, axis=1)
    
    # Ուղղահայաց projection (սյուների հայտնաբերում)  
    vertical_projection = np.sum(binary, axis=0)
    
    # Գտնել դատարկ տարածքները (բաժանարարներ)
    h_threshold = np.mean(horizontal_projection) * 0.3
    v_threshold = np.mean(vertical_projection) * 0.3
    
    # Գտնել հորիզոնական բաժանումները
    h_splits = []
    in_gap = False
    gap_start = 0
    min_gap_size = height * 0.02
    
    for i, val in enumerate(horizontal_projection):
        if val < h_threshold and not in_gap:
            gap_start = i
            in_gap = True
        elif val >= h_threshold and in_gap:
            if i - gap_start > min_gap_size:
                h_splits.append((gap_start + i) // 2)
            in_gap = False
    
    # Գտնել ուղղահայաց բաժանումները
    v_splits = []
    in_gap = False
    gap_start = 0
    min_gap_size = width * 0.02
    
    for i, val in enumerate(vertical_projection):
        if val < v_threshold and not in_gap:
            gap_start = i
            in_gap = True
        elif val >= v_threshold and in_gap:
            if i - gap_start > min_gap_size:
                v_splits.append((gap_start + i) // 2)
            in_gap = False
    
    # Ստեղծել grid-երի ցուցակը
    h_splits = [0] + h_splits + [height]
    v_splits = [0] + v_splits + [width]
    
    grids = []
    for i in range(len(h_splits) - 1):
        for j in range(len(v_splits) - 1):
            y1, y2 = h_splits[i], h_splits[i + 1]
            x1, x2 = v_splits[j], v_splits[j + 1]
            
            # Զտել շատ փոքր տարածքները
            if (y2 - y1) > height * 0.05 and (x2 - x1) > width * 0.05:
                grids.append({
                    'x': x1, 'y': y1,
                    'w': x2 - x1, 'h': y2 - y1,
                    'region': (x1, y1, x2, y2)
                })
    
    # Եթե չգտավ grid-եր, վերադարձնել ամբողջ նկարը
    if len(grids) == 0:
        grids.append({
            'x': 0, 'y': 0,
            'w': width, 'h': height,
            'region': (0, 0, width, height)
        })
    
    return grids