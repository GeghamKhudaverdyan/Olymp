# export_results.py
import json
import csv
from typing import List, Dict

def export_results_to_formats(rectangles: List[Dict], x_marks: List[Dict], blocks: List[List[Dict]], output_prefix: str = "result"):
    """
    Արդյունքները export անում է մի քանի ֆորմատներով՝ հեշտ համեմատության համար
    """
    
    # 1. Simple Text Format - ամենահարմարը համեմատելու համար
    with open(f"{output_prefix}_simple.txt", "w", encoding="utf-8") as f:
        f.write("# Matrix Results (1=X found, 0=No X)\n")
        f.write("# Format: position value\n\n")
        
        for block_idx, block in enumerate(blocks):
            f.write(f"Block {block_idx + 1}:\n")
            
            # Խմբավորում ըստ տողերի
            rows = {}
            for rect in block:
                row = rect.get('row', 0)
                if row not in rows:
                    rows[row] = []
                rows[row].append(rect)
            
            # Տեսակավորում
            for row_num in sorted(rows.keys()):
                sorted_row = sorted(rows[row_num], key=lambda r: r.get('col', 0))
                for rect in sorted_row:
                    label = rect.get('matrix_label', '?')
                    value = 1 if rect.get('has_x', False) else 0
                    f.write(f"{label} {value}\n")
            f.write("\n")
    
    # 2. Compact Format - մեկ տողում բոլոր արդյունքները
    with open(f"{output_prefix}_compact.txt", "w", encoding="utf-8") as f:
        for block_idx, block in enumerate(blocks):
            f.write(f"Block{block_idx + 1}: ")
            
            sorted_block = sorted(block, key=lambda r: (r.get('row', 0), r.get('col', 0)))
            results = [f"{r.get('matrix_label', '?')}:{1 if r.get('has_x', False) else 0}" 
                      for r in sorted_block]
            f.write(" ".join(results))
            f.write("\n")
    
    # 3. CSV Format - Excel-ում բացելու համար
    with open(f"{output_prefix}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Block", "Position", "Has_X", "Row", "Col"])
        
        for block_idx, block in enumerate(blocks):
            sorted_block = sorted(block, key=lambda r: (r.get('row', 0), r.get('col', 0)))
            for rect in sorted_block:
                writer.writerow([
                    block_idx + 1,
                    rect.get('matrix_label', '?'),
                    1 if rect.get('has_x', False) else 0,
                    rect.get('row', 0),
                    rect.get('col', 0)
                ])
    
    # 4. JSON Format - ծրագրային համեմատության համար
    json_data = {
        "blocks": []
    }
    
    for block_idx, block in enumerate(blocks):
        block_data = {
            "block_id": block_idx + 1,
            "cells": []
        }
        
        sorted_block = sorted(block, key=lambda r: (r.get('row', 0), r.get('col', 0)))
        for rect in sorted_block:
            block_data["cells"].append({
                "position": rect.get('matrix_label', '?'),
                "has_x": 1 if rect.get('has_x', False) else 0,
                "row": rect.get('row', 0),
                "col": rect.get('col', 0)
            })
        
        json_data["blocks"].append(block_data)
    
    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 5. Grid View - վիզուալ ներկայացում
    with open(f"{output_prefix}_grid.txt", "w", encoding="utf-8") as f:
        for block_idx, block in enumerate(blocks):
            f.write(f"Block {block_idx + 1}:\n")
            
            rows = {}
            for rect in block:
                row = rect.get('row', 0)
                if row not in rows:
                    rows[row] = []
                rows[row].append(rect)
            
            for row_num in sorted(rows.keys()):
                sorted_row = sorted(rows[row_num], key=lambda r: r.get('col', 0))
                row_str = " ".join([str(1 if r.get('has_x', False) else 0) for r in sorted_row])
                f.write(f"  Row {row_num}: {row_str}\n")
            f.write("\n")
    
    print(f"✅ Results exported:")
    print(f"   - {output_prefix}_simple.txt (best for comparison)")
    print(f"   - {output_prefix}_compact.txt (one-line format)")
    print(f"   - {output_prefix}.csv (Excel compatible)")
    print(f"   - {output_prefix}.json (programmatic)")
    print(f"   - {output_prefix}_grid.txt (visual grid)")