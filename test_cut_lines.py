# import cv2
# import json
# import os
# import hashlib
# import numpy as np
# from typing import Dict, List, Tuple, Optional
# from datetime import datetime
# from dataclasses import dataclass, asdict
# from enum import Enum
# import logging
# from pathlib import Path

# class ValidationLevel(Enum):
#     """Validation մակարդակ"""
#     STRICT = "strict"      # Ամենաշատ ստուգումներ
#     NORMAL = "normal"      # Ստանդարտ
#     RELAXED = "relaxed"    # Քիչ ստուգումներ (ռիսկային)


# class ConfidenceThreshold(Enum):
#     """X հայտնաբերման confidence շեմեր"""
#     HIGH = 0.85      # Շատ վստահելի
#     MEDIUM = 0.70    # Նորմալ
#     LOW = 0.55       # Ցածր - պետք է manual review


# @dataclass
# class DetectionResult:
#     """Մեկ քառակուսու հայտնաբերման արդյունք"""
#     block_id: int
#     matrix_label: str
#     row: int
#     col: int
#     has_x: bool
#     confidence: float
#     needs_review: bool
#     x: int
#     y: int
#     w: int
#     h: int


# @dataclass
# class ValidationResult:
#     """Validation արդյունք"""
#     is_valid: bool
#     warnings: List[str]
#     errors: List[str]
#     confidence_score: float


# # ============================================================================
# # LOGGING SETUP
# # ============================================================================

# class GradingLogger:
#     """Մանրամասն logging համակարգ"""
    
#     def __init__(self, log_dir: str = "logs"):
#         self.log_dir = Path(log_dir)
#         self.log_dir.mkdir(exist_ok=True)
        
#         # Setup logging
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         log_file = self.log_dir / f"grading_{timestamp}.log"
        
#         logging.basicConfig(
#             level=logging.DEBUG,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler(log_file, encoding='utf-8'),
#                 logging.StreamHandler()
#             ]
#         )
        
#         self.logger = logging.getLogger('BulletproofGrading')
#         self.logger.info("="*70)
#         self.logger.info("🛡️ Bulletproof Grading System initialized")
#         self.logger.info("="*70)
    
#     def log_step(self, step: str, details: str):
#         """Log քայլ"""
#         self.logger.info(f"📍 STEP: {step}")
#         self.logger.info(f"   {details}")
    
#     def log_warning(self, warning: str):
#         """Log warning"""
#         self.logger.warning(f"⚠️  WARNING: {warning}")
    
#     def log_error(self, error: str):
#         """Log error"""
#         self.logger.error(f"❌ ERROR: {error}")
    
#     def log_validation(self, result: ValidationResult):
#         """Log validation result"""
#         self.logger.info(f"🔍 VALIDATION RESULT:")
#         self.logger.info(f"   Valid: {result.is_valid}")
#         self.logger.info(f"   Confidence: {result.confidence_score:.2%}")
#         if result.warnings:
#             self.logger.warning(f"   Warnings: {len(result.warnings)}")
#             for w in result.warnings:
#                 self.logger.warning(f"      - {w}")
#         if result.errors:
#             self.logger.error(f"   Errors: {len(result.errors)}")
#             for e in result.errors:
#                 self.logger.error(f"      - {e}")


# # ============================================================================
# # X DETECTION WITH CONFIDENCE SCORING
# # ============================================================================

# class ConfidentXDetector:
#     """X հայտնաբերում confidence scoring-ով"""
    
#     def __init__(self, threshold: ConfidenceThreshold = ConfidenceThreshold.MEDIUM):
#         self.threshold = threshold.value
    
#     def detect_x_with_confidence(self, roi: np.ndarray, 
#                                  cell_id: str,
#                                  debug: bool = False,
#                                  debug_dir: str = None) -> Tuple[bool, float]:
#         """
#         Հայտնաբերում է X-ը և վերադարձնում confidence score
        
#         Returns:
#             (has_x, confidence_score)
#         """
        
#         if roi.size == 0:
#             return False, 0.0
        
#         # Multi-method approach
#         scores = []
        
#         # Method 1: Line detection (Hough)
#         score1 = self._detect_by_lines(roi)
#         scores.append(score1)
        
#         # Method 2: Diagonal pixel density
#         score2 = self._detect_by_diagonals(roi)
#         scores.append(score2)
        
#         # Method 3: Corner detection
#         score3 = self._detect_by_corners(roi)
#         scores.append(score3)
        
#         # Method 4: Template matching
#         score4 = self._detect_by_template(roi)
#         scores.append(score4)
        
#         # Weighted average
#         weights = [0.35, 0.30, 0.20, 0.15]
#         confidence = sum(s * w for s, w in zip(scores, weights))
        
#         # Debug
#         if debug and debug_dir:
#             self._save_debug_info(roi, cell_id, scores, confidence, debug_dir)
        
#         has_x = confidence >= self.threshold
        
#         return has_x, confidence
    
#     def _detect_by_lines(self, roi: np.ndarray) -> float:
#         """Հայտնաբերում Hough line transform-ով"""
#         try:
#             edges = cv2.Canny(roi, 50, 150)
#             lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
#                                    threshold=int(min(roi.shape) * 0.3),
#                                    minLineLength=int(min(roi.shape) * 0.4),
#                                    maxLineGap=int(min(roi.shape) * 0.2))
            
#             if lines is None or len(lines) < 2:
#                 return 0.0
            
#             # Ստուգել անկյունները
#             angles = []
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
#                 angles.append(angle)
            
#             # X-ը պետք է ունենա ~45° և ~135° գծեր
#             has_diagonal1 = any(35 <= a <= 55 for a in angles)
#             has_diagonal2 = any(125 <= a <= 145 for a in angles)
            
#             if has_diagonal1 and has_diagonal2:
#                 return min(1.0, len(lines) / 4.0)
            
#             return 0.0
            
#         except:
#             return 0.0
    
#     def _detect_by_diagonals(self, roi: np.ndarray) -> float:
#         """Ստուգել diagonal pixel density"""
#         try:
#             h, w = roi.shape
            
#             # Main diagonal
#             diag1 = np.array([roi[i, i] for i in range(min(h, w))])
            
#             # Anti-diagonal
#             diag2 = np.array([roi[i, w-1-i] for i in range(min(h, w))])
            
#             # Pixel density (black pixels)
#             density1 = np.sum(diag1 < 128) / len(diag1)
#             density2 = np.sum(diag2 < 128) / len(diag2)
            
#             # X-ը պետք է ունենա բարձր density երկու diagonals-ում
#             avg_density = (density1 + density2) / 2
            
#             return min(1.0, avg_density * 2)
            
#         except:
#             return 0.0
    
#     def _detect_by_corners(self, roi: np.ndarray) -> float:
#         """Հայտնաբերում corner detection-ով"""
#         try:
#             corners = cv2.goodFeaturesToTrack(roi, 10, 0.01, 10)
            
#             if corners is None:
#                 return 0.0
            
#             # X-ը պետք է ունենա corner կենտրոնում
#             h, w = roi.shape
#             center = (w // 2, h // 2)
            
#             # Ստուգել արդյոք corner կա կենտրոնի մոտ
#             for corner in corners:
#                 x, y = corner.ravel()
#                 dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
#                 if dist < min(h, w) * 0.3:
#                     return 0.8
            
#             return 0.0
            
#         except:
#             return 0.0
    
#     def _detect_by_template(self, roi: np.ndarray) -> float:
#         """Template matching"""
#         try:
#             # Ստեղծել X template
#             h, w = roi.shape
#             template = np.ones((h, w), dtype=np.uint8) * 255
            
#             # Նկարել X
#             cv2.line(template, (0, 0), (w-1, h-1), 0, max(2, w//20))
#             cv2.line(template, (0, h-1), (w-1, 0), 0, max(2, w//20))
            
#             # Match
#             result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
#             _, max_val, _, _ = cv2.minMaxLoc(result)
            
#             return max(0.0, min(1.0, max_val))
            
#         except:
#             return 0.0
    
#     def _save_debug_info(self, roi, cell_id, scores, confidence, debug_dir):
#         """Պահպանել debug ինֆորմացիա"""
#         try:
#             os.makedirs(debug_dir, exist_ok=True)
            
#             # Save ROI
#             cv2.imwrite(f"{debug_dir}/cell_{cell_id}_roi.png", roi)
            
#             # Save info
#             with open(f"{debug_dir}/cell_{cell_id}_scores.txt", 'w') as f:
#                 f.write(f"Cell: {cell_id}\n")
#                 f.write(f"Lines score: {scores[0]:.3f}\n")
#                 f.write(f"Diagonals score: {scores[1]:.3f}\n")
#                 f.write(f"Corners score: {scores[2]:.3f}\n")
#                 f.write(f"Template score: {scores[3]:.3f}\n")
#                 f.write(f"Final confidence: {confidence:.3f}\n")
#                 f.write(f"Threshold: {self.threshold:.3f}\n")
#                 f.write(f"Result: {'X DETECTED' if confidence >= self.threshold else 'NO X'}\n")
#         except:
#             pass


# # ============================================================================
# # LAYOUT VALIDATOR
# # ============================================================================

# class LayoutValidator:
#     """Layout-ի validation և համեմատություն"""
    
#     def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
#         self.validation_level = validation_level
    
#     def validate_against_master(self, 
#                                 student_rectangles: List[Dict],
#                                 answer_key: Dict,
#                                 student_name: str = "Unknown") -> ValidationResult:
#         """
#         Համեմատել student-ի layout-ը master-ի հետ
#         """
        
#         warnings = []
#         errors = []
#         confidence_scores = []
        
#         # 1. Ստուգել քառակուսիների քանակը
#         expected_total = answer_key['total_questions']
#         detected_total = len(student_rectangles)
        
#         if detected_total != expected_total:
#             error_msg = (f"❌ CRITICAL: Քառակուսիների քանակ չի համապատասխանում! "
#                         f"Հայտնաբերված {detected_total}, ակնկալվում էր {expected_total}")
#             errors.append(error_msg)
#             confidence_scores.append(0.0)
#         else:
#             confidence_scores.append(1.0)
        
#         # 2. Ստուգել բլոկների քանակը
#         detected_blocks = len(set(r.get('block_id', 1) for r in student_rectangles))
#         expected_blocks = answer_key['layout']['total_blocks']
        
#         if detected_blocks != expected_blocks:
#             error_msg = (f"❌ CRITICAL: Բլոկների քանակ չի համապատասխանում! "
#                         f"Հայտնաբերված {detected_blocks}, ակնկալվում էր {expected_blocks}")
#             errors.append(error_msg)
#             confidence_scores.append(0.0)
#         else:
#             confidence_scores.append(1.0)
        
#         # 3. Ստուգել յուրաքանչյուր բլոկի կառուցվածքը
#         for block_id, block_data in answer_key['blocks'].items():
#             expected_cells = block_data['metadata']['total_cells']
            
#             detected_cells = sum(1 for r in student_rectangles 
#                                if str(r.get('block_id', 1)) == block_id)
            
#             if detected_cells != expected_cells:
#                 warning_msg = (f"⚠️  Block {block_id}: Հայտնաբերված {detected_cells} վանդակ, "
#                              f"ակնկալվում էր {expected_cells}")
#                 warnings.append(warning_msg)
#                 confidence_scores.append(0.7)
#             else:
#                 confidence_scores.append(1.0)
        
#         # 4. Ստուգել matrix labels-ը
#         expected_labels = set()
#         for block_data in answer_key['blocks'].values():
#             expected_labels.update(block_data['answers'].keys())
        
#         detected_labels = set(r.get('matrix_label', '') for r in student_rectangles)
        
#         missing_labels = expected_labels - detected_labels
#         extra_labels = detected_labels - expected_labels
        
#         if missing_labels:
#             error_msg = f"❌ Բացակա վանդակներ: {', '.join(sorted(missing_labels))}"
#             errors.append(error_msg)
#             confidence_scores.append(0.5)
        
#         if extra_labels:
#             warning_msg = f"⚠️  Ավելորդ վանդակներ: {', '.join(sorted(extra_labels))}"
#             warnings.append(warning_msg)
#             confidence_scores.append(0.8)
        
#         # 5. Հաշվել ընդհանուր confidence
#         overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
#         # 6. Որոշել is_valid ըստ validation level-ի
#         is_valid = self._determine_validity(errors, warnings, overall_confidence)
        
#         return ValidationResult(
#             is_valid=is_valid,
#             warnings=warnings,
#             errors=errors,
#             confidence_score=overall_confidence
#         )
    
#     def _determine_validity(self, errors: List[str], warnings: List[str], 
#                            confidence: float) -> bool:
#         """Որոշել արդյոք valid է"""
        
#         if self.validation_level == ValidationLevel.STRICT:
#             return len(errors) == 0 and len(warnings) == 0 and confidence >= 0.95
        
#         elif self.validation_level == ValidationLevel.NORMAL:
#             return len(errors) == 0 and confidence >= 0.80
        
#         else:  # RELAXED
#             return len(errors) == 0 and confidence >= 0.60


# # ============================================================================
# # BULLETPROOF ANSWER KEY SYSTEM
# # ============================================================================

# class BulletproofAnswerKeySystem:
#     """
#     🛡️ Ամբողջական պաշտպանված Answer Key համակարգ
#     """
    
#     def __init__(self, 
#                  validation_level: ValidationLevel = ValidationLevel.NORMAL,
#                  confidence_threshold: ConfidenceThreshold = ConfidenceThreshold.MEDIUM,
#                  enable_manual_review: bool = True,
#                  log_dir: str = "logs"):
        
#         self.validation_level = validation_level
#         self.confidence_threshold = confidence_threshold
#         self.enable_manual_review = enable_manual_review
        
#         # Directories
#         self.answer_keys_dir = Path('answer_keys')
#         self.manual_review_dir = Path('manual_review')
#         self.answer_keys_dir.mkdir(exist_ok=True)
#         self.manual_review_dir.mkdir(exist_ok=True)
        
#         # Components
#         self.logger = GradingLogger(log_dir)
#         self.x_detector = ConfidentXDetector(confidence_threshold)
#         self.layout_validator = LayoutValidator(validation_level)
        
#         self.logger.logger.info(f"⚙️  Configuration:")
#         self.logger.logger.info(f"   Validation level: {validation_level.value}")
#         self.logger.logger.info(f"   Confidence threshold: {confidence_threshold.value}")
#         self.logger.logger.info(f"   Manual review: {enable_manual_review}")
    
#     def create_answer_key_from_master(self,
#                                      master_image_path: str,
#                                      test_name: str,
#                                      description: str = "",
#                                      verify_scan_quality: bool = True) -> Dict:
#         """
#         Ստեղծել Answer Key master նկարից՝ ամբողջական validation-ով
#         """
        
#         self.logger.log_step("CREATE ANSWER KEY", f"Test: {test_name}")
        
#         # 1. Կարդալ նկարը
#         self.logger.log_step("1", "Կարդում ենք master նկարը...")
#         img = cv2.imread(master_image_path)
        
#         if img is None:
#             self.logger.log_error(f"Նկարը չի գտնվել: {master_image_path}")
#             raise FileNotFoundError(f"Master image not found: {master_image_path}")
        
#         self.logger.logger.info(f"✅ Նկարը կարդացվեց: {img.shape[1]}x{img.shape[0]}px")
        
#         # 2. Ստուգել scan որակը
#         if verify_scan_quality:
#             quality_score = self._assess_scan_quality(img)
#             self.logger.logger.info(f"📊 Scan որակ: {quality_score:.2%}")
            
#             if quality_score < 0.6:
#                 self.logger.log_warning("Scan որակը ցածր է! Խորհուրդ է տրվում վերականգնել։")
        
#         # 3. Հայտնաբերել քառակուսիները և X-երը
#         self.logger.log_step("2", "Հայտնաբերում ենք քառակուսիները...")
        
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Import քո ֆունկցիան
#         from your_module import find_rectangles_and_x
        
#         all_rectangles, x_marks, processed_img, blocks = find_rectangles_and_x(
#             img, gray,
#             output_path=str(self.answer_keys_dir / f'{test_name}_master_detected.jpg'),
#             save_crops=True,
#             crops_dir=str(self.answer_keys_dir / f'{test_name}_master_crops'),
#             debug=True,
#             debug_dir=str(self.answer_keys_dir / f'{test_name}_debug')
#         )
        
#         self.logger.logger.info(f"✅ Հայտնաբերված:")
#         self.logger.logger.info(f"   Քառակուսիներ: {len(all_rectangles)}")
#         self.logger.logger.info(f"   X-եր (ճիշտ պատասխաններ): {len(x_marks)}")
#         self.logger.logger.info(f"   Բլոկներ: {len(blocks)}")
        
#         # 4. Վերստուգել X հայտնաբերումը confidence-ով
#         self.logger.log_step("3", "Վերստուգում ենք X-երը confidence scoring-ով...")
        
#         verified_results = []
#         needs_review = []
        
#         for rect in all_rectangles:
#             x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
            
#             # Extract ROI
#             margin = max(5, int(min(w, h) * 0.18))
#             roi = gray[y+margin:y+h-margin, x+margin:x+w-margin]
            
#             # Detect with confidence
#             cell_id = f"b{rect.get('block_id', 1)}_{rect.get('matrix_label', 'unknown')}"
#             has_x, confidence = self.x_detector.detect_x_with_confidence(
#                 roi, cell_id, debug=True,
#                 debug_dir=str(self.answer_keys_dir / f'{test_name}_confidence_debug')
#             )
            
#             result = DetectionResult(
#                 block_id=rect.get('block_id', 1),
#                 matrix_label=rect.get('matrix_label', ''),
#                 row=rect.get('row', 0),
#                 col=rect.get('col', 0),
#                 has_x=has_x,
#                 confidence=confidence,
#                 needs_review=confidence < ConfidenceThreshold.HIGH.value,
#                 x=x, y=y, w=w, h=h
#             )
            
#             verified_results.append(result)
            
#             if result.needs_review:
#                 needs_review.append(result)
        
#         self.logger.logger.info(f"✅ X հայտնաբերում:")
#         self.logger.logger.info(f"   Բարձր confidence: {sum(1 for r in verified_results if r.confidence >= ConfidenceThreshold.HIGH.value)}")
#         self.logger.logger.info(f"   Ցածր confidence (needs review): {len(needs_review)}")
        
#         # 7. Manual review եթե անհրաժեշտ է
#         if needs_review and self.enable_manual_review:
#             self.logger.log_warning(f"{len(needs_review)} վանդակ պահանջում է manual review")
#             self._prepare_manual_review(img, needs_review, student_name, "student")
        
#         # 8. Գնահատել
#         self.logger.log_step("6", "Գնահատում ենք պատասխանները...")
        
#         grading_result = self._grade_answers(
#             verified_results, answer_key, student_name,
#             validation_result, quality_score
#         )
        
#         # 9. Ստեղծել վիզուալ արդյունքներ
#         self.logger.log_step("7", "Ստեղծում ենք վիզուալ արդյունքներ...")
        
#         self._generate_comprehensive_visual_results(
#             img, all_rectangles, grading_result, 
#             verified_results, output_path
#         )
        
#         # 10. Պահպանել հաշվետվությունները
#         self._save_comprehensive_reports(grading_result, output_path)
        
#         # 11. Calibration mode - ցույց տալ մանրամասն ինֆորմացիա
#         if calibration_mode:
#             self._display_calibration_summary(grading_result, verified_results)
        
#         self.logger.log_step("COMPLETE", f"Գնահատումը ավարտված - {grading_result['score_percentage']:.1f}%")
        
#         return grading_result
    
#     def batch_grade_with_calibration(self,
#                                     answer_key_name: str,
#                                     students_dir: str,
#                                     output_base_dir: str = 'graded_batch',
#                                     run_calibration: bool = True) -> List[Dict]:
#         """
#         Batch grading՝ calibration step-ով
#         """
        
#         self.logger.log_step("BATCH GRADING START", f"Directory: {students_dir}")
        
#         import glob
        
#         # Գտնել բոլոր նկարները
#         image_files = []
#         for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
#             image_files.extend(glob.glob(os.path.join(students_dir, ext)))
        
#         if not image_files:
#             self.logger.log_error(f"Նկարներ չգտնվեցին: {students_dir}")
#             return []
        
#         self.logger.logger.info(f"✅ Գտնվել է {len(image_files)} նկար")
        
#         # CALIBRATION STEP
#         if run_calibration and len(image_files) > 0:
#             self.logger.log_step("CALIBRATION", "Calibration test 1 սովորողի վրա")
            
#             calibration_file = image_files[0]
#             calibration_name = Path(calibration_file).stem
            
#             self.logger.logger.info(f"📋 Calibration test: {calibration_name}")
            
#             try:
#                 calibration_result = self.grade_student_work(
#                     calibration_file,
#                     answer_key_name,
#                     calibration_name,
#                     output_dir=f"{output_base_dir}/calibration_{calibration_name}",
#                     calibration_mode=True
#                 )
                
#                 # Ցույց տալ calibration արդյունքները
#                 print("\n" + "="*70)
#                 print("🔍 CALIBRATION TEST COMPLETE")
#                 print("="*70)
#                 print(f"Student: {calibration_name}")
#                 print(f"Score: {calibration_result['score_percentage']:.1f}%")
#                 print(f"Confidence: {calibration_result['overall_confidence']:.2%}")
#                 print(f"Low confidence detections: {calibration_result['stats']['needs_review']}")
#                 print("="*70)
                
#                 # Հարցնել օգտատիրոջը
#                 print("\n👉 Խնդրում ենք ստուգել calibration արդյունքները:")
#                 print(f"   📁 Folder: {output_base_dir}/calibration_{calibration_name}")
#                 print(f"   🖼️  Visual: graded_visual.jpg")
#                 print(f"   📄 Report: detailed_report.txt\n")
                
#                 response = input("Շարունակե՞լ batch processing-ը? (yes/no): ").strip().lower()
                
#                 if response not in ['yes', 'y', 'այո']:
#                     self.logger.logger.info("❌ Batch processing ընդհատված օգտատիրոջ կողմից")
#                     return []
                
#                 self.logger.logger.info("✅ Calibration հաստատված - շարունակում ենք")
                
#             except Exception as e:
#                 self.logger.log_error(f"Calibration failed: {e}")
#                 raise
        
#         # BATCH PROCESSING
#         all_results = []
#         failed = []
        
#         for i, img_path in enumerate(image_files, 1):
#             student_name = Path(img_path).stem
#             output_dir = os.path.join(output_base_dir, student_name)
            
#             print(f"\n[{i}/{len(image_files)}] Processing: {student_name}")
#             print("-" * 70)
            
#             try:
#                 result = self.grade_student_work(
#                     img_path,
#                     answer_key_name,
#                     student_name,
#                     output_dir
#                 )
                
#                 all_results.append(result)
#                 print(f"✅ Score: {result['score_percentage']:.1f}%")
                
#             except Exception as e:
#                 self.logger.log_error(f"Failed to grade {student_name}: {e}")
#                 failed.append((student_name, str(e)))
#                 print(f"❌ Error: {e}")
        
#         # Ստեղծել class summary
#         if all_results:
#             self._generate_class_summary(all_results, output_base_dir)
        
#         # Ցույց տալ failed cases
#         if failed:
#             self.logger.log_warning(f"Failed to grade {len(failed)} students:")
#             for name, error in failed:
#                 self.logger.logger.warning(f"   - {name}: {error}")
        
#         self.logger.log_step("BATCH COMPLETE", f"Successfully graded {len(all_results)}/{len(image_files)}")
        
#         return all_results
    
#     # ========================================================================
#     # HELPER METHODS
#     # ========================================================================
    
#     def _assess_scan_quality(self, img: np.ndarray) -> float:
#         """Գնահատել scan որակը"""
#         scores = []
        
#         # 1. Resolution check
#         h, w = img.shape[:2]
#         resolution_score = min(1.0, (h * w) / (1000 * 1000))
#         scores.append(resolution_score)
        
#         # 2. Brightness/Contrast
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         mean_brightness = np.mean(gray)
#         contrast = np.std(gray)
        
#         brightness_score = 1.0 - abs(mean_brightness - 127) / 127
#         contrast_score = min(1.0, contrast / 50)
        
#         scores.append(brightness_score)
#         scores.append(contrast_score)
        
#         # 3. Blur detection (Laplacian variance)
#         laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         blur_score = min(1.0, laplacian_var / 500)
#         scores.append(blur_score)
        
#         return np.mean(scores)
    
#     def _build_answer_key_structure(self, test_name, description, master_path,
#                                    verified_results, img_shape, num_blocks) -> Dict:
#         """Ստեղծել answer key structure"""
        
#         answer_key = {
#             'test_name': test_name,
#             'description': description,
#             'created_at': datetime.now().isoformat(),
#             'master_image': master_path,
#             'total_questions': len(verified_results),
#             'correct_answers_count': sum(1 for r in verified_results if r.has_x),
#             'layout': {
#                 'total_blocks': num_blocks,
#                 'image_width': img_shape[1],
#                 'image_height': img_shape[0],
#             },
#             'blocks': {}
#         }
        
#         # Խմբավորել ըստ բլոկների
#         for result in verified_results:
#             block_id = str(result.block_id)
            
#             if block_id not in answer_key['blocks']:
#                 answer_key['blocks'][block_id] = {
#                     'type': 'multiple_choice',
#                     'answers': {},
#                     'metadata': {
#                         'rows': set(),
#                         'cols': set()
#                     }
#                 }
            
#             answer_key['blocks'][block_id]['answers'][result.matrix_label] = result.has_x
#             answer_key['blocks'][block_id]['metadata']['rows'].add(result.row)
#             answer_key['blocks'][block_id]['metadata']['cols'].add(result.col)
        
#         # Convert sets to lists
#         for block_id in answer_key['blocks']:
#             metadata = answer_key['blocks'][block_id]['metadata']
#             metadata['rows'] = sorted(list(metadata['rows']))
#             metadata['cols'] = sorted(list(metadata['cols']))
#             metadata['total_cells'] = len(answer_key['blocks'][block_id]['answers'])
        
#         return answer_key
    
#     def _grade_answers(self, verified_results, answer_key, student_name,
#                       validation_result, quality_score) -> Dict:
#         """Գնահատել պատասխանները"""
        
#         result = {
#             'student_name': student_name,
#             'test_name': answer_key['test_name'],
#             'graded_at': datetime.now().isoformat(),
#             'scan_quality': quality_score,
#             'validation_result': asdict(validation_result),
#             'total_questions': answer_key['total_questions'],
#             'correct': 0,
#             'incorrect': 0,
#             'missing': 0,
#             'extra': 0,
#             'score_percentage': 0,
#             'overall_confidence': 0,
#             'details_by_block': {},
#             'all_details': [],
#             'stats': {
#                 'high_confidence': 0,
#                 'medium_confidence': 0,
#                 'low_confidence': 0,
#                 'needs_review': 0
#             }
#         }
        
#         # Ստեղծել detected map
#         detected_map = {}
#         confidence_map = {}
        
#         for res in verified_results:
#             block_id = str(res.block_id)
#             if block_id not in detected_map:
#                 detected_map[block_id] = set()
#                 confidence_map[block_id] = {}
            
#             if res.has_x:
#                 detected_map[block_id].add(res.matrix_label)
            
#             confidence_map[block_id][res.matrix_label] = res.confidence
            
#             # Stats
#             if res.confidence >= ConfidenceThreshold.HIGH.value:
#                 result['stats']['high_confidence'] += 1
#             elif res.confidence >= ConfidenceThreshold.MEDIUM.value:
#                 result['stats']['medium_confidence'] += 1
#             else:
#                 result['stats']['low_confidence'] += 1
            
#             if res.needs_review:
#                 result['stats']['needs_review'] += 1
        
#         # Համեմատել
#         confidences = []
        
#         for block_id, block_data in answer_key['blocks'].items():
#             block_result = {
#                 'block_id': block_id,
#                 'correct': 0,
#                 'incorrect': 0,
#                 'details': []
#             }
            
#             for question_id, should_be_marked in block_data['answers'].items():
#                 is_marked = question_id in detected_map.get(block_id, set())
#                 confidence = confidence_map.get(block_id, {}).get(question_id, 0.0)
                
#                 confidences.append(confidence)
                
#                 detail = {
#                     'block': block_id,
#                     'question': question_id,
#                     'expected': should_be_marked,
#                     'detected': is_marked,
#                     'confidence': confidence,
#                     'status': '',
#                     'icon': ''
#                 }
                
#                 if should_be_marked and is_marked:
#                     result['correct'] += 1
#                     block_result['correct'] += 1
#                     detail['status'] = 'correct'
#                     detail['icon'] = '✅'
                    
#                 elif should_be_marked and not is_marked:
#                     result['missing'] += 1
#                     result['incorrect'] += 1
#                     block_result['incorrect'] += 1
#                     detail['status'] = 'missing'
#                     detail['icon'] = '❌'
#                     detail['note'] = 'Պետք է նշված լիներ'
                    
#                 elif not should_be_marked and is_marked:
#                     result['extra'] += 1
#                     result['incorrect'] += 1
#                     block_result['incorrect'] += 1
#                     detail['status'] = 'extra'
#                     detail['icon'] = '❌'
#                     detail['note'] = 'Չպետք է նշված լիներ'
                    
#                 else:
#                     result['correct'] += 1
#                     block_result['correct'] += 1
#                     detail['status'] = 'correct'
#                     detail['icon'] = '✅'
                
#                 block_result['details'].append(detail)
#                 result['all_details'].append(detail)
            
#             result['details_by_block'][block_id] = block_result
        
#         # Հաշվել գնահատականը
#         if result['total_questions'] > 0:
#             result['score_percentage'] = round(
#                 (result['correct'] / result['total_questions']) * 100, 2
#             )
        
#         result['overall_confidence'] = np.mean(confidences) if confidences else 0.0
        
#         return result
    
#     def _prepare_manual_review(self, img, needs_review, name, type_str):
#         """Պատրաստել manual review"""
        
#         review_dir = self.manual_review_dir / f"{type_str}_{name}"
#         review_dir.mkdir(parents=True, exist_ok=True)
        
#         # Պահպանել յուրաքանչյուր անորոշ վանդակը
#         for i, result in enumerate(needs_review):
#             x, y, w, h = result.x, result.y, result.w, result.h
#             crop = img[y:y+h, x:x+w]
            
#             filename = f"review_{i+1}_b{result.block_id}_{result.matrix_label}_conf{result.confidence:.2f}.jpg"
#             cv2.imwrite(str(review_dir / filename), crop)
        
#         # Ստեղծել review report
#         with open(review_dir / "review_needed.txt", 'w', encoding='utf-8') as f:
#             f.write(f"MANUAL REVIEW NEEDED\n")
#             f.write(f"{'='*70}\n\n")
#             f.write(f"Name: {name}\n")
#             f.write(f"Type: {type_str}\n")
#             f.write(f"Total items needing review: {len(needs_review)}\n\n")
#             f.write(f"{'='*70}\n\n")
            
#             for i, result in enumerate(needs_review):
#                 f.write(f"{i+1}. Block {result.block_id}, Cell {result.matrix_label}\n")
#                 f.write(f"   Detected as: {'X' if result.has_x else 'Empty'}\n")
#                 f.write(f"   Confidence: {result.confidence:.2%}\n")
#                 f.write(f"   Image: review_{i+1}_b{result.block_id}_{result.matrix_label}_conf{result.confidence:.2f}.jpg\n\n")
        
#         self.logger.logger.info(f"📋 Manual review պատրաստված: {review_dir}")
    
#     def _generate_comprehensive_visual_results(self, img, rectangles, 
#                                               grading_result, verified_results, output_path):
#         """Ստեղծել comprehensive վիզուալ արդյունքներ"""
        
#         img_result = img.copy()
        
#         # Ստեղծել confidence map
#         confidence_map = {}
#         for vr in verified_results:
#             key = f"{vr.block_id}_{vr.matrix_label}"
#             confidence_map[key] = vr.confidence
        
#         # Ստեղծել details map
#         details_map = {}
#         for detail in grading_result['all_details']:
#             key = f"{detail['block']}_{detail['question']}"
#             details_map[key] = detail
        
#         # Նկարել քառակուսիները
#         for rect in rectangles:
#             block_id = str(rect.get('block_id', 1))
#             matrix_label = rect.get('matrix_label')
#             key = f"{block_id}_{matrix_label}"
            
#             if key not in details_map:
#                 continue
            
#             detail = details_map[key]
#             confidence = confidence_map.get(key, 0.0)
            
#             x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
            
#             # Ընտրել գույնը confidence-ի համաձայն
#             if detail['status'] == 'correct':
#                 if confidence >= ConfidenceThreshold.HIGH.value:
#                     color = (0, 200, 0)  # Մուգ կանաչ
#                 else:
#                     color = (100, 200, 100)  # Բաց կանաչ
#                 thickness = 2
#             else:
#                 if confidence >= ConfidenceThreshold.MEDIUM.value:
#                     color = (0, 0, 200)  # Մուգ կարմիր
#                 else:
#                     color = (100, 100, 200)  # Բաց կարմիր
#                 thickness = 3
            
#             cv2.rectangle(img_result, (x, y), (x+w, y+h), color, thickness)
            
#             # Confidence indicator (փոքր շրջան անկյունում)
#             radius = 8
#             conf_x = x + w - radius - 3
#             conf_y = y + radius + 3
            
#             if confidence >= ConfidenceThreshold.HIGH.value:
#                 conf_color = (0, 255, 0)
#             elif confidence >= ConfidenceThreshold.MEDIUM.value:
#                 conf_color = (255, 165, 0)
#             else:
#                 conf_color = (0, 0, 255)
            
#             cv2.circle(img_result, (conf_x, conf_y), radius, conf_color, -1)
#             cv2.circle(img_result, (conf_x, conf_y), radius, (0, 0, 0), 1)
        
#         # Score box
#         box_h = 180
#         cv2.rectangle(img_result, (10, 10), (500, box_h), (255, 255, 255), -1)
#         cv2.rectangle(img_result, (10, 10), (500, box_h), (0, 0, 0), 2)
        
#         # Score
#         score_color = (0, 150, 0) if grading_result['score_percentage'] >= 70 else (0, 0, 200)
#         cv2.putText(img_result, f"Score: {grading_result['score_percentage']:.1f}%", 
#                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 2)
        
#         # Stats
#         cv2.putText(img_result, f"Correct: {grading_result['correct']}/{grading_result['total_questions']}", 
#                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
#         cv2.putText(img_result, f"Errors: {grading_result['incorrect']} (Miss: {grading_result['missing']}, Extra: {grading_result['extra']})", 
#                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
#         cv2.putText(img_result, f"Confidence: {grading_result['overall_confidence']:.1%}", 
#                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
#         cv2.putText(img_result, f"Scan Quality: {grading_result['scan_quality']:.1%}", 
#                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
#         if grading_result['stats']['needs_review'] > 0:
#             cv2.putText(img_result, f"Needs Review: {grading_result['stats']['needs_review']}", 
#                        (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
        
#         # Legend
#         legend_y = box_h + 30
#         cv2.putText(img_result, "Legend:", (20, legend_y), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
#         # High confidence
#         cv2.circle(img_result, (30, legend_y + 20), 6, (0, 255, 0), -1)
#         cv2.putText(img_result, "High confidence", (45, legend_y + 25), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
#         # Medium
#         cv2.circle(img_result, (30, legend_y + 40), 6, (255, 165, 0), -1)
#         cv2.putText(img_result, "Medium confidence", (45, legend_y + 45), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
#         # Low
#         cv2.circle(img_result, (30, legend_y + 60), 6, (0, 0, 255), -1)
#         cv2.putText(img_result, "Low confidence - Review!", (45, legend_y + 65), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
#         cv2.imwrite(str(output_path / 'graded_visual.jpg'), img_result)
    
#     def _save_comprehensive_reports(self, grading_result, output_path):
#         """Պահպանել comprehensive reports"""
        
#         # JSON
#         json_path = output_path / 'grading_report.json'
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(grading_result, f, indent=2, ensure_ascii=False)
        
#         # Detailed text report
#         txt_path = output_path / 'detailed_report.txt'
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.write(f"{'='*70}\n")
#             f.write(f"DETAILED GRADING REPORT\n")
#             f.write(f"{'='*70}\n\n")
            
#             f.write(f"Student: {grading_result['student_name']}\n")
#             f.write(f"Test: {grading_result['test_name']}\n")
#             f.write(f"Date: {grading_result['graded_at']}\n\n")
            
#             f.write(f"{'─'*70}\n")
#             f.write(f"SCORE: {grading_result['score_percentage']:.1f}%\n")
#             f.write(f"{'─'*70}\n\n")
            
#             f.write(f"Total Questions: {grading_result['total_questions']}\n")
#             f.write(f"✅ Correct: {grading_result['correct']}\n")
#             f.write(f"❌ Incorrect: {grading_result['incorrect']}\n")
#             f.write(f"   - Missing: {grading_result['missing']}\n")
#             f.write(f"   - Extra: {grading_result['extra']}\n\n")
            
#             f.write(f"Scan Quality: {grading_result['scan_quality']:.1%}\n")
#             f.write(f"Overall Confidence: {grading_result['overall_confidence']:.1%}\n\n")
            
#             f.write(f"{'─'*70}\n")
#             f.write(f"CONFIDENCE STATISTICS\n")
#             f.write(f"{'─'*70}\n\n")
#             f.write(f"High confidence detections: {grading_result['stats']['high_confidence']}\n")
#             f.write(f"Medium confidence: {grading_result['stats']['medium_confidence']}\n")
#             f.write(f"Low confidence: {grading_result['stats']['low_confidence']}\n")
#             f.write(f"⚠️  Needs manual review: {grading_result['stats']['needs_review']}\n\n")
            
#             if grading_result['incorrect'] > 0:
#                 f.write(f"{'─'*70}\n")
#                 f.write(f"ERROR DETAILS\n")
#                 f.write(f"{'─'*70}\n\n")
                
#                 for detail in grading_result['all_details']:
#                     if detail['status'] != 'correct':
#                         f.write(f"{detail['icon']} Block {detail['block']}, Question {detail['question']}\n")
#                         f.write(f"   Expected: {'X' if detail['expected'] else 'Empty'}\n")
#                         f.write(f"   Detected: {'X' if detail['detected'] else 'Empty'}\n")
#                         f.write(f"   Confidence: {detail['confidence']:.2%}\n")
#                         f.write(f"   Note: {detail.get('note', 'N/A')}\n\n")
            
#             # Validation warnings
#             if grading_result['validation_result']['warnings']:
#                 f.write(f"\n{'─'*70}\n")
#                 f.write(f"VALIDATION WARNINGS\n")
#                 f.write(f"{'─'*70}\n\n")
#                 for w in grading_result['validation_result']['warnings']:
#                     f.write(f"⚠️  {w}\n")
    
#     def _display_calibration_summary(self, grading_result, verified_results):
#         """Ցույց տալ calibration summary console-ում"""
        
#         print("\n" + "="*70)
#         print("🔍 CALIBRATION MODE - DETAILED SUMMARY")
#         print("="*70)
#         print(f"\nStudent: {grading_result['student_name']}")
#         print(f"Score: {grading_result['score_percentage']:.1f}%")
#         print(f"Overall Confidence: {grading_result['overall_confidence']:.2%}")
#         print(f"Scan Quality: {grading_result['scan_quality']:.2%}")
        
#         print(f"\nConfidence Distribution:")
#         print(f"  🟢 High: {grading_result['stats']['high_confidence']}")
#         print(f"  🟡 Medium: {grading_result['stats']['medium_confidence']}")
#         print(f"  🔴 Low: {grading_result['stats']['low_confidence']}")
#         print(f"  ⚠️  Needs Review: {grading_result['stats']['needs_review']}")
        
#         if grading_result['validation_result']['warnings']:
#             print(f"\n⚠️  Validation Warnings:")
#             for w in grading_result['validation_result']['warnings']:
#                 print(f"  - {w}")
        
#         if grading_result['stats']['needs_review'] > 0:
#             print(f"\n⚠️  {grading_result['stats']['needs_review']} cells need manual review!")
#             print(f"  Check: manual_review/ folder")
        
#         print("="*70 + "\n")
    
#     def _generate_class_summary(self, all_results, output_dir: str):
#         """Ստեղծել class summary"""
        
#         if not all_results:
#             return
        
#         # Sort by score
#         all_results.sort(key=lambda x: x['score_percentage'], reverse=True)
        
#         summary = {
#             'generated_at': datetime.now().isoformat(),
#             'test_name': all_results[0]['test_name'],
#             'total_students': len(all_results),
#             'class_statistics': {
#                 'average_score': sum(r['score_percentage'] for r in all_results) / len(all_results),
#                 'highest_score': all_results[0]['score_percentage'],
#                 'lowest_score': all_results[-1]['score_percentage'],
#                 'students_passed': sum(1 for r in all_results if r['score_percentage'] >= 70),
#                 'students_failed': sum(1 for r in all_results if r['score_percentage'] < 70),
#                 'average_confidence': sum(r['overall_confidence'] for r in all_results) / len(all_results),
#             },
#             'student_results': all_results
#         }
        
#         # JSON
#         json_path = Path(output_dir) / 'class_summary.json'
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, indent=2, ensure_ascii=False)
        
#         # Text
#         txt_path = Path(output_dir) / 'class_summary.txt'
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.write(f"{'='*70}\n")
#             f.write(f"CLASS SUMMARY REPORT\n")
#             f.write(f"{'='*70}\n\n")
#             f.write(f"Test: {summary['test_name']}\n")
#             f.write(f"Date: {summary['generated_at']}\n")
#             f.write(f"Total Students: {summary['total_students']}\n\n")
            
#             f.write(f"{'─'*70}\n")
#             f.write(f"STATISTICS\n")
#             f.write(f"{'─'*70}\n")
#             stats = summary['class_statistics']
#             f.write(f"Average Score: {stats['average_score']:.1f}%\n")
#             f.write(f"Highest Score: {stats['highest_score']:.1f}%\n")
#             f.write(f"Lowest Score: {stats['lowest_score']:.1f}%\n")
#             f.write(f"Passed (≥70%): {stats['students_passed']}\n")
#             f.write(f"Failed (<70%): {stats['students_failed']}\n")
#             f.write(f"Average Confidence: {stats['average_confidence']:.1%}\n\n")
            
#             f.write(f"{'─'*70}\n")
#             f.write(f"STUDENT RANKINGS\n")
#             f.write(f"{'─'*70}\n\n")
            
#             for i, result in enumerate(all_results, 1):
#                 status = "✅" if result['score_percentage'] >= 70 else "❌"
#                 f.write(f"{i:2d}. {status} {result['student_name']}: "
#                        f"{result['score_percentage']:.1f}% "
#                        f"({result['correct']}/{result['total_questions']}) "
#                        f"Conf: {result['overall_confidence']:.1%}\n")
        
#         self.logger.logger.info(f"✅ Class summary ստեղծված: {output_dir}")
    
#     def _save_answer_key(self, answer_key: Dict, test_name: str):
#         """Պահպանել answer key"""
        
#         # JSON
#         json_path = self.answer_keys_dir / f'{test_name}.json'
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(answer_key, f, indent=2, ensure_ascii=False)
        
#         # Readable
#         txt_path = self.answer_keys_dir / f'{test_name}_readable.txt'
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.write(f"ANSWER KEY: {answer_key['test_name']}\n")
#             f.write(f"Created: {answer_key['created_at']}\n")
#             f.write(f"Total Questions: {answer_key['total_questions']}\n")
#             f.write(f"Correct Answers: {answer_key['correct_answers_count']}\n\n")
            
#             for block_id, block_data in answer_key['blocks'].items():
#                 f.write(f"\nBLOCK {block_id}:\n")
#                 f.write(f"{'─'*50}\n")
#                 correct_answers = [k for k, v in block_data['answers'].items() if v]
#                 if correct_answers:
#                     for ans in sorted(correct_answers):
#                         f.write(f"  ✓ {ans}\n")
        
#         self.logger.logger.info(f"✅ Answer key պահպանված: {json_path}")
    
#     def _load_answer_key(self, answer_key_name: str) -> Dict:
#         """Բեռնել answer key"""
        
#         json_path = self.answer_keys_dir / f'{answer_key_name}.json'
        
#         if not json_path.exists():
#             raise FileNotFoundError(f"Answer key not found: {json_path}")
        
#         with open(json_path, 'r', encoding='utf-8') as f:
#             answer_key = json.load(f)
        
#         self.logger.logger.info(f"✅ Answer key բեռնված: {answer_key['test_name']}")
        
#         return answer_key
    
#     def _save_validation_report(self, validation_result: ValidationResult, 
#                                output_path: Path, student_name: str):
#         """Պահպանել validation report"""
        
#         report_path = output_path / 'validation_report.txt'
        
#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write(f"VALIDATION REPORT\n")
#             f.write(f"{'='*70}\n\n")
#             f.write(f"Student: {student_name}\n")
#             f.write(f"Valid: {validation_result.is_valid}\n")
#             f.write(f"Confidence: {validation_result.confidence_score:.2%}\n\n")
            
#             if validation_result.errors:
#                 f.write(f"ERRORS ({len(validation_result.errors)}):\n")
#                 f.write(f"{'─'*70}\n")
#                 for error in validation_result.errors:
#                     f.write(f"❌ {error}\n")
#                 f.write("\n")
            
#             if validation_result.warnings:
#                 f.write(f"WARNINGS ({len(validation_result.warnings)}):\n")
#                 f.write(f"{'─'*70}\n")
#                 for warning in validation_result.warnings:
#                     f.write(f"⚠️  {warning}\n")
    
#     def _calculate_checksum(self, answer_key: Dict) -> str:
#         """Հաշվել checksum"""
        
#         # Serialize relevant parts
#         data = {
#             'test_name': answer_key['test_name'],
#             'blocks': answer_key['blocks']
#         }
        
#         json_str = json.dumps(data, sort_keys=True)
#         return hashlib.md5(json_str.encode()).hexdigest()


# # ============================================================================
# # USAGE EXAMPLES
# # ============================================================================

# if __name__ == "__main__":
#     print("🛡️ Bulletproof Answer Key System\n")
    
#     # Initialize
#     system = BulletproofAnswerKeySystem(
#         validation_level=ValidationLevel.NORMAL,
#         confidence_threshold=ConfidenceThreshold.MEDIUM,
#         enable_manual_review=True
#     )
    
#     # Example 1: Create answer key
#     # answer_key = system.create_answer_key_from_master(
#     #     master_image_path='master_filled.jpg',
#     #     test_name='math_test_v1',
#     #     description='Մաթեմատիկա Թեստ 1'
#     # )
    
#     # Example 2: Grade single student with calibration
#     # result = system.grade_student_work(
#     #     student_image_path='student1.jpg',
#     #     answer_key_name='math_test_v1',
#     #     student_name='Անի Սարգսյան',
#     #     calibration_mode=True
#     # )
    
#     # Example 3: Batch grading with calibration
#     # results = system.batch_grade_with_calibration(
#     #     answer_key_name='math_test_v1',
#     #     students_dir='scanned_tests/',
#     #     run_calibration=True
#     # )
    
#     print("\n✅ System ready!")