def compare_with_correct(test_file, correct_file="correct_answers.txt"):
    """
    Համեմատում է թեստի արդյունքները ճիշտ պատասխանների հետ
    ՍՅՈՒՆԵՐՈՎ ստուգում - ամբողջ սյունը համեմատում է ամբողջ սյունի հետ
    """
    
    # Կարդում ենք ֆայլերը
    with open(correct_file, 'r', encoding='utf-8') as f:
        correct_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Խմբավորում ենք ըստ բլոկների
    correct_blocks = {}
    test_blocks = {}
    current_block = None
    
    for line in correct_lines:
        if line.startswith('Block'):
            current_block = line
            correct_blocks[current_block] = []
        elif current_block and ' ' in line:
            correct_blocks[current_block].append(line)
    
    current_block = None
    for line in test_lines:
        if line.startswith('Block'):
            current_block = line
            test_blocks[current_block] = []
        elif current_block and ' ' in line:
            test_blocks[current_block].append(line)
    
    # Հաշվում ենք միավորները
    total_columns = 0
    correct_count = 0
    errors = []
    
    for block_name in correct_blocks:
        if block_name not in test_blocks:
            errors.append(f"❌ {block_name} չի գտնվել թեստում")
            continue
        
        correct_answers = correct_blocks[block_name]
        test_answers = test_blocks[block_name]
        
        # Խմբավորում ենք ըստ սյուներ/կոլոնների
        # Օրինակ 1.1, 1.2, 1.3, 1.4 → 1. բլոկի սյունը 1, սյունը 2 եւ այլն
        correct_columns = {}
        for answer in correct_answers:
            parts = answer.split()
            if len(parts) == 2:
                position, value = parts
                # position մոտ: 1.1 → սյունը = 1 (տասնորդական մասից առաջ)
                column_num = position.split('.')[0]
                if column_num not in correct_columns:
                    correct_columns[column_num] = []
                correct_columns[column_num].append(answer)
        
        test_columns = {}
        for answer in test_answers:
            parts = answer.split()
            if len(parts) == 2:
                position, value = parts
                column_num = position.split('.')[0]
                if column_num not in test_columns:
                    test_columns[column_num] = []
                test_columns[column_num].append(answer)
        
        # Ստուգում ենք ամեն սյունը
        for col_num in sorted(correct_columns.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            total_columns += 1
            
            if col_num not in test_columns:
                errors.append(f"❌ {block_name}, սյունը {col_num}: չի գտնվել թեստում")
                continue
            
            # Համեմատում ենք ԱՄԲՈՂՋ սյունը
            if sorted(correct_columns[col_num]) == sorted(test_columns[col_num]):
                correct_count += 1
                print(f"✅ {block_name}, սյունը {col_num}: ՃԻՇՏ")
            else:
                errors.append(
                    f"❌ {block_name}, սյունը {col_num}: ՍԽԱԼ\n"
                    f"   Ճիշտ:  {', '.join(sorted(correct_columns[col_num]))}\n"
                    f"   Թեստ:  {', '.join(sorted(test_columns[col_num]))}"
                )
    
    # Արդյունքներ
    print(f"\n{'='*50}")
    print(f"📊 ԸՆԴՀԱՆՈՒՐ ԱՐԴՅՈՒՆՔ (ՍՅՈՒՆԵՐՈՎ)")
    print(f"{'='*50}")
    print(f"Ընդամենը սյուներ: {total_columns}")
    print(f"Ճիշտ պատասխաններ: {correct_count}")
    print(f"Սխալ պատասխաններ: {total_columns - correct_count}")
    print(f"Միավոր: {correct_count}/{total_columns}")
    if total_columns > 0:
        print(f"Տոկոս: {(correct_count/total_columns*100):.1f}%")
    
    if errors:
        print(f"\n❌ ՍԽԱԼՆԵՐԻ ՑՈՒՑԱԿ:")
        for error in errors:
            print(error)
    
    return correct_count, total_columns, errors


# Օգտագործում
if __name__ == "__main__":
    score, total, errors = compare_with_correct("rectangles_full_image/result_simple.txt")



























# def compare_with_correct(test_file, correct_file="correct_answers.txt"):
#     """
#     Համեմատում է թեստի արդյունքները ճիշտ պատասխանների հետ
#     Վերադարձնում է միավորը և սխալների ցուցակը
#     """
    
#     # Կարդում ենք ֆայլերը
#     with open(correct_file, 'r', encoding='utf-8') as f:
#         correct_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
#     with open(test_file, 'r', encoding='utf-8') as f:
#         test_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
#     # Խմբավորում ենք ըստ բլոկների
#     correct_blocks = {}
#     test_blocks = {}
#     current_block = None
    
#     for line in correct_lines:
#         if line.startswith('Block'):
#             current_block = line
#             correct_blocks[current_block] = []
#         elif current_block and ' ' in line:
#             correct_blocks[current_block].append(line)
    
#     current_block = None
#     for line in test_lines:
#         if line.startswith('Block'):
#             current_block = line
#             test_blocks[current_block] = []
#         elif current_block and ' ' in line:
#             test_blocks[current_block].append(line)
    
#     # Հաշվում ենք միավորները
#     total_exercises = 0
#     correct_count = 0
#     errors = []
    
#     for block_name in correct_blocks:
#         if block_name not in test_blocks:
#             errors.append(f"❌ {block_name} չի գտնվել թեստում")
#             continue
        
#         correct_answers = correct_blocks[block_name]
#         test_answers = test_blocks[block_name]
        
#         # Խմբավորում ենք ըստ վարժությունների (ըստ տողերի)
#         # Օրինակ 1.1, 1.2, 1.3, 1.4 → վարժություն 1
#         correct_exercises = {}
#         for answer in correct_answers:
#             parts = answer.split()
#             if len(parts) == 2:
#                 position, value = parts
#                 exercise_num = position.split('.')[1]  # Վերցնում ենք տող համարը
#                 if exercise_num not in correct_exercises:
#                     correct_exercises[exercise_num] = []
#                 correct_exercises[exercise_num].append(answer)
        
#         test_exercises = {}
#         for answer in test_answers:
#             parts = answer.split()
#             if len(parts) == 2:
#                 position, value = parts
#                 exercise_num = position.split('.')[1]
#                 if exercise_num not in test_exercises:
#                     test_exercises[exercise_num] = []
#                 test_exercises[exercise_num].append(answer)
        
#         # Ստուգում ենք ամեն վարժությունը
#         for ex_num in correct_exercises:
#             total_exercises += 1
            
#             if ex_num not in test_exercises:
#                 errors.append(f"❌ {block_name}, տող {ex_num}: չի գտնվել թեստում")
#                 continue
            
#             # Համեմատում ենք ամբողջ տողը
#             if correct_exercises[ex_num] == test_exercises[ex_num]:
#                 correct_count += 1
#                 print(f"✅ {block_name}, տող {ex_num}: ՃԻՇՏ")
#             else:
#                 errors.append(
#                     f"❌ {block_name}, տող {ex_num}: ՍԽԱԼ\n"
#                     f"   Ճիշտ:  {', '.join(correct_exercises[ex_num])}\n"
#                     f"   Թեստ:  {', '.join(test_exercises[ex_num])}"
#                 )
    
#     # Արդյունքներ
#     print(f"\n{'='*50}")
#     print(f"📊 ԸՆԴՀԱՆՈՒՐ ԱՐԴՅՈՒՆՔ")
#     print(f"{'='*50}")
#     print(f"Ընդամենը վարժություններ: {total_exercises}")
#     print(f"Ճիշտ պատասխաններ: {correct_count}")
#     print(f"Սխալ պատասխաններ: {total_exercises - correct_count}")
#     print(f"Միավոր: {correct_count}/{total_exercises}")
#     print(f"Տոկոս: {(correct_count/total_exercises*100):.1f}%")
    
#     if errors:
#         print(f"\n❌ ՍԽԱԼՆԵՐԻ ՑՈՒՑԱԿ:")
#         for error in errors:
#             print(error)
    
#     return correct_count, total_exercises, errors


# # Օգտագործում
# if __name__ == "__main__":
#     score, total, errors = compare_with_correct("rectangles_full_image/result_simple.txt")