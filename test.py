# def copy_lines_range(input_path, output_path, start, end):
#     with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
#         for i, line in enumerate(fin):
#             if i < start:
#                 continue
#             if i >= end:
#                 break
#             fout.write(line)

# ... existing code ...

# 示例用法：复制第100行到第200行（包含第100行，不包含第200行）
# copy_lines_range('moodle_trace7_8kb.txt', 'train.txt', 0, 53474)
# copy_lines_range('moodle_trace7_8kb.txt', 'warm.txt', 53474, 59105)
# copy_lines_range('moodle_trace7_8kb.txt', 'test.txt', 59105, 60000)

# import tensorflow as tf
# print("是否启用 GPU:", tf.config.list_physical_devices('GPU'))