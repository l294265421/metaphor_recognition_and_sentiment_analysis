def write_lines(lines, file_path):
    with open(file_path, mode='w', encoding='utf-8') as out_file:
        for line in lines:
            out_file.write(line + '\n')


def read_all_lines(file_path):
    lines = []
    with open(file_path, encoding='utf-8') as in_file:
        for line in in_file:
            lines.append(line.strip())
    return lines