def compare_binary_files(file1, file2):
    chunk_size = 4096
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            b1 = f1.read(chunk_size)
            b2 = f2.read(chunk_size)
            if b1 != b2:
                return False
            if not b1:  # EOF reached
                break
    return True

# 用法示例
if __name__ == "__main__":
    file_path1 = "../../../Desktop/test/1"
    file_path2 = "../../../Desktop/test/2"
    if compare_binary_files(file_path1, file_path2):
        print("两个二进制文件相同")
    else:
        print("两个二进制文件不同")
    # 对第一个文件分段，然后在第二个文件中查找每个分段
    segment_size = 1024  # 可以根据需要调整分段大小
    not_found_indices = []

    with open(file_path1, 'rb') as f1, open(file_path2, 'rb') as f2:
        data1 = f1.read()
        data2 = f2.read()
        num_segments = (len(data1) + segment_size - 1) // segment_size

        for idx in range(num_segments):
            start = idx * segment_size
            end = min(start + segment_size, len(data1))
            segment = data1[start:end]
            if segment not in data2:
                not_found_indices.append(idx)

    if not_found_indices:
        print("以下分段在文件2中未找到：", not_found_indices)
    else:
        print("文件1的所有分段都能在文件2中找到")