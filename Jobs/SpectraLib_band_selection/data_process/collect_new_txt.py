import os
import shutil

def collect_binavg_txt(src_root, dst_root, target_filename='trend_mean_10nm_binavg.txt'):
    for dirpath, dirnames, filenames in os.walk(src_root):
        if target_filename in filenames:
            rel_dir = os.path.relpath(dirpath, src_root)  # 保留原有相对目录结构
            dst_dir = os.path.join(dst_root, rel_dir)
            os.makedirs(dst_dir, exist_ok=True)
            src_file = os.path.join(dirpath, target_filename)
            dst_file = os.path.join(dst_dir, '10nm_binavg.txt')
            shutil.copy2(src_file, dst_file)
            print(f"已复制: {src_file} → {dst_file}")

if __name__ == '__main__':
    collect_binavg_txt(
        src_root="/home/Dataset/nw/spectraLib/unify_v2",
        dst_root="/home/Dataset/nw/spectraLib/unify_v3"
    )