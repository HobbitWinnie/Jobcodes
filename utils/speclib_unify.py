import os
import math

ERROR_VALUE = -1.23e+34  # 异常值
WAVELENGTHS = list(range(401, 2501))  # 401-2500nm
EXCEPTION_VAL = float('nan')

def process_asd_file(filepath, start_wavelength=350, step=1):
    with open(filepath, 'r', encoding='utf8') as f:
        lines = f.readlines()
    values = []
    for line in lines[1:]:
        try:
            value = float(line.strip())
            if value == ERROR_VALUE:
                value = EXCEPTION_VAL
            else:
                value = round(value, 8)
            values.append(value)
        except Exception:
            values.append(EXCEPTION_VAL)
    data = {}
    for i, v in enumerate(values):
        wl = int(start_wavelength + i * step)
        data[wl] = v
    return data

def process_xy_file(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                wl = float(parts[0])
                v = float(parts[1])
                if v == ERROR_VALUE:
                    v = EXCEPTION_VAL
                else:
                    v = round(v, 8)
                iw = int(wl)
                data[iw] = v
            except Exception:
                data[iw] = EXCEPTION_VAL
    return data

def write_unified_txt(data_dict, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', encoding='utf8') as f:
        for wl in WAVELENGTHS:
            value = data_dict.get(wl, EXCEPTION_VAL)
            # 检查是否为nan（float类型），nan只输出nan字符串
            if isinstance(value, float) and math.isnan(value):
                f.write(f"{wl:.2f}\tnan\n")  # 你需要什么格式就写什么，比如nan或NA
            else:
                # 去掉无意义补零，最多6位
                f.write(f"{wl:.2f}\t{value:.8f}\n")

def unify_files_recursive(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith('.txt'):
                continue
            in_path = os.path.join(root, fname)
            relative_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, relative_path)
            with open(in_path, 'r', encoding='utf8') as f:
                first_line = f.readline().strip()
            if first_line.startswith('s07_ASD'):
                data_dict = process_asd_file(in_path, start_wavelength=350, step=1)
            else:
                data_dict = process_xy_file(in_path)
            write_unified_txt(data_dict, out_path)
            print(f"处理完成: {in_path} -> {out_path}")

# ========== 用法示例 ==========
if __name__ == "__main__":
    input_folder = "/home/Dataset/nw/spectraLib/按地类"
    output_folder = "/home/Dataset/nw/spectraLib/unify"
    unify_files_recursive(input_folder, output_folder)