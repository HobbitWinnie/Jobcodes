import logging
from datetime import datetime
import os
import json 

# 设置日志
def setup_logging(log_dir: str = None, level: int = logging.INFO, log_to_console: bool = True):
    """  
    设置标准日志输出，支持保存到文件与控制台。  

    Args:  
        log_dir (str): 日志文件夹，如 None 则只控制台日志。  
        level (int): 日志等级，默认 logging.INFO。  
        log_to_console (bool): 是否输出到终端。  
    """  
    log_format = '%(asctime)s - %(levelname)s - %(message)s'    

    # 清空之前的句柄，确保不重复  
    for handler in logging.root.handlers[:]:  
        logging.root.removeHandler(handler)  
    
    handlers = []  

    if log_dir:
        import os
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"app_{datetime.now():%Y%m%d_%H%M%S}.log")  
        file_handler = logging.FileHandler(file_path)  
        file_handler.setFormatter(logging.Formatter(log_format))  
        handlers.append(file_handler)  

    if log_to_console:  
        stream_handler = logging.StreamHandler()  
        stream_handler.setFormatter(logging.Formatter(log_format))  
        handlers.append(stream_handler)  

    logging.basicConfig(  
        level=level,  
        handlers=handlers  
    )  


def save_model_config(config, output_dir):  
    """保存模型配置到 JSON 文件"""  
    model_config_path = os.path.join(output_dir, 'model_config.json')  
    with open(model_config_path, 'w') as f:  
        json.dump(config, f, indent=4)  
    logging.info(f"模型配置已保存到: {model_config_path}.")  