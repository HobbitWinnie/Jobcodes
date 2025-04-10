import logging
from datetime import datetime
import os
import json 

# 设置日志
def setup_logging(log_dir: str = None):
    """设置日志配置"""
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if log_dir:
        import os
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"app_{datetime.now():%Y%m%d_%H%M%S}.log")
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)



def save_model_config(model, config, output_dir):  
    """保存模型配置到 JSON 文件"""  
    model_config_path = os.path.join(output_dir, 'model_config.json')  
    with open(model_config_path, 'w') as f:  
        json.dump(config, f, indent=4)  
    logging.info(f"模型配置已保存到: {model_config_path}.")  