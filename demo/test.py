import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import shutil
import subprocess



def run_data_cleaning():
    """启动数据清洗训练脚本，并开启一个后台线程监控其状态"""
    # 1. 确定脚本路径
    # 工作目录是 /home/extra_home/PPO-HRL/demo
    # 脚本在 /home/extra_home/PPO-HRL/train_multi_selector_v2.py
    base_dir = os.path.dirname(os.path.abspath(__file__))  # demo 目录
    script_path = os.path.join(base_dir, '..', 'PPO-HRL', 'train_multi_selector_v2.py')
    script_path = os.path.normpath(script_path)  # 规范化路径


    # 3. 构建命令
    cmd = ['python', script_path, '--dataset', 'adult', '--interactive']
    # 工作目录设置为 PPO-HRL 目录，确保相对路径正确
    cwd = os.path.dirname(script_path)

    # 4. 使用子进程异步启动脚本
    # 使用 Popen 并重定向输出到日志文件，避免阻塞
    log_path = open(os.path.join(cwd, 'training.log'), 'w')
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=log_path,
        stderr=subprocess.STDOUT,  # 将错误输出也重定向到日志
        shell=False
    )
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"\n📋 日志前几行:")
                for line in lines[-5:]:  # 显示最后5行
                    print(f"   {line.strip()}")
            else:
                print("📭 日志文件为空，脚本可能仍在初始化")

 

run_data_cleaning()