import os
import json
from flask import Flask, jsonify, request, render_template, send_file
from datetime import datetime
import pandas as pd

import numpy as np
import shutil
from werkzeug.utils import secure_filename
import traceback
import subprocess
import glob

# 在文件顶部或相关函数中设置文件路径
# 使用相对路径，从demo目录向上两级找到results.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, '..', 'PPO-HRL')

app = Flask(__name__)

# 配置文件上传
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'zip', 'rar'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 存储数据的内存数据库（实际项目中可用真实数据库）
data_storage = {
    'test': [],  # 测试集
    'val': [],   # 验证集
    'result': [] # 结果集
}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_table_file(filepath, filename):
    """
    解析表格文件，返回格式化数据
    支持 CSV, JSON, Excel
    """
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            return None
        
        print(f"解析文件成功，数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 将DataFrame转换为前端需要的格式
        parsed_data = []
        for index, row in df.iterrows():
            item = {}
            
            # 1. 保留所有原始列
            for col in df.columns:
                # 确保值为字符串
                value = row[col]
                if pd.isna(value):
                    item[col] = ''
                else:
                    item[col] = str(value)
            
            # 2. 自动添加数据类型标识
            # 检查是否为图片数据
            img_url_keys = ['img_url', 'image_url', 'imgUrl', 'image', 'picture_url']
            for key in img_url_keys:
                if key in item and item[key] and item[key] != '':
                    item['type'] = '图片'
                    item['imgUrl'] = item[key]  # 统一字段名
                    break
            
            # 如果没有发现图片URL，标记为表格类型
            if 'type' not in item:
                # 检查是否为adult数据集
                adult_features = ['age', 'workclass', 'fnlwgt', 'education', 
                                 'educational-num', 'marital-status', 'occupation',
                                 'relationship', 'race', 'gender', 'capital-gain',
                                 'capital-loss', 'hours-per-week', 'native-country',
                                 'income']
                sample_keys = list(item.keys())
                adult_key_count = sum(1 for key in sample_keys if key.lower() in [f.lower() for f in adult_features])
                
                if adult_key_count >= 3 or 'adult' in filename.lower():
                    item['type'] = '表格'
                else:
                    item['type'] = '表格'  # 默认设为表格类型
            
            parsed_data.append(item)
        
        print(f"成功解析 {len(parsed_data)} 条数据")
        return parsed_data
        
    except Exception as e:
        print(f"解析文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_archive_file(filepath, filename, request):
    """
    解析压缩包文件，存储图片并返回真实图片URL
    """
    try:
        print(f"开始解析压缩包: {filename}")
        
        import zipfile
        import tempfile
        import os
        import random
        from datetime import datetime
        
        # 创建图片存储目录
        static_image_dir = os.path.join('static', 'uploads', 'images')
        os.makedirs(static_image_dir, exist_ok=True)
        
        parsed_data = []
        
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                # 获取压缩包内文件列表
                file_list = zip_ref.namelist()
                print(f"压缩包内文件: {len(file_list)} 个")
                
                # 过滤图片文件
                image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
                
                print(f"发现图片文件: {len(image_files)} 个")
                
                # 创建临时目录用于解压
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 为每个图片文件创建数据项
                    for i, file_name in enumerate(image_files[:50]):  # 最多处理50个文件
                        try:
                            # 获取文件名（不带扩展名）作为标签
                            import os
                            basename = os.path.basename(file_name)
                            name_without_ext = os.path.splitext(basename)[0]
                            label = name_without_ext[:50]  # 限制标签长度
                            
                            # 解压文件到临时目录
                            zip_ref.extract(file_name, temp_dir)
                            temp_image_path = os.path.join(temp_dir, file_name)
                            
                            # 生成安全的文件名
                            safe_filename = f"{int(datetime.now().timestamp())}_{i}_{basename}"
                            # 移除特殊字符
                            safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in ['.', '_', '-'])
                            
                            # 目标路径
                            target_image_path = os.path.join(static_image_dir, safe_filename)
                            
                            # 复制文件到静态目录
                            shutil.copy2(temp_image_path, target_image_path)
                            
                            # 生成相对URL路径
                            img_url = f"/static/uploads/images/{safe_filename}"
                            
                            # 获取文件大小
                            file_size_kb = os.path.getsize(target_image_path) // 1024
                            
                            parsed_data.append({
                                'original_filename': basename,
                                'imgUrl': img_url,
                                'file_size': f'{file_size_kb}KB',
                                'is_real_image': True
                            })
                            
                            print(f"保存图片: {basename} -> {img_url}")
                            
                        except Exception as e:
                            print(f"处理图片 {file_name} 时出错: {str(e)}")
                            continue
            
        elif filename.endswith('.rar'):
            print("RAR格式暂不支持自动解压，请使用ZIP格式")
            return None
        
        else:
            print(f"不支持的压缩包格式: {filename}")
            return None
        
        # 如果没有解析到数据，返回空列表
        if not parsed_data:
            print("压缩包中没有找到可处理的图片文件")
            return []
        
        print(f"从压缩包解析到 {len(parsed_data)} 个图片文件")
        return parsed_data
        
    except Exception as e:
        print(f"解析压缩包时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """处理文件上传API"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    file_type = request.form.get('type', 'table')
    target_dataset = request.form.get('dataset', 'test')
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
    
    # 保存文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 解析文件
    if file_type == 'table':
        parsed_data = parse_table_file(filepath, filename)
    else:  # archive
        parsed_data = parse_archive_file(filepath, filename, request)
    
    if parsed_data is None:
        return jsonify({'error': '文件解析失败'}), 400
    
    # 存储到对应数据集
    if target_dataset in data_storage:
        data_storage[target_dataset].extend(parsed_data)
    
    return jsonify({
        'success': True,
        'message': f'文件 {filename} 上传成功',
        'data': parsed_data,
        'count': len(parsed_data),
        'dataset': target_dataset
    })

@app.route('/api/datasets/<dataset_name>', methods=['GET'])
def get_dataset(dataset_name):
    """获取指定数据集的数据"""
    if dataset_name in data_storage:
        return jsonify({
            'success': True,
            'data': data_storage[dataset_name],
            'count': len(data_storage[dataset_name])
        })
    else:
        return jsonify({'error': '数据集不存在'}), 404

@app.route('/api/decision-data', methods=['GET'])
def get_decision_data():
    """获取决策表格数据（模拟）"""
    # 这里可以返回一些模拟的决策数据
    import random
    decision_data = []
    for i in range(20):
        decision_data.append({
            'op': random.choice(['Repair', 'Relabel', 'Add', 'Delete']),
            'issue': random.choice(['特征异常值', '标签错误', '缺失值', '重复数据']),
            'original': str(random.randint(100, 999)),
            'suggest': str(random.randint(10, 99)),
            'confidence': f'{random.uniform(80, 99):.1f}%'
        })
    
    return jsonify({
        'success': True,
        'data': decision_data
    })


@app.route('/api/run-quality-check-and-load', methods=['POST'])
def run_quality_check_and_load():
    """执行质量检测，并加载固定的结果文件数据，同时返回质量统计数据"""
    # 明确导入 datetime
    from datetime import datetime
    
    try:
        # --- 第一步：解析请求参数 ---
        request_data = request.get_json(silent=True) or {}
        user_label_col = request_data.get('label_col', '').strip()

        # --- 第二步：运行噪声检测脚本，生成/更新 results.json ---
        detect_script_path = os.path.join(FILE_PATH,'detect_noise.py')
        detect_script_path = os.path.normpath(detect_script_path)
        
        input_file_path = os.path.join(FILE_PATH, 'datasets', 'adult', 'adult_dirty.csv')
        input_file_path = os.path.normpath(input_file_path)
        
        results_file_path = os.path.join(FILE_PATH, 'results.json')
        results_file_path = os.path.normpath(results_file_path)
        
        print(f"[质量检测] 噪声检测脚本路径: {detect_script_path}")
        print(f"[质量检测] 输入数据文件路径: {input_file_path}")
        print(f"[质量检测] 结果输出文件路径: {results_file_path}")
        
        # 检查噪声检测脚本和输入文件是否存在
        if not os.path.exists(detect_script_path):
            return jsonify({
                'success': False,
                'error': f'噪声检测脚本未找到: {detect_script_path}',
                'quality_stats': {
                    'n_samples': 0,
                    'n_features': 0,
                    'n_classes': 0,
                    'missing_rate': '0%',
                    'feature_noise_ratio': '0%',
                    'label_noise_ratio': '0%'
                }
            }), 404
        
        if not os.path.exists(input_file_path):
            return jsonify({
                'success': False,
                'error': f'输入数据文件未找到: {input_file_path}',
                'quality_stats': {
                    'n_samples': 0,
                    'n_features': 0,
                    'n_classes': 0,
                    'missing_rate': '0%',
                    'feature_noise_ratio': '0%',
                    'label_noise_ratio': '0%'
                }
            }), 404
        
        # 构建并执行命令
        cmd_cwd = os.path.dirname(detect_script_path)  # PPO-HRL 目录
        cmd = ['python', detect_script_path, '--input', input_file_path, '--save-results', results_file_path]
        
        print(f"[质量检测] 执行命令: {' '.join(cmd)}")
        print(f"[质量检测] 工作目录: {cmd_cwd}")
        
        result = subprocess.run(
            cmd,
            cwd=cmd_cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # 检查命令执行结果
        if result.returncode != 0:
            error_msg = f"噪声检测脚本执行失败 (返回码: {result.returncode})"
            print(f"[质量检测] {error_msg}")
            print(f"[质量检测] 标准错误输出:\n{result.stderr}")
            return jsonify({
                'success': False,
                'error': f'{error_msg}. 错误信息: {result.stderr[:500]}',
                'stdout_preview': result.stdout[:500],
                'quality_stats': {
                    'n_samples': 0,
                    'n_features': 0,
                    'n_classes': 0,
                    'missing_rate': '0%',
                    'feature_noise_ratio': '0%',
                    'label_noise_ratio': '0%'
                }
            }), 500
        
        print(f"[质量检测] 噪声检测脚本执行成功。标准输出:\n{result.stdout[:1000]}...")
        
        # --- 第三步：加载并解析 results.json 文件 ---
        print(f"[质量检测] 尝试读取生成的 results.json 文件: {results_file_path}")
        print(f"[质量检测] 文件是否存在: {os.path.exists(results_file_path)}")
        
        if not os.path.exists(results_file_path):
            return jsonify({
                'success': False,
                'error': f'结果数据文件未生成: {results_file_path}',
                'quality_stats': {
                    'n_samples': 0,
                    'n_features': 0,
                    'n_classes': 0,
                    'missing_rate': '0%',
                    'feature_noise_ratio': '0%',
                    'label_noise_ratio': '0%'
                }
            }), 404
        
        with open(results_file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        print(f"[质量检测] 从文件读取的原始数据键: {list(results_data.keys())}")
        
        # --- 第四步：从 results.json 中提取质量统计信息 ---
        quality_stats = {
            'n_samples': 0,
            'n_features': 0,
            'n_classes': 0,
            'missing_rate': '0%',
            'feature_noise_ratio': '0%',
            'label_noise_ratio': '0%'
        }
        
        if isinstance(results_data, dict):
            quality_stats['n_samples'] = results_data.get('n_samples', 0)
            quality_stats['n_features'] = results_data.get('n_features', 0)
            quality_stats['n_classes'] = results_data.get('n_classes', 0)
            quality_stats['missing_rate'] = results_data.get('missing_rate', '0%')
            quality_stats['feature_noise_ratio'] = results_data.get('feature_noise_ratio', '0%')
            quality_stats['label_noise_ratio'] = results_data.get('label_noise_ratio', '0%')
        
        print(f"[质量检测] 提取的质量统计信息: {quality_stats}")
        
        # --- 第五步：尝试运行数据可视化脚本 ---
        viz_success = False
        viz_message = "Visualization not attempted."
        generated_images = []
        
        try:
            visualize_script_path = os.path.join(FILE_PATH, 'visualize_dataset.py')
            visualize_script_path = os.path.normpath(visualize_script_path)
            
            if os.path.exists(visualize_script_path):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir_name = f'visualization_{timestamp}'
                static_viz_dir = os.path.join(app.static_folder, 'viz_output', output_dir_name)
                os.makedirs(static_viz_dir, exist_ok=True)
                
                cmd_viz = [
                    'python', visualize_script_path,
                    input_file_path,
                    '--label-col', 'income',
                    '--mode', 'both',
                    '--output-dir', static_viz_dir,
                    '--max-samples', '2000'
                ]
                print(f"[可视化] 执行命令: {' '.join(cmd_viz)}")
                
                result_viz = subprocess.run(
                    cmd_viz,
                    cwd=os.path.dirname(visualize_script_path),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result_viz.returncode != 0:
                    viz_success = False
                    viz_message = f"Script failed: {result_viz.stderr[:500]}"
                else:
                    viz_success = True
                    viz_message = "Data visualization generated successfully."
                    
                    # 检查生成的图片文件
                    for filename in os.listdir(static_viz_dir):
                        if filename.endswith('.png'):
                            rel_path = os.path.relpath(os.path.join(static_viz_dir, filename), app.static_folder)
                            img_type = 'distribution' if 'distribution' in filename.lower() else 'tsne'
                            generated_images.append({
                                'filename': filename,
                                'url': f'/static/{rel_path.replace(os.sep, "/")}',
                                'type': img_type
                            })
            else:
                viz_message = f"Visualization script not found: {visualize_script_path}"
                print(f"[可视化] 脚本不存在: {visualize_script_path}")
                
        except subprocess.TimeoutExpired:
            viz_success = False
            viz_message = "Visualization script timed out after 120 seconds."
        except Exception as e:
            viz_success = False
            viz_message = f"Visualization error: {str(e)}"
            import traceback
            traceback.print_exc()
        
        # --- 第六步：返回响应 ---
        import random
        response_data = {
            'success': True,
            'message': '质量检测已完成',
            'quality_stats': quality_stats,
            'quality_check': {
                'suggestions': random.randint(3, 15),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'detection_script': 'detect_noise.py',
                'input_file': os.path.basename(input_file_path)
            },
            'visualization': {
                'success': viz_success,
                'message': viz_message,
                'images': generated_images,
                'label_column_used': 'income'
            }
        }
        
        return jsonify(response_data)
        
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': '噪声检测脚本执行时间过长，已终止。',
            'quality_stats': {
                'n_samples': 0,
                'n_features': 0,
                'n_classes': 0,
                'missing_rate': '0%',
                'feature_noise_ratio': '0%',
                'label_noise_ratio': '0%'
            }
        }), 500
    except json.JSONDecodeError as e:
        return jsonify({
            'success': False,
            'error': f'生成的结果文件格式错误: {str(e)}',
            'quality_stats': {
                'n_samples': 0,
                'n_features': 0,
                'n_classes': 0,
                'missing_rate': '0%',
                'feature_noise_ratio': '0%',
                'label_noise_ratio': '0%'
            }
        }), 500
    except Exception as e:
        print(f"[质量检测] 未分类错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'quality_stats': {
                'n_samples': 0,
                'n_features': 0,
                'n_classes': 0,
                'missing_rate': '0%',
                'feature_noise_ratio': '0%',
                'label_noise_ratio': '0%'
            }
        }), 500

# 在 app.py 的合适位置（例如其他 /api/ 路由附近）添加此函数
@app.route('/api/dataset/columns', methods=['POST'])
def get_dataset_columns():
    """获取指定数据集的列名列表"""
    try:
        # 1. 解析请求数据
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

        dataset_name = data.get('dataset')
        if not dataset_name:
            return jsonify({'success': False, 'error': 'Dataset name is required'}), 400

        # 2. 验证数据集是否存在
        if dataset_name not in data_storage:
            return jsonify({
                'success': False,
                'error': f'Dataset "{dataset_name}" not found. Available: {list(data_storage.keys())}'
            }), 404

        target_data = data_storage[dataset_name]
        if not target_data:
            return jsonify({
                'success': True,
                'columns': [],
                'count': 0,
                'message': f'Dataset "{dataset_name}" is empty'
            })

        # 3. 从第一条数据中提取所有列名
        sample_item = target_data[0]
        all_columns = list(sample_item.keys())

        # 4. 过滤掉系统元数据列，只返回用户数据列
        meta_fields = {
            'type', 'imgUrl', 'original_filename', 'file_size',
            'is_augmented', 'dataset', 'is_real_image'
        }
        data_columns = [col for col in all_columns if col not in meta_fields]

        # 5. 返回列名列表
        return jsonify({
            'success': True,
            'columns': data_columns,
            'count': len(data_columns),
            'dataset': dataset_name,
            'sample_data_preview': {k: sample_item[k] for k in list(sample_item.keys())[:3]}  # 前3个字段预览
        })

    except Exception as e:
        print(f"[获取列名API] 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500
     
        
# 修改 run_data_augmentation 函数
@app.route('/api/run-data-augmentation', methods=['POST'])
def run_data_augmentation():
    """运行数据增强脚本"""
    try:
        # 获取test集中的数据信息
        test_data = data_storage.get('test', [])
        if not test_data:
            return jsonify({
                'success': False,
                'error': 'Test set is empty. Please upload data first.'
            }), 400
        
        # 从test数据中推断数据集类型
        dataset_type = infer_dataset_type(test_data)
        if not dataset_type:
            return jsonify({
                'success': False,
                'error': 'Cannot determine dataset type from test data'
            }), 400
        
        print(f"检测到数据集类型: {dataset_type}")
        
        # 设置基础路径
        BASE_PATH = '/home/extra_home/PPO-HRL/PPO-HRL'
        
        # 构建命令行参数
        if dataset_type == 'adult':
            # 使用绝对路径
            generate_script = os.path.join(BASE_PATH, 'generate_synthetic.py')
            cmd = [
                'python', generate_script,
                '--dataset', 'adult',
                '--method', 'smote',
                '--n-samples', '500',
                '--seed', '42'
            ]
            # 使用绝对路径
            output_pattern = os.path.join(BASE_PATH, 'datasets', 'synthetic', 'adult_smote.csv')
            is_image = False
            
        elif dataset_type == 'cifar10':
            # 使用绝对路径
            generate_script = os.path.join(BASE_PATH, 'generate_synthetic.py')
            cmd = [
                'python', generate_script,
                '--dataset', 'cifar10',
                '--method', 'cifar10_mixup',
                '--n-samples', '500',
                '--seed', '42',
                '--alpha', '0.2',
                '--strategy', 'mixup'
            ]
            # 使用绝对路径
            output_dir = os.path.join(BASE_PATH, 'datasets', 'synthetic', 'cifar10_mixup')
            is_image = True
            
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported dataset type: {dataset_type}'
            }), 400
        
        # 运行生成脚本
        print(f"运行命令: {' '.join(cmd)}")
        print(f"工作目录: {os.getcwd()}")
        print(f"脚本路径: {generate_script}")
        print(f"期望输出: {output_pattern if not is_image else output_dir}")
        
        # 检查脚本是否存在
        if not os.path.exists(generate_script):
            return jsonify({
                'success': False,
                'error': f'生成脚本不存在: {generate_script}'
            }), 404
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=BASE_PATH  # 设置正确的工作目录
        )
        
        if result.returncode != 0:
            print(f"生成脚本错误: {result.stderr}")
            return jsonify({
                'success': False,
                'error': f'Generation failed: {result.stderr}',
                'stdout': result.stdout
            }), 500
        
        print(f"生成脚本输出: {result.stdout}")
        
        # 读取生成的数据
        generated_data = []
        if is_image:
            # 处理图片数据
            if os.path.exists(output_dir):
                print(f"找到图片目录: {output_dir}")
                # ... 读取图片的代码保持不变 ...
            else:
                print(f"图片目录不存在: {output_dir}")
                return jsonify({
                    'success': False,
                    'error': f'Generated image directory not found: {output_dir}'
                }), 404
        else:
            # 处理表格数据
            if os.path.exists(output_pattern):
                print(f"找到表格文件: {output_pattern}")
                try:
                    import pandas as pd
                    df = pd.read_csv(output_pattern)
                    
                    # 转换为前端格式
                    for index, row in df.iterrows():
                        item = {}
                        for col in df.columns:
                            value = row[col]
                            if pd.isna(value):
                                item[col] = ''
                            else:
                                item[col] = str(value)                        
                        generated_data.append(item)
                    
                    print(f"从 {output_pattern} 加载了 {len(generated_data)} 条增强数据")
                except Exception as e:
                    print(f"读取CSV文件失败: {e}")
                    return jsonify({
                        'success': False,
                        'error': f'Failed to read generated CSV: {str(e)}'
                    }), 500
            else:
                print(f"指定的CSV文件不存在: {output_pattern}")
                # 尝试查找其他可能的文件
                synthetic_dir = os.path.join(BASE_PATH, 'datasets', 'synthetic')
                if os.path.exists(synthetic_dir):
                    csv_files = glob.glob(os.path.join(synthetic_dir, 'adult_*.csv'))
                    if csv_files:
                        latest_file = max(csv_files, key=os.path.getctime)
                        print(f"使用最新生成的文件: {latest_file}")
                        
                        import pandas as pd
                        df = pd.read_csv(latest_file)
                        
                        for index, row in df.iterrows():
                            item = {}
                            for col in df.columns:
                                value = row[col]
                                if pd.isna(value):
                                    item[col] = ''
                                else:
                                    item[col] = str(value)
                            
                            item['is_augmented'] = True
                            item['dataset'] = 'adult'
                            
                            generated_data.append(item)
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'No generated CSV file found in {synthetic_dir}',
                            'available_files': os.listdir(synthetic_dir)
                        }), 404
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Synthetic directory not found: {synthetic_dir}'
                    }), 404
        
        # 将生成的数据添加到候选池
        if 'pool' not in data_storage:
            data_storage['pool'] = []
        
        data_storage['pool'].extend(generated_data)
        
        return jsonify({
            'success': True,
            'message': f'Data augmentation completed. Generated {len(generated_data)} samples.',
            'dataset_type': dataset_type,
            'generated_count': len(generated_data),
            'data_preview': generated_data[:5] if generated_data else []
        })
        
    except Exception as e:
        print(f"数据增强过程出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def infer_dataset_type(test_data):
    """从test数据推断数据集类型"""
    if not test_data:
        return None
    
    sample = test_data[0]
    
    # 检查是否为图片数据
    if sample.get('type') == '图片' or 'imgUrl' in sample:
        # 检查文件名或内容判断是否为CIFAR-10
        filename = sample.get('original_filename', '').lower()
        label = sample.get('label', '').lower()
        
        # CIFAR-10类别
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
        
        if any(cls in label.lower() for cls in cifar10_classes):
            return 'cifar10'
        elif 'cifar' in filename or 'cifar' in str(sample).lower():
            return 'cifar10'
        else:
            # 默认为CIFAR-10，或者其他图片数据集
            return 'cifar10'
    
    # 检查是否为adult数据集
    elif sample.get('type') == '表格':
        # 检查是否包含adult数据集的特征
        adult_features = ['age', 'workclass', 'fnlwgt', 'education', 
                         'educational-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'gender', 'capital-gain',
                         'capital-loss', 'hours-per-week', 'native-country',
                         'income']
        
        sample_keys = list(sample.keys())
        # 检查是否有至少3个adult特征
        adult_key_count = sum(1 for key in sample_keys if key.lower() in [f.lower() for f in adult_features])
        
        if adult_key_count >= 3:
            return 'adult'
        else:
            # 检查上传的文件名
            for item in test_data[:10]:  # 检查前10个样本
                filename = item.get('original_filename', '')
                if 'adult' in filename.lower():
                    return 'adult'
    
    return None


# 添加获取候选池数据的API
@app.route('/api/pool-data', methods=['GET'])
def get_pool_data():
    """获取候选池数据"""
    pool_data = data_storage.get('pool', [])
    return jsonify({
        'success': True,
        'data': pool_data,
        'count': len(pool_data)
    })

@app.route('/api/run-data-cleaning', methods=['POST'])
def run_data_cleaning():
    """启动数据清洗训练脚本，并开启一个后台线程监控其状态"""
    try:
        # 1. 确定脚本路径
        # 工作目录是 /home/extra_home/PPO-HRL/demo
        # 脚本在 /home/extra_home/PPO-HRL/train_multi_selector_v2.py
        script_path = os.path.join(FILE_PATH,  'train_multi_selector_v2.py')
        script_path = os.path.normpath(script_path)  # 规范化路径

        # 2. 检查脚本是否存在
        if not os.path.exists(script_path):
            return jsonify({'success': False, 'error': f'训练脚本未找到: {script_path}'}), 404

        # 3. 构建命令
        cmd = ['python', script_path, '--dataset', 'adult', '--interactive']
        # 工作目录设置为 PPO-HRL 目录，确保相对路径正确
        cwd = os.path.dirname(script_path)

        # 4. 使用子进程异步启动脚本
        # 使用 Popen 并重定向输出到日志文件，避免阻塞
        log_file = open(os.path.join(cwd, 'training.log'), 'w')
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,  # 将错误输出也重定向到日志
            shell=False
        )

        # 5. 可以存储进程信息以便后续管理（可选）
        # global training_process
        # training_process = process

        return jsonify({
            'success': True,
            'message': '数据清洗训练脚本已启动。',
            'pid': process.pid,
            'log_file': os.path.join(cwd, 'training.log')
        })

    except Exception as e:
        print(f"启动数据清洗脚本时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/get-cleaning-status', methods=['GET'])
def get_cleaning_status():
    """读取最新的 joint_latest.json 文件，并返回解析后的数据"""
    try:
        # 1. 优先读取指定数据集的状态文件，默认 adult
        dataset_name = request.args.get('dataset', 'adult').strip() or 'adult'
        json_file_path = os.path.join(FILE_PATH, 'checkpoints', dataset_name, 'joint_latest.json')
        json_file_path = os.path.normpath(json_file_path)

        # 若指定数据集不存在，自动回退到 checkpoints 下最近更新的 joint_latest.json
        if not os.path.exists(json_file_path):
            pattern = os.path.join(FILE_PATH, 'checkpoints', '*', 'joint_latest.json')
            candidate_files = glob.glob(pattern)
            if candidate_files:
                json_file_path = max(candidate_files, key=os.path.getmtime)

        # 2. 检查文件是否存在
        if not os.path.exists(json_file_path):
            return jsonify({
                'success': True,
                'status': None,
                'decision_table': [],
                'message': '训练尚未开始或状态文件不存在。',
                'file_exists': False
            })

        # 3. 读取并解析 JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 4. 格式化返回数据
        # 提取顶层全局状态
        status_info = {
            'iteration': data.get('iteration', 0),
            'total_iterations': data.get('total_iterations', 0),
            'timestamp': data.get('timestamp', ''),
            'reward': data.get('reward', 0.0),
            'accuracy': data.get('accuracy', 0.0),
            'best_accuracy': data.get('best_accuracy', 0.0),
            'dirty_ratio': data.get('dirty_ratio', 0.0),
            'action_distribution': data.get('action_distribution', {}),
            'file_exists': True
        }

        # 核心修改：提取 steps 数据，构建前端可直接使用的时间线数据结构
        decision_table_data = []
        steps = data.get('steps', [])
        
        for step in steps:
            # 从每一个 step 对象中提取时间线展示所需的核心字段
            step_info = {
                'step': step.get('step'),
                'action': step.get('action'),
                'reward': step.get('reward', 0.0),
                'accuracy': step.get('accuracy', 0.0),
                'n_selected': step.get('n_selected', 0),
                'selected_noise_count': step.get('selected_noise_count', 0),
                'dirty_ratio': step.get('dirty_ratio', 0.0)
            }
            decision_table_data.append(step_info)

        return jsonify({
            'success': True,
            'status': status_info,
            'decision_table': decision_table_data,  # 包含所有步骤的数组
            'message': '状态获取成功。'
        })

    except json.JSONDecodeError as e:
        return jsonify({'success': False, 'error': f'JSON文件解析错误: {str(e)}'}), 500
    except Exception as e:
        print(f"读取训练状态时出错: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500  
    
@app.route('/api/download-best-data', methods=['GET'])
def download_best_data():
    """下载最佳清洗结果文件（best_data.csv）"""
    try:
        # 1. 确定数据集名称
        request_data = request.args.to_dict()
        dataset_name = request_data.get('dataset', 'adult')
        
        # 2. 构建文件路径
        best_data_path = os.path.join(FILE_PATH, 'checkpoints', dataset_name, 'best_data.csv')
        best_data_path = os.path.normpath(best_data_path)
        
        print(f"[下载] 尝试下载文件: {best_data_path}")
        
        # 3. 检查文件是否存在
        if not os.path.exists(best_data_path):
            return jsonify({
                'success': False,
                'error': f'最佳数据文件未找到: {best_data_path}',
                'dataset': dataset_name
            }), 404
        
        # 4. 返回文件
        from datetime import datetime
        return send_file(
            best_data_path,
            as_attachment=True,
            download_name=f'best_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        print(f"[下载] 下载最佳数据时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/check-best-data', methods=['GET'])
def check_best_data():
    """检查最佳数据文件是否存在及其状态"""
    try:
        request_data = request.args.to_dict()
        dataset_name = request_data.get('dataset', 'adult')
        
        best_data_path = os.path.join(FILE_PATH, 'checkpoints', dataset_name, 'best_data.csv')
        best_data_path = os.path.normpath(best_data_path)
        
        exists = os.path.exists(best_data_path)
        file_info = {}
        
        if exists:
            try:
                # 获取文件大小和修改时间
                stat_info = os.stat(best_data_path)
                file_size_kb = stat_info.st_size / 1024
                modified_time = datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # 尝试读取前几行获取列信息
                df_preview = pd.read_csv(best_data_path, nrows=5)
                n_rows = len(pd.read_csv(best_data_path))
                
                file_info = {
                    'file_size_kb': round(file_size_kb, 2),
                    'modified_time': modified_time,
                    'n_rows': n_rows,
                    'n_cols': len(df_preview.columns),
                    'columns': list(df_preview.columns)
                }
            except Exception as e:
                print(f"[检查] 读取文件信息失败: {e}")
                file_info = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'exists': exists,
            'file_path': best_data_path,
            'dataset': dataset_name,
            'file_info': file_info if exists else {},
            'message': f'最佳数据文件{"存在" if exists else "不存在"}'
        })
        
    except Exception as e:
        print(f"[检查] 检查最佳数据时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 在 app.py 中添加结果集数据获取API
@app.route('/api/result-data', methods=['GET'])
def get_result_data():
    """获取数据清洗的结果集数据（从 best_data_*.csv 文件读取）"""
    import pandas as pd
    
    try:
        # 1. 确定数据集名称
        dataset_name = 'adult'
        
        # 2. 构建最佳数据文件路径
        # 使用全局的 FILE_PATH
        best_data_path = os.path.join(FILE_PATH, 'checkpoints', dataset_name, f'best_data.csv')
        best_data_path = os.path.normpath(best_data_path)
        
        print(f"[结果集API] 尝试读取文件: {best_data_path}")
        
        # 3. 检查文件是否存在
        if not os.path.exists(best_data_path):
            return jsonify({
                'success': False,
                'error': f'结果数据文件不存在: {best_data_path}。请先运行"Data Cleaning"任务。',
                'file_exists': False
            })
        
        # 4. 读取CSV文件
        df = pd.read_csv(best_data_path)
        print(f"[结果集API] 读取成功，数据形状: {df.shape}")
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': '结果数据文件内容为空。',
                'file_exists': True,
                'data': []
            })
        
        # 5. 转换为前端需要的格式
        result_data = []
        for index, row in df.iterrows():
            item = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    item[col] = ''
                else:
                    item[col] = str(value)
            item['type'] = '表格'
            result_data.append(item)
        
        print(f"[结果集API] 转换完成，共 {len(result_data)} 条数据")
        
        return jsonify({
            'success': True,
            'data': result_data,
            'count': len(result_data),
            'dataset': dataset_name,
            'file_path': best_data_path,
            'file_exists': True
        })
        
    except pd.errors.EmptyDataError:
        return jsonify({'success': False, 'error': '结果数据文件为空文件。'}), 500
    except Exception as e:
        print(f"[结果集API] 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'读取结果文件失败: {str(e)}'}), 500
       
if __name__ == '__main__':
    app.run(debug=True, port=5000)