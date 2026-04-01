import os
import json
from flask import Flask, jsonify, request, render_template, send_file
from datetime import datetime
import pandas as pd
import numpy as np
import shutil
from werkzeug.utils import secure_filename
import traceback

# 在文件顶部或相关函数中设置文件路径
# 使用相对路径，从demo目录向上两级找到results.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE_PATH = os.path.join(BASE_DIR, '..', 'PPO-HRL')

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

            
            # 3. 如果有图片URL字段，标记为图片类型
            img_url_keys = ['img_url', 'image_url', 'imgUrl', 'image', 'picture_url']
            for key in img_url_keys:
                if key in item and item[key] and item[key] != '':
                    item['imgUrl'] = item[key]  # 统一字段名
                    break
            
            parsed_data.append(item)
        
        print(f"成功解析 {len(parsed_data)} 条数据")
        return parsed_data
        
    except Exception as e:
        print(f"解析文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    

# 在 app.py 中修改 parse_archive_file 函数

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

# 在 app.py 中添加或修改以下路由
@app.route('/api/run-quality-check-and-load', methods=['POST'])
def run_quality_check_and_load():
    """执行质量检测，并加载固定的结果文件数据，同时返回质量统计数据"""
    try:
        # --- 第一部分：加载并解析 results.json 文件 ---
        # 使用相对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_file_path = os.path.join(base_dir, '..', 'PPO-HRL', 'results.json')
        
        # 或者使用绝对路径（确保路径正确）
        # results_file_path = r'D:\Desktop\北交\北交项目\hrl\PPO-HRL\results.json'
        
        print(f"尝试读取文件: {results_file_path}")
        print(f"文件是否存在: {os.path.exists(results_file_path)}")
        
        if not os.path.exists(results_file_path):
            print(f"文件不存在: {results_file_path}")
            return jsonify({
                'success': False,
                'error': f'结果数据文件未找到: {results_file_path}',
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
        
        print(f"从文件读取的原始数据: {results_data}")
        
        # --- 第二部分：从 results.json 中提取质量统计信息 ---
        # 根据您提供的results.json结构提取数据
        quality_stats = {
            'n_samples': 0,
            'n_features': 0,
            'n_classes': 0,
            'missing_rate': '0%',
            'feature_noise_ratio': '0%',
            'label_noise_ratio': '0%'
        }
        
        # 直接使用results_data中的键名，假设results_data是字典
        if isinstance(results_data, dict):
            # 检查results_data是否包含我们需要的键
            print(f"results_data的键: {list(results_data.keys())}")
            
            # 直接提取，如果没有找到键则使用默认值
            quality_stats['n_samples'] = results_data.get('n_samples', 0)
            quality_stats['n_features'] = results_data.get('n_features', 0)
            quality_stats['n_classes'] = results_data.get('n_classes', 0)
            quality_stats['missing_rate'] = results_data.get('missing_rate', '0%')
            quality_stats['feature_noise_ratio'] = results_data.get('feature_noise_ratio', '0%')
            quality_stats['label_noise_ratio'] = results_data.get('label_noise_ratio', '0%')
        
        print(f"提取的质量统计信息: {quality_stats}")
        
        # --- 第三部分：格式化数据以供前端表格显示 ---
        formatted_data = []
        if isinstance(results_data, dict):
            item = {'id': 'QUALITY_RESULTS_1', 'type': '质量报告', 'source': '固定结果文件'}
            for key, value in results_data.items():
                # 跳过已提取的统计信息字段，避免重复显示
                if key not in ['n_samples', 'n_features', 'n_classes', 'missing_rate', 'feature_noise_ratio', 'label_noise_ratio']:
                    item[key] = str(value) if value is not None else ''
            formatted_data.append(item)
        elif isinstance(results_data, list):
            for idx, row in enumerate(results_data):
                item = {'id': f'QUALITY_RESULTS_{idx+1}', 'type': '质量报告'}
                if isinstance(row, dict):
                    for k, v in row.items():
                        item[k] = str(v) if v is not None else ''
                formatted_data.append(item)
        
        # 将数据存储到结果集
        data_storage['result'] = formatted_data
        
        # --- 第四部分：返回响应，现在包含 quality_stats ---
        import random
        response_data = {
            'success': True,
            'message': '质量检测已完成，并已加载检测结果数据',
            'quality_stats': quality_stats,  # 新增：质量统计数据
            'quality_check': {  # 原有的质量检测报告（模拟）
                'suggestions': random.randint(3, 15),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'loaded_data': {  # 加载的数据信息
                'count': len(formatted_data),
                'dataset': 'result',
                'sample_fields': list(formatted_data[0].keys())[:5] if formatted_data else []
            }
        }
        
        print(f"质量检测完成。加载了 {len(formatted_data)} 条结果数据。统计信息: {quality_stats}")
        return jsonify(response_data)
        
    except json.JSONDecodeError as e:
        print(f"解析 results.json 失败: {e}")
        return jsonify({
            'success': False,
            'error': f'结果文件格式错误: {str(e)}',
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
        print(f"质量检测与数据加载过程出错: {e}")
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





# app.py 添加的新函数和路由
import subprocess
import glob
import os
from werkzeug.utils import secure_filename

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
        
        # 构建命令行参数
        if dataset_type == 'adult':
            cmd = [
                'python', 'generate_synthetic.py',
                '--dataset', 'adult',
                '--method', 'smote',
                '--n-samples', '500',
                '--seed', '42'
            ]
            output_pattern = os.path.join('..', 'PPO-HRL', 'datasets', 'synthetic', 'adult_smote.csv')
            is_image = False
        elif dataset_type == 'cifar10':
            cmd = [
                'python', 'generate_synthetic.py',
                '--dataset', 'cifar10',
                '--method', 'cifar10_mixup',
                '--n-samples', '500',
                '--seed', '42',
                '--alpha', '0.2',
                '--strategy', 'mixup'
            ]
            # CIFAR-10生成图片，命名格式为 cifar10_mixup
            output_dir = os.path.join('..', 'PPO-HRL', 'datasets', 'synthetic', 'cifar10_mixup')
            is_image = True
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported dataset type: {dataset_type}'
            }), 400
        
        # 运行生成脚本
        print(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
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
                # 遍历所有子文件夹中的图片
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                            img_path = os.path.join(root, file)
                            rel_path = os.path.relpath(img_path, '..')
                            
                            # 获取类别名（文件夹名）
                            class_name = os.path.basename(root)
                            
                            # 添加到生成数据
                            generated_data.append({
                                'original_filename': file,
                                'imgUrl': f'/{rel_path}'.replace('\\', '/'),  # 确保URL格式正确
                                'is_augmented': True,
                                'dataset': 'cifar10'
                            })
                
                print(f"从 {output_dir} 加载了 {len(generated_data)} 张增强图片")
            else:
                return jsonify({
                    'success': False,
                    'error': f'Generated image directory not found: {output_dir}'
                }), 404
        else:
            # 处理表格数据
            if os.path.exists(output_pattern):
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
                        
                        # 添加系统字段
                        item['is_augmented'] = True
                        item['dataset'] = 'adult'
                        
                        generated_data.append(item)
                    
                    print(f"从 {output_pattern} 加载了 {len(generated_data)} 条增强数据")
                except Exception as e:
                    print(f"读取CSV文件失败: {e}")
                    return jsonify({
                        'success': False,
                        'error': f'Failed to read generated CSV: {str(e)}'
                    }), 500
            else:
                # 尝试查找其他可能的文件
                synthetic_dir = os.path.join('..', 'PPO-HRL', 'datasets', 'synthetic')
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
                        'error': f'No generated CSV file found in {synthetic_dir}'
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
        base_dir = os.path.dirname(os.path.abspath(__file__))  # demo 目录
        script_path = os.path.join(base_dir, '..', 'PPO-HRL', 'train_multi_selector_v2.py')
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
        # 1. 确定 JSON 文件路径
        # 根据文档，文件在 demo/checkpoints/adult/joint_latest.json
        base_dir = os.path.dirname(os.path.abspath(__file__))  # demo 目录
        json_file_path = os.path.join(base_dir, 'checkpoints', 'adult', 'joint_latest.json')
        json_file_path = os.path.normpath(json_file_path)

        # 2. 检查文件是否存在
        if not os.path.exists(json_file_path):
            # 文件可能尚未生成，返回初始状态
            return jsonify({
                'success': True,
                'data': None,
                'message': '训练尚未开始或状态文件不存在。',
                'file_exists': False
            })

        # 3. 读取并解析 JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 4. 格式化返回数据，方便前端使用
        # 提取您关心的顶层状态信息
        status_info = {
            'iteration': data.get('iteration', 0),
            'total_iterations': data.get('total_iterations', 0),
            'reward': data.get('reward', 0.0),
            'accuracy': data.get('accuracy', 0.0),
            'best_accuracy': data.get('best_accuracy', 0.0),
            'dirty_ratio': data.get('dirty_ratio', 0.0),
            'action_distribution': data.get('action_distribution', {}),
            'timestamp': data.get('timestamp', ''),
            'file_exists': True
        }

        # 5. 提取决策表格所需数据
        # 决策表格通常展示最近几步（steps）的详细样本操作
        decision_table_data = []
        steps = data.get('steps', [])
        for step in steps[-5:]:  # 取最近的5步用于展示
            step_info = {
                'step': step.get('step'),
                'action': step.get('action'),
                'reward': step.get('reward'),
                'n_selected': step.get('n_selected'),
                'selected_noise_count': step.get('selected_noise_count'),
                'accuracy': step.get('accuracy'),
                'dirty_ratio': step.get('dirty_ratio'),
                'samples': []  # 用于存放当前步的样本决策
            }
            samples = step.get('samples', [])
            for sample in samples[:20]:  # 每步最多取20个样本展示，避免数据量过大
                sample_data = {
                    'sample_id': sample.get('sample_id'),
                    'action': sample.get('action'),
                    'confidence': sample.get('confidence'),
                    'original_label': sample.get('original_label'),
                    'data_type': sample.get('data_type'),
                }
                # 根据动作类型添加特定字段
                if sample.get('action') == 'modify_labels':
                    sample_data['predicted_label'] = sample.get('predicted_label')
                    sample_data['label_changed'] = sample.get('label_changed')
                elif sample.get('action') == 'modify_features':
                    # 对于修改特征，原始特征值可能很长，可以只取摘要或第一个值
                    original_features = sample.get('original_features', [])
                    sample_data['original_features_preview'] = str(original_features[:3]) + '...' if len(original_features) > 3 else str(original_features)
                    sample_data['features_changed'] = sample.get('features_changed')
                elif sample.get('action') in ['add_samples', 'delete_samples']:
                    sample_data['from_pool'] = sample.get('from_pool', False)
                step_info['samples'].append(sample_data)
            decision_table_data.append(step_info)

        return jsonify({
            'success': True,
            'status': status_info,
            'decision_table': decision_table_data,
            'message': '状态获取成功。'
        })

    except json.JSONDecodeError as e:
        return jsonify({'success': False, 'error': f'JSON文件解析错误: {str(e)}'}), 500
    except Exception as e:
        print(f"读取训练状态时出错: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)




