// demo/static/js/main.js
// 全局图表实例
let dataChart = null;
let currentChartType = 'class';
let currentUploadType = 'table';
let isAutoProcess = false;

// 数据集管理
let testSetData = [];
let valSetData = [];
let resultSetData = [];
let currentDisplayDataset = 'test';
let currentTargetDataset = 'test';
let currentTargetColumn = '';


// 分页配置
const uploadPageConfig = {
    currentPage: 1,
    pageSize: 15,
    totalPages: 1,
    totalItems: 0
};

const decisionPageConfig = {
    currentPage: 1,
    pageSize: 12,
    totalPages: 1,
    totalItems: 0
};

// ****************** 新增：数据清洗任务相关全局变量 ******************
let cleaningStatusTimer = null; // 轮询定时器
let isCleaningRunning = false;  // 标记训练是否正在进行
let pendingDecisionData = null; // pending_decision.json 当前内容
let timelineCollapsedIterations = {}; // 时间线折叠状态
let lastTimelineIteration = null; // 用于控制是否自动滚动
// ****************************************************************

// 页面加载初始化
window.onload = function() {
    initChart();
    initUploadTablePagination();
    initDecisionTablePagination();
    changeUploadPage(1);
    changeDecisionPage(1);
    renderDisplayData();
    updateStatistics(); // 页面加载时即从 results.json 获取统计数据
    loadPoolData();
    
    // 确保上传侧边栏的默认类型按钮是激活状态
    const uploadTypeButtons = document.querySelectorAll('.upload-type-btn');
    uploadTypeButtons.forEach(btn => {
        if (btn.textContent.includes('Tabular Data')) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // 确保目标数据集按钮的激活状态
    const targetDatasetButtons = document.querySelectorAll('.target-dataset-btn');
    targetDatasetButtons.forEach(btn => {
        if (btn.textContent.includes('Test Set')) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    window.addEventListener('resize', function() {
        if (dataChart) dataChart.resize();
    });
    
    bindFileSelectionEvents();

    // 页面初始化时先拉取一次清洗状态：即使未点击“数据清洗”，也可展示已有 joint_latest.json 的旧结果
    fetchAndUpdateCleaningStatus();
    
    // ****************** 新增：页面卸载时清理轮询定时器 ******************
    window.addEventListener('beforeunload', function() {
        stopCleaningStatusPolling();
    });
    // ****************************************************************
};

// ****************** 新增：数据清洗任务核心函数 ******************

/**
 * 启动数据清洗任务
 */
async function runDataCleaning() {
    if (isCleaningRunning) {
        alert('数据清洗任务正在进行中，请勿重复启动。');
        return;
    }

    if (!confirm('确定要开始数据清洗吗？这将启动强化学习训练过程。')) {
        return;
    }

    try {
        const response = await fetch('/api/run-data-cleaning', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || '启动失败');
        }

        alert(result.message || '数据清洗已启动！');
        isCleaningRunning = true;

        // 启动状态轮询
        startCleaningStatusPolling();
        // 更新右侧时间线标题，提示任务开始
        updateTimelineWithStart();

    } catch (error) {
        console.error('启动数据清洗失败:', error);
        alert(`启动失败: ${error.message}`);
    }
}

/**
 * 开始轮询获取数据清洗状态
 */
function startCleaningStatusPolling() {
    if (cleaningStatusTimer) {
        clearInterval(cleaningStatusTimer);
    }
    // 每2秒获取一次状态
    cleaningStatusTimer = setInterval(fetchAndUpdateCleaningStatus, 2000);
    // 立即获取一次
    fetchAndUpdateCleaningStatus();
}

/**
 * 获取并更新数据清洗状态
 */
async function fetchAndUpdateCleaningStatus() {
    try {
        const [statusResp, pendingResp] = await Promise.all([
            fetch('/api/get-cleaning-status?dataset=adult'),
            fetch('/api/pending-decision?dataset=adult')
        ]);

        const statusResult = await statusResp.json();
        const pendingResult = await pendingResp.json();

        if (!statusResp.ok) {
            throw new Error(statusResult.error || '获取状态失败');
        }

        if (statusResult.success) {
            const status = statusResult.status;
            const decisionTable = statusResult.decision_table || [];
            const iterationHistory = statusResult.iteration_history || [];

            // 如果 pending=false 或 decision!=null，则认为未进入交互等待，不覆盖决策表
            pendingDecisionData = (pendingResult && pendingResult.success) ? pendingResult.data : null;
            const pendingValid = pendingDecisionData && pendingDecisionData.pending === true && Array.isArray(pendingDecisionData.samples) && pendingDecisionData.samples.length > 0 && pendingDecisionData.decision == null;

            if (pendingValid) {
                updateDecisionTableWithPending(pendingDecisionData);
            } else {
                updateDecisionTableWithRLData(decisionTable);
            }

            // 时间线优先使用 joint_latest.json 的 steps；若为空则回退到 pending_decision.json
            if (status) {
                updateTimelineWithIteration(status, decisionTable, pendingDecisionData, iterationHistory);
            } else {
                updateTimelineWithEmptyStatus();
            }

            if (status && status.iteration >= status.total_iterations && status.total_iterations > 0) {
                stopCleaningStatusPolling();
                alert(`数据清洗训练完成！最终准确率: ${(status.accuracy * 100).toFixed(2)}%`);
                isCleaningRunning = false;

                if (currentDisplayDataset === 'result') {
                    console.log('训练完成，自动刷新结果集数据。');
                    loadResultData();
                }
            }
        } else {
            console.warn('获取状态成功但数据异常:', statusResult.message);
        }
    } catch (error) {
        console.error('获取数据清洗状态失败:', error);
    }
}


/**
 * 更新动作分布图表（示例）
 */
function updateActionDistributionChart(actionDist) {
    if (!actionDist || !dataChart) return;
    const chart = dataChart;
    const labels = [];
    const values = [];
    for (const [action, prob] of Object.entries(actionDist)) {
        // 将动作名转换为更易读的形式
        const actionMap = {
            'modify_features': '修改特征',
            'modify_labels': '修改标签',
            'delete_samples': '删除样本',
            'add_samples': '添加样本',
            'no_op': '无操作'
        };
        labels.push(actionMap[action] || action);
        values.push((prob * 100).toFixed(1));
    }

    const option = {
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c}%'
        },
        legend: {
            orient: 'vertical',
            left: 'left',
            top: 'center',
            textStyle: { fontSize: 10 }
        },
        series: [
            {
                name: '动作分布',
                type: 'pie',
                radius: '60%',
                center: ['50%', '50%'],
                data: labels.map((label, idx) => ({ name: label, value: values[idx] })),
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                label: {
                    fontSize: 10
                }
            }
        ]
    };
    chart.setOption(option, true);
}

/**
 * 用强化学习决策数据更新决策表格
 */
function updateDecisionTableWithRLData(decisionTable) {
    const tbody = document.getElementById('decisionTableBody');
    if (!tbody) return;

    if (!decisionTable || decisionTable.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-center py-4 text-gray-dark">暂无决策数据，或训练尚未产生步骤。</td></tr>`;
        return;
    }

    let tableHTML = '';

    decisionTable.forEach((stepData) => {
        const stepNum = Number(stepData.step ?? 0);

        // 兼容两种结构：
        // 1) 老结构: stepData.samples 是样本级详情
        // 2) 新结构: 只有 step 级统计字段（step/action/reward/accuracy/dirty_ratio）
        if (Array.isArray(stepData.samples) && stepData.samples.length > 0) {
            stepData.samples.forEach((sample) => {
                const sampleId = sample.sample_id;
                const dataType = sample.data_type === 'tabular' ? '表格' : sample.data_type;
                const opType = mapActionToOperation(sample.action);
                const issueDetail = getIssueDetailByAction(sample);
                const originalVal = getOriginalValueByAction(sample);
                const suggestedVal = getSuggestedValueByAction(sample);
                const confidenceValue = Number(sample.confidence ?? 0);
                const confidence = `${(confidenceValue * 100).toFixed(1)}%`;

                tableHTML += `
                    <tr>
                        <td class="text-center"><input type="checkbox" class="row-checkbox" data-id="${sampleId}-${stepNum}"></td>
                        <td>${sampleId}</td>
                        <td>${dataType || '--'}</td>
                        <td><span class="diff-badge" style="background: ${getColorForOperation(opType)};">${opType}</span></td>
                        <td>${issueDetail}</td>
                        <td><code>${originalVal}</code></td>
                        <td><code>${suggestedVal}</code></td>
                        <td><span class="confidence-badge" style="background: ${getColorForConfidence(confidenceValue)};">${confidence}</span></td>
                        <td>
                            <button class="action-btn btn-success btn-sm" onclick="acceptDecision('${sampleId}', ${stepNum})">接受</button>
                            <button class="action-btn btn-danger btn-sm" onclick="rejectDecision('${sampleId}', ${stepNum})">拒绝</button>
                        </td>
                    </tr>
                `;
            });
        } else {
            // 只有 step 级信息时，生成占位行，避免 JS 报错并保证时间线可继续更新
            const reward = Number(stepData.reward ?? 0);
            const acc = Number(stepData.accuracy ?? 0);
            const dirty = Number(stepData.dirty_ratio ?? 0);

            tableHTML += `
                <tr>
                    <td class="text-center"><input type="checkbox" class="row-checkbox" data-id="step-${stepNum}"></td>
                    <td>step-${stepNum}</td>
                    <td>统计</td>
                    <td><span class="diff-badge" style="background: ${getColorForOperation(mapActionToOperation(stepData.action || 'no_op'))};">${mapActionToOperation(stepData.action || 'no_op')}</span></td>
                    <td>reward=${reward.toFixed(4)} | dirty_ratio=${(dirty * 100).toFixed(2)}%</td>
                    <td><code>--</code></td>
                    <td><code>--</code></td>
                    <td><span class="confidence-badge" style="background:#86909C;">acc ${(acc * 100).toFixed(2)}%</span></td>
                    <td>
                        <button class="action-btn btn-success btn-sm" disabled>接受</button>
                        <button class="action-btn btn-danger btn-sm" disabled>拒绝</button>
                    </td>
                </tr>
            `;
        }
    });

    tbody.innerHTML = tableHTML || `<tr><td colspan="9" class="text-center py-4 text-gray-dark">暂无可展示数据。</td></tr>`;
    updateSelectedCount();
}

// 辅助函数：将动作映射为操作类型
function mapActionToOperation(action) {
    const map = {
        'modify_features': 'Repair',
        'modify_labels': 'Relabel',
        'add_samples': 'Add',
        'delete_samples': 'Delete',
        'no_op': 'No-Op'
    };
    return map[action] || action;
}

// 辅助函数：根据动作生成问题详情
function getIssueDetailByAction(sample) {
    switch (sample.action) {
        case 'modify_features':
            return '特征异常值';
        case 'modify_labels':
            return sample.label_changed === 1 ? '标签可能错误' : '标签检查通过';
        case 'add_samples':
            return '从候选池添加';
        case 'delete_samples':
            return '疑似噪声样本';
        default:
            return '无操作';
    }
}

// 辅助函数：获取原始值
function getOriginalValueByAction(sample) {
    switch (sample.action) {
        case 'modify_features':
            return sample.original_features_preview || 'N/A';
        case 'modify_labels':
            return `Label: ${sample.original_label}`;
        case 'add_samples':
            return `Label: ${sample.original_label}`;
        default:
            return 'N/A';
    }
}

// 辅助函数：获取建议值
function getSuggestedValueByAction(sample) {
    switch (sample.action) {
        case 'modify_features':
            return '特征已修正';
        case 'modify_labels':
            return `预测: ${sample.predicted_label} (${sample.label_changed ? '已更改' : '未更改'})`;
        case 'add_samples':
            return '已加入训练集';
        case 'delete_samples':
            return '将删除';
        default:
            return 'N/A';
    }
}

// 辅助函数：根据操作类型获取颜色
function getColorForOperation(op) {
    const colorMap = {
        'Repair': '#FF7D00',
        'Relabel': '#00B42A',
        'Add': '#165DFF',
        'Delete': '#F53F3F',
        'No-Op': '#86909C'
    };
    return colorMap[op] || '#86909C';
}

// 辅助函数：根据置信度获取颜色
function getColorForConfidence(conf) {
    if (conf >= 0.9) return '#00B42A';
    if (conf >= 0.7) return '#FF7D00';
    return '#F53F3F';
}

/**
 * 更新右侧时间线
 */

function updateTimelineWithIteration(status, decisionTable, pendingDecisionData = null, iterationHistory = []) {
    if (!status) return;
    const timelineContainer = document.querySelector('.timeline-container');
    if (!timelineContainer) return;

    const safeNumber = (value, defaultValue = 0) => {
        const n = Number(value);
        return Number.isFinite(n) ? n : defaultValue;
    };

    const currentIteration = safeNumber(status.iteration);
    const isCollapsed = !!timelineCollapsedIterations[currentIteration];

    let timelineHTML = `
        <div class="timeline-title">
            <i class="fa fa-history"></i>强化学习清洗进度
            <small class="text-gray-dark">(第 ${currentIteration}/${safeNumber(status.total_iterations)} 轮 | ${status.timestamp || '--'})</small>
        </div>
    `;

    // 历史轮次（来自 training_log.json）
    if (Array.isArray(iterationHistory) && iterationHistory.length > 0) {
        const sortedHistory = [...iterationHistory]
            .filter(it => safeNumber(it.iteration) > 0)
            .sort((a, b) => safeNumber(a.iteration) - safeNumber(b.iteration));

        sortedHistory.forEach((it) => {
            const iter = safeNumber(it.iteration);
            const historyCollapsed = !!timelineCollapsedIterations[`hist-${iter}`];
            const steps = Array.isArray(it.steps) ? it.steps : [];

            timelineHTML += `
                <div class="timeline-item">
                    <div class="timeline-dot" style="background:#86909C;"></div>
                    <div class="timeline-line"></div>
                    <div class="timeline-iteration-title" style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;" onclick="toggleTimelineIteration('hist-${iter}')">
                        <span>Iteration ${iter}（历史）</span>
                        <span class="badge">${historyCollapsed ? '展开 Steps' : '收起 Steps'}</span>
                    </div>
                    <div class="timeline-content-row"><strong>timestamp:</strong> ${it.timestamp || '--'}</div>
                    <div class="timeline-content-row"><strong>accuracy:</strong> ${(safeNumber(it.accuracy) * 100).toFixed(2)}%</div>
                    <div class="timeline-content-row"><strong>best_accuracy:</strong> ${(safeNumber(it.best_accuracy) * 100).toFixed(2)}%</div>
                    <div class="timeline-content-row"><strong>dirty_ratio:</strong> ${(safeNumber(it.dirty_ratio) * 100).toFixed(2)}%</div>
                </div>
            `;

            if (!historyCollapsed && steps.length > 0) {
                const sortedSteps = [...steps].sort((a, b) => safeNumber(a.step) - safeNumber(b.step));
                sortedSteps.forEach((s) => {
                    timelineHTML += `
                        <div class="timeline-item">
                            <div class="timeline-dot" style="background:#C9CDD4;"></div>
                            <div class="timeline-iteration-title">Step ${safeNumber(s.step)}</div>
                            <div class="timeline-content-row"><strong>action:</strong> ${s.action || '--'}</div>
                            <div class="timeline-content-row"><strong>reward:</strong> ${safeNumber(s.reward).toFixed(6)}</div>
                        </div>
                    `;
                });
            }
        });
    }

    timelineHTML += `
        <div class="timeline-item">
            <div class="timeline-dot" style="background:#165DFF;"></div>
            <div class="timeline-line"></div>
            <div class="timeline-iteration-title" style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;" onclick="toggleTimelineIteration(${currentIteration})">
                <span>Iteration ${currentIteration} 概览</span>
                <span class="badge">${isCollapsed ? '展开 Steps' : '收起 Steps'}</span>
            </div>
            <div class="timeline-content-row"><strong>timestamp:</strong> ${status.timestamp || '--'}</div>
            <div class="timeline-content-row"><strong>action:</strong> ${(decisionTable && decisionTable.length > 0) ? (decisionTable[decisionTable.length - 1].action || '--') : '--'}</div>
            <div class="timeline-content-row"><strong>best_accuracy:</strong> ${(safeNumber(status.best_accuracy) * 100).toFixed(2)}%</div>
            <div class="timeline-content-row"><strong>accuracy:</strong> ${(safeNumber(status.accuracy) * 100).toFixed(2)}%</div>
            <div class="timeline-content-row"><strong>dirty_ratio:</strong> ${(safeNumber(status.dirty_ratio) * 100).toFixed(2)}%</div>
        </div>
    `;

    if (!isCollapsed) {
        if (decisionTable && decisionTable.length > 0) {
            const sortedSteps = [...decisionTable].sort((a, b) => safeNumber(a.step) - safeNumber(b.step));
            sortedSteps.forEach((stepData, index) => {
                const isLatestStep = index === sortedSteps.length - 1;
                const rewardValue = safeNumber(stepData.reward);
                const accuracyValue = safeNumber(stepData.accuracy);
                const dirtyRatioValue = safeNumber(stepData.dirty_ratio);

                timelineHTML += `
                    <div class="timeline-item">
                        <div class="timeline-dot" style="background: ${isLatestStep ? '#00B42A' : '#86909C'};"></div>
                        ${index < sortedSteps.length - 1 ? '<div class="timeline-line"></div>' : ''}
                        <div class="timeline-iteration-title">Step ${safeNumber(stepData.step)} ${isLatestStep ? '<span class="badge">最新</span>' : ''}</div>
                        <div class="timeline-content-row"><strong>step:</strong> ${safeNumber(stepData.step)}</div>
                        <div class="timeline-content-row"><strong>action:</strong> ${stepData.action || '--'}</div>
                        <div class="timeline-content-row"><strong>reward:</strong> <span class="${rewardValue >= 0 ? 'text-success' : 'text-danger'}">${rewardValue.toFixed(6)}</span></div>
                        <div class="timeline-content-row"><strong>accuracy:</strong> ${(accuracyValue * 100).toFixed(2)}%</div>
                        <div class="timeline-content-row"><strong>dirty_ratio:</strong> ${(dirtyRatioValue * 100).toFixed(2)}%</div>
                    </div>
                `;
            });
        } else if (pendingDecisionData && typeof pendingDecisionData === 'object') {
            const pendingStep = safeNumber(pendingDecisionData.step ?? 0);
            const rewardValue = safeNumber(pendingDecisionData.reward);
            const accuracyValue = safeNumber(pendingDecisionData.post_accuracy ?? pendingDecisionData.accuracy ?? 0);
            const dirtyRatioValue = safeNumber(pendingDecisionData.post_dirty_ratio ?? pendingDecisionData.dirty_ratio ?? 0);
            timelineHTML += `
                <div class="timeline-item">
                    <div class="timeline-dot" style="background: #FF7D00;"></div>
                    <div class="timeline-iteration-title">Step ${pendingStep}（等待确认）</div>
                    <div class="timeline-content-row"><strong>step:</strong> ${pendingStep}</div>
                    <div class="timeline-content-row"><strong>action:</strong> ${pendingDecisionData.action || '--'}</div>
                    <div class="timeline-content-row"><strong>reward:</strong> <span class="${rewardValue >= 0 ? 'text-success' : 'text-danger'}">${rewardValue.toFixed(6)}</span></div>
                    <div class="timeline-content-row"><strong>accuracy:</strong> ${(accuracyValue * 100).toFixed(2)}%</div>
                    <div class="timeline-content-row"><strong>dirty_ratio:</strong> ${(dirtyRatioValue * 100).toFixed(2)}%</div>
                </div>
            `;
        } else {
            timelineHTML += `
                <div class="timeline-item">
                    <div class="timeline-dot"></div>
                    <div class="timeline-iteration-title">等待 step 数据</div>
                    <div class="timeline-content-row text-gray-dark"><i>训练启动后将从 joint_latest.json 的 steps 字段实时刷新。</i></div>
                </div>
            `;
        }
    }

    if (safeNumber(status.iteration) >= safeNumber(status.total_iterations) && safeNumber(status.total_iterations) > 0) {
        timelineHTML += `
            <div class="timeline-item">
                <div class="timeline-dot" style="background: #00B42A;"></div>
                <div class="timeline-iteration-title">训练完成</div>
                <div class="timeline-content-row text-success"><i class="fa fa-check-circle"></i> 数据清洗训练已全部完成。</div>
            </div>
        `;
    }

    const shouldStickBottom = Math.abs((timelineContainer.scrollTop + timelineContainer.clientHeight) - timelineContainer.scrollHeight) < 24;
    const isNewIteration = lastTimelineIteration === null || currentIteration !== lastTimelineIteration;

    timelineContainer.innerHTML = timelineHTML;

    // 仅当用户本来就在底部，或出现新 iteration 时，才自动滚动到底
    if (shouldStickBottom || isNewIteration) {
        timelineContainer.scrollTop = timelineContainer.scrollHeight;
    }

    lastTimelineIteration = currentIteration;
}

function updateTimelineWithStart() {
    const timelineContainer = document.querySelector('.timeline-container');
    if (timelineContainer) {
        const startHTML = `
            <div class="timeline-title">
                <i class="fa fa-history"></i>数据清洗进度
            </div>
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-line"></div>
                <div class="timeline-iteration-title">任务启动 | ${new Date().toLocaleString()}</div>
                <div class="timeline-content-row">
                    <strong>状态:</strong> 正在启动强化学习训练脚本...
                </div>
                <div class="timeline-content-row">
                    <strong>信息:</strong> 开始监听训练状态更新。
                </div>
            </div>
        `;
        timelineContainer.innerHTML = startHTML;
    }
}

function updateTimelineWithEmptyStatus() {
    const timelineContainer = document.querySelector('.timeline-container');
    if (!timelineContainer) return;
    timelineContainer.innerHTML = `
        <div class="timeline-title">
            <i class="fa fa-history"></i>强化学习清洗进度
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-iteration-title">等待 joint_latest.json</div>
            <div class="timeline-content-row text-gray-dark">当前尚未读取到轮次状态，请确认训练已启动或检查 checkpoints 路径。</div>
        </div>
    `;
}

function toggleTimelineIteration(iteration) {
    timelineCollapsedIterations[iteration] = !timelineCollapsedIterations[iteration];
    fetchAndUpdateCleaningStatus();
}

function updateDecisionTableWithPending(pendingData) {
    const tbody = document.getElementById('decisionTableBody');
    if (!tbody) return;

    if (!pendingData || !Array.isArray(pendingData.samples) || pendingData.samples.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-center py-4 text-gray-dark">暂无待确认决策（pending_decision.json）。</td></tr>`;
        return;
    }

    const stepNum = Number(pendingData.step ?? 0);
    const actionName = pendingData.action || '--';

    const rows = pendingData.samples.map((sample) => {
        const sampleId = sample.sample_id ?? '--';
        const opType = mapActionToOperation(sample.action || actionName);
        const confidenceValue = Number(sample.noise_score ?? 0);
        const confidence = `${(confidenceValue * 100).toFixed(1)}%`;
        const originalVal = sample.original_features ? JSON.stringify(sample.original_features) : (sample.original_label ?? '--');
        const suggestedVal = sample.predicted_features ? JSON.stringify(sample.predicted_features) : (sample.predicted_label ?? '--');

        return `
            <tr>
                <td class="text-center"><input type="checkbox" class="row-checkbox" data-id="${sampleId}-${stepNum}"></td>
                <td>${sampleId}</td>
                <td>表格</td>
                <td><span class="diff-badge" style="background: ${getColorForOperation(opType)};">${opType}</span></td>
                <td>${sample.action || actionName}</td>
                <td><code>${originalVal}</code></td>
                <td><code>${suggestedVal}</code></td>
                <td><span class="confidence-badge" style="background: ${getColorForConfidence(confidenceValue)};">${confidence}</span></td>
                <td>
                    <button class="action-btn btn-success btn-sm" onclick="confirmPendingDecision()">确定</button>
                </td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = rows;
    updateSelectedCount();
}

async function sendDecisionSignal(signal) {
    const response = await fetch('/api/submit-decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signal })
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
        throw new Error(result.error || '发送决策信号失败');
    }
}

async function confirmPendingDecision() {
    try {
        await sendDecisionSignal('y');
        alert('已发送确认信号 y');
    } catch (error) {
        alert(`发送失败: ${error.message}`);
    }
}

/**
 * 停止轮询
 */
function stopCleaningStatusPolling() {
    if (cleaningStatusTimer) {
        clearInterval(cleaningStatusTimer);
        cleaningStatusTimer = null;
    }
    isCleaningRunning = false;
}

// 接受单个决策（示例函数）
function acceptDecision(sampleId, stepNum) {
    alert(`已接受对样本 ${sampleId} (步骤 ${stepNum}) 的决策。`);
    // 这里可以调用后端API确认此决策
}

// 拒绝单个决策（示例函数）
function rejectDecision(sampleId, stepNum) {
    alert(`已拒绝对样本 ${sampleId} (步骤 ${stepNum}) 的决策。`);
    // 这里可以调用后端API拒绝此决策
}

// ****************************************************************
// ****************** 以下是原有的main.js功能函数 ******************

// 更新数据质量统计信息 - 修改：从 ../results.json 读取真实数据
async function updateStatistics() {
    console.log("正在从 /api/statistics 获取统计数据...");
    
    try {
        // 调用专门的统计API
        const response = await fetch('/api/statistics');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const stats = await response.json();
        
        console.log("获取到统计信息:", stats);
        
        // 更新页面上的统计数字
        const metricValues = document.querySelectorAll('.metric-value');
        if (metricValues.length >= 6) {
            // 直接使用API返回的数据
            metricValues[0].textContent = stats.n_samples ? stats.n_samples.toLocaleString() : 'N/A';
            metricValues[1].textContent = stats.n_features || 'N/A';
            metricValues[2].textContent = stats.n_classes || 'N/A';
            metricValues[3].textContent = stats.missing_rate || 'N/A';
            metricValues[4].textContent = stats.feature_noise_ratio || 'N/A';
            metricValues[5].textContent = stats.label_noise_ratio || 'N/A';
        }
    } catch (error) {
        console.error('获取统计信息失败:', error);
        // 如果获取失败，保留页面上的静态数据
    }
    
    // 更新选择/未处理计数
    const testCount = testSetData.length;
    const valCount = valSetData.length;
    const totalCount = testCount + valCount;
    const selectedElement = document.getElementById('selectedCount');
    if (selectedElement) {
        selectedElement.textContent = `0/${totalCount}`;
    }
}

function triggerTableFileInput() {
    document.getElementById('tableFileInput').click();
}

function triggerArchiveFileInput() {
    document.getElementById('archiveFileInput').click();
}


function initChart(){
        dataChart = echarts.init(document.getElementById('dataVisualChart'));
    
    // 初始显示"暂无图片"提示
    const chartContainer = document.getElementById('dataVisualChart');
    if (chartContainer) {
        chartContainer.innerHTML = '';
        
        const noDataDiv = document.createElement('div');
        noDataDiv.className = 'w-full h-full flex flex-col items-center justify-center text-gray-500';
        noDataDiv.innerHTML = `
            <i class="fa fa-image text-4xl mb-3 opacity-30"></i>
            <p class="text-sm font-medium">等待数据可视化</p>
            <p class="text-xs mt-1">点击"质量检测"生成图表</p>
        `;
        chartContainer.appendChild(noDataDiv);
    }
}
// 切换上传类型
function switchUploadType(type) {
    currentUploadType = type;
    const buttons = document.querySelectorAll('.upload-type-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
        if ((type === 'table' && btn.textContent.includes('Tabular Data')) ||
            (type === 'archive' && btn.textContent.includes('Archive'))) {
            btn.classList.add('active');
        }
    });
    document.getElementById('tableUploadArea').classList.add('hidden');
    document.getElementById('archiveUploadArea').classList.add('hidden');
    document.getElementById(type + 'UploadArea').classList.remove('hidden');
}  

// 切换上传目标数据集
function switchTargetDataset(dataset) {
    currentTargetDataset = dataset;
    const buttons = document.querySelectorAll('.target-dataset-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
}       

// 绑定文件选择预览事件
function bindFileSelectionEvents() {
    document.getElementById('tableFileInput').addEventListener('change', function(e) {
        if (this.files.length > 0) {
            const file = this.files[0];
            handleFileSelect([file], 'table');
        }
    });
    document.getElementById('archiveFileInput').addEventListener('change', function(e) {
        if (this.files.length > 0) {
            const file = this.files[0];
            handleFileSelect([file], 'archive');
        }
    });
}

function populateColumnSelect(columns) {
    const select = document.getElementById('targetColumnSelect');
    const section = document.getElementById('targetColumnSection');
    
    if (!select || !section) return;
    
    // 清空现有选项
    select.innerHTML = '<option value="">-- Select target column --</option>';
    
    // 添加列选项
    columns.forEach(col => {
        if (col && col.trim()) {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            select.appendChild(option);
        }
    });
    
    // 显示目标列选择器
    section.classList.remove('hidden');
    
    // 尝试自动选择常见的标签列
    const commonLabels = ['label', 'target', 'y', 'class', 'income', 'category', 'type'];
    for (const label of commonLabels) {
        if (columns.includes(label)) {
            select.value = label;
            break;
        }
    }
}

// 显示列名输入框
function showColumnInput() {
    const section = document.getElementById('targetColumnSection');
    if (section) {
        section.classList.remove('hidden');
    }
}

// 修改 uploadTableFile 函数，加入目标列选择
async function uploadTableFile() {
    const fileInput = document.getElementById('tableFileInput');
    if (!fileInput.files.length) {
        alert('Please select a table file first!');
        return;
    }
    
    const file = fileInput.files[0];
    const targetColumnSelect = document.getElementById('targetColumnSelect');
    const targetColumn = targetColumnSelect ? targetColumnSelect.value : '';
    
    if (!targetColumn) {
        alert('Please select a target column for classification!');
        return;
    }
    
    const targetName = currentTargetDataset === 'test' ? 'Test Set' : 'Validation Set';
    updateFileInfoUI(file, 'table', 'uploading');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', 'table');
        formData.append('dataset', currentTargetDataset);
        formData.append('target_column', targetColumn);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || 'Upload failed');
        }
        
        const newData = result.data;
        const targetArray = currentTargetDataset === 'test' ? testSetData : valSetData;
        targetArray.push(...newData);
        
        updateFileInfoUI(file, 'table', 'success');
        
        // 存储选中的目标列
        currentTargetColumn = targetColumn;
        
        if (currentDisplayDataset === currentTargetDataset) {
            setTimeout(() => {
                try {
                    renderDisplayData();
                } catch (renderError) {
                    console.error('Rendering data error:', renderError);
                }
            }, 0);
        }
        
        updateStatistics();
        alert(`Successfully added ${newData.length} records to ${targetName}`);
        
        // 重置文件输入
        fileInput.value = '';
        if (targetColumnSelect) {
            targetColumnSelect.value = '';
        }
        
    } catch (error) {
        updateFileInfoUI(file, 'table', 'error');
        console.error('Upload failed:', error);
        alert(`Upload failed: ${error.message}`);
    }
}
        
async function uploadArchiveFile() {
    const fileInput = document.getElementById('archiveFileInput');
    if (!fileInput.files.length) {
        alert('Please select an archive file!');
        return;
    }
    const file = fileInput.files[0];
    const targetName = currentTargetDataset === 'test' ? 'Test Set' : 'Validation Set';
    updateFileInfoUI(file, 'archive', 'uploading');
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', 'archive');
        formData.append('dataset', currentTargetDataset);
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || 'Upload failed');
        }
        const newData = result.data;
        const targetArray = currentTargetDataset === 'test' ? testSetData : valSetData;
        targetArray.push(...newData);
        updateFileInfoUI(file, 'archive', 'success');
        if (currentDisplayDataset === currentTargetDataset) {
            setTimeout(() => {
                try {
                    renderDisplayData();
                } catch (renderError) {
                    console.error('渲染数据时出错:', renderError);
                    alert('数据上传成功，但渲染时出错: ' + renderError.message);
                }
            }, 0);
        }
        updateStatistics(); // 数据更新后，重新获取统计信息
        alert(`Successfully uploaded archive to ${targetName}`);
        fileInput.value = '';
    } catch (error) {
        updateFileInfoUI(file, 'archive', 'error');
        console.error('Upload failed:', error);
        alert(`Upload failed: ${error.message}`);
    }
}

// 切换图表类型

function switchChart(type) {
    if (currentChartType === type) return;
    const tabs = document.querySelectorAll('.chart-tab');
    tabs.forEach(tab => {
        tab.classList.toggle('active', tab.textContent.includes(type === 'class' ? 'Class Distribution' : 'TSNE Dimensionality Reduction'));
    });
    currentChartType = type;
    
    // 优先显示生成的图表图片
    if (window.generatedChartImages && window.generatedChartImages.length > 0) {
        displayGeneratedChart(type === 'class' ? 'distribution' : 'tsne');
    } else {
        // 如果没有生成图片，显示暂无图片提示
        const chartContainer = document.getElementById('dataVisualChart');
        if (chartContainer) {
            chartContainer.innerHTML = '';
            
            const noDataDiv = document.createElement('div');
            noDataDiv.className = 'w-full h-full flex flex-col items-center justify-center text-gray-500';
            noDataDiv.innerHTML = `
                <i class="fa fa-image text-4xl mb-3 opacity-30"></i>
                <p class="text-sm font-medium">暂无可视化图片</p>
                <p class="text-xs mt-1">点击"质量检测"按钮生成图表</p>
            `;
            chartContainer.appendChild(noDataDiv);
        }
    }
}

// 显示生成的图表图片
function displayGeneratedChart(type) {
    if (!window.generatedChartImages || window.generatedChartImages.length === 0) {
        console.warn("没有可显示的图表图片");
        return;
    }
    
    // 根据类型筛选图片
    const chartImages = window.generatedChartImages.filter(img => {
        if (type === 'distribution') {
            return img.type === 'distribution';
        } else if (type === 'tsne') {
            return img.type === 'tsne';
        }
        return false;
    });
    
    if (chartImages.length === 0) {
        console.warn(`没有找到类型为 ${type} 的图表图片`);
        return;
    }
    
    // 显示第一张图片
    const imgUrl = chartImages[0].url;
    
    // 检查图片是否可以访问
    console.log(`尝试加载图表: ${imgUrl}`);
    
    // 使用图片元素显示
    const chartContainer = document.getElementById('dataVisualChart');
    if (chartContainer) {
        // 清空之前的图表
        chartContainer.innerHTML = '';
        
        // 创建图片元素
        const img = document.createElement('img');
        img.src = imgUrl;
        img.alt = `${type} 图表`;
        img.className = 'w-full h-full object-contain';
        img.onload = function() {
            console.log(`图表图片加载成功: ${imgUrl}`);
        };
        
        chartContainer.appendChild(img);
    }
}


// 初始化上传表格分页控件
function initUploadTablePagination() {
    updatePaginationButtons();
}

// 初始化决策表格分页控件
function initDecisionTablePagination() {
    updateDecisionPaginationButtons();
}

// 切换决策表格页码
function changeDecisionPage(target) {
    let newPage = decisionPageConfig.currentPage;
    if (target === 'prev') {
        newPage--;
    } else if (target === 'next') {
        newPage++;
    } else {
        newPage = parseInt(target);
    }
    if (newPage < 1) newPage = 1;
    if (newPage > decisionPageConfig.totalPages) newPage = decisionPageConfig.totalPages;
    decisionPageConfig.currentPage = newPage;
    renderDecisionTable();
    document.getElementById('decisionPrevBtn').disabled = newPage === 1;
    document.getElementById('decisionNextBtn').disabled = newPage === decisionPageConfig.totalPages;
    document.getElementById('decisionPageInput').value = newPage;
    const pageBtns = document.querySelectorAll('#decisionPageNumbers .pagination-btn');
    pageBtns.forEach((btn, idx) => {
        btn.classList.toggle('active', idx + 1 === newPage);
    });
}

// 跳转决策表格页码
function jumpDecisionPage() {
    const input = document.getElementById('decisionPageInput');
    const page = parseInt(input.value);
    if (!isNaN(page) && page >= 1 && page <= decisionPageConfig.totalPages) {
        changeDecisionPage(page);
    } else {
        input.value = decisionPageConfig.currentPage;
    }
}

// 渲染决策表格数据
function renderDecisionTable() {
    // 此处省略决策表格渲染的具体实现，与主要修改无关
    console.log("渲染决策表格...");
}

// ==================== 核心修改：实现真实分页的数据渲染 ====================
async function renderDisplayData() {
    const container = document.getElementById('dataDisplayContainer');
    if (!container) return;
    
    // 清空容器
    container.innerHTML = '';
    
    // 根据当前显示的数据集选择数据
    let allData = [];
    switch (currentDisplayDataset) {
        case 'test':
            allData = testSetData;
            break;
        case 'val':
            allData = valSetData;
            break;
        case 'pool':
            allData = poolData;
            break;
        case 'result':
            allData = resultSetData;
            break;
        default:
            allData = testSetData;
    }
    
    // 如果没有数据，显示提示
    if (!allData || allData.length === 0) {
        let message = '';
        let icon = 'fa-database';
        
        switch (currentDisplayDataset) {
            case 'result':
                message = '暂无清洗结果数据。请先运行数据清洗任务。';
                icon = 'fa-filter';
                break;
            case 'pool':
                message = '候选池为空。请先运行数据增强任务。';
                icon = 'fa-cube';
                break;
            default:
                message = '暂无数据。请点击"上传数据"按钮添加数据。';
        }
        
        container.innerHTML = `
            <div class="no-data-hint">
                <i class="fa ${icon}"></i>
                <p>${message}</p>
            </div>
        `;
        
        // 更新分页信息为空状态
        uploadPageConfig.totalItems = 0;
        uploadPageConfig.totalPages = 1;
        updatePaginationInfo();
        return;
    }
    
    // ✅ 关键修复1：计算分页数据
    const pageSize = uploadPageConfig.pageSize;
    const currentPage = uploadPageConfig.currentPage;
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, allData.length);
    
    // 只渲染当前页的数据
    const dataToShow = allData.slice(startIndex, endIndex);
    
    // ✅ 关键修复2：更新分页配置
    uploadPageConfig.totalItems = allData.length;
    uploadPageConfig.totalPages = Math.max(1, Math.ceil(allData.length / pageSize));
    
    // 确保当前页不越界
    if (currentPage > uploadPageConfig.totalPages) {
        uploadPageConfig.currentPage = 1;
        // 递归调用，重新渲染第一页
        renderDisplayData();
        return;
    }
    
    // 创建表格
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // 创建表头
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    // 获取所有可能的列（从第一条数据中）
    if (dataToShow.length > 0) {
        const sample = dataToShow[0];
        const columns = Object.keys(sample);
        
        // 添加表头
        columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            headerRow.appendChild(th);
        });
    }
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // 创建表体
    const tbody = document.createElement('tbody');
    
    // ✅ 关键修复3：只遍历当前页数据
    dataToShow.forEach((item, index) => {
        const row = document.createElement('tr');
        
        // 为每一列创建单元格
        Object.values(item).forEach(value => {
            const cell = document.createElement('td');
            
            // 特殊处理图片URL
            if (typeof value === 'string' && 
                (value.includes('http') || value.includes('.jpg') || value.includes('.png') || value.includes('.jpeg'))) {
                cell.innerHTML = `
                    <img src="${value}" 
                         alt="预览" 
                         class="img-preview" 
                         onclick="openImgModal('${value}')"
                         onerror="this.style.display='none'">
                `;
            } else {
                cell.textContent = value || '';
            }
            
            row.appendChild(cell);
        });
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    container.appendChild(table);
    
    // ✅ 关键修复4：更新分页UI
    updatePaginationInfo();
}

// 新增：更新分页信息显示
function updatePaginationInfo() {
    const pageInfo = document.getElementById('uploadPageInfo');
    if (pageInfo) {
        pageInfo.textContent = `Total ${uploadPageConfig.totalPages} Pages, ${uploadPageConfig.totalItems} Items`;
    }
    
    // 更新分页按钮状态
    document.getElementById('uploadPrevBtn').disabled = uploadPageConfig.currentPage === 1;
    document.getElementById('uploadNextBtn').disabled = uploadPageConfig.currentPage === uploadPageConfig.totalPages;
    document.getElementById('uploadPageInput').value = uploadPageConfig.currentPage;
    
    // 更新分页按钮
    updatePaginationButtons();
}



// 加载结果集数据（从后端API获取）
async function loadResultData() {
    try {
        const response = await fetch('/api/result-data');
        const result = await response.json();
        
        if (result.success) {
            resultSetData = result.data || [];
            console.log(`结果集数据加载成功，共 ${resultSetData.length} 条记录。`);
            
            // 只重新渲染，不重新计算分页，除非需要
            if (currentDisplayDataset === 'result') {
                // 重置到第一页
                uploadPageConfig.currentPage = 1;
                renderDisplayData();
            }
        } else {
            console.warn('获取结果集数据失败:', result.error);
            resultSetData = [];
        }
    } catch (error) {
        console.error('加载结果集数据失败:', error);
        resultSetData = [];
    }
}

// 当切换到结果集时，自动加载数据

// 切换上传数据标签
function switchDisplayDataset(dataset) {
    if (currentDisplayDataset === dataset) return;
    
    // 显示加载状态
    const container = document.getElementById('dataDisplayContainer');
    if (container) {
        container.innerHTML = `
            <div class="loading-hint" style="text-align: center; padding: 40px; color: #666;">
                <i class="fa fa-spinner fa-spin fa-2x"></i>
                <p style="margin-top: 10px;">正在加载数据...</p>
            </div>
        `;
    }
    
    // 使用requestAnimationFrame让UI有机会更新
    requestAnimationFrame(() => {
        currentDisplayDataset = dataset;
        
        // 更新按钮激活状态
        const buttons = document.querySelectorAll('.upload-card .tab-btn');
        buttons.forEach(btn => {
            if ((btn.textContent === 'Test Set' && dataset === 'test') ||
                (btn.textContent === 'Validation Set' && dataset === 'val') ||
                (btn.textContent === 'Candidate Pool' && dataset === 'pool') ||
                (btn.textContent === 'Result Set' && dataset === 'result')) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        // 重置到第一页
        uploadPageConfig.currentPage = 1;
        
        // 异步加载数据
        setTimeout(() => {
            if (dataset === 'result' && resultSetData.length === 0) {
                console.log('切换到结果集，开始加载数据...');
                loadResultData();
            } else {
                renderDisplayData();
            }
        }, 50); // 微小延迟，确保UI响应
    });
}

// ==================== 智能分页函数 ====================
function generatePaginationArray(currentPage, totalPages, maxVisible = 5) {
    const pages = [];
    if (totalPages <= maxVisible) {
        for (let i = 1; i <= totalPages; i++) {
            pages.push(i);
        }
    } else {
        let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
        let endPage = Math.min(totalPages, startPage + maxVisible - 1);
        if (endPage - startPage + 1 < maxVisible) {
            startPage = Math.max(1, endPage - maxVisible + 1);
        }
        if (startPage > 1) {
            pages.push(1);
            if (startPage > 2) {
                pages.push('...');
            }
        }
        for (let i = startPage; i <= endPage; i++) {
            pages.push(i);
        }
        if (endPage < totalPages) {
            if (endPage < totalPages - 1) {
                pages.push('...');
            }
            pages.push(totalPages);
        }
    }
    return pages;
}

function updatePaginationButtons() {
    const pageNumbers = document.getElementById('uploadPageNumbers');
    if (!pageNumbers) return;
    const totalPages = uploadPageConfig.totalPages;
    const currentPage = uploadPageConfig.currentPage;
    pageNumbers.innerHTML = '';
    const pages = generatePaginationArray(currentPage, totalPages, 3);
    pages.forEach(page => {
        const button = document.createElement('button');
        if (page === '...') {
            button.className = 'pagination-ellipsis';
            button.textContent = '...';
            button.disabled = true;
        } else {
            button.className = `pagination-btn ${page === currentPage ? 'active' : ''}`;
            button.textContent = page;
            button.onclick = () => changeUploadPage(page);
        }
        pageNumbers.appendChild(button);
    });
}

function changeUploadPage(target) {
    let newPage = uploadPageConfig.currentPage;
    if (target === 'prev') {
        newPage--;
    } else if (target === 'next') {
        newPage++;
    } else {
        newPage = parseInt(target);
    }
    if (newPage < 1) newPage = 1;
    if (newPage > uploadPageConfig.totalPages) newPage = uploadPageConfig.totalPages;
    uploadPageConfig.currentPage = newPage;
    document.getElementById('uploadPrevBtn').disabled = newPage === 1;
    document.getElementById('uploadNextBtn').disabled = newPage === uploadPageConfig.totalPages;
    document.getElementById('uploadPageInput').value = newPage;
    updatePaginationButtons();
    renderDisplayData();
}

function jumpUploadPage() {
    const input = document.getElementById('uploadPageInput');
    const page = parseInt(input.value);
    if (!isNaN(page) && page >= 1 && page <= uploadPageConfig.totalPages) {
        changeUploadPage(page);
    } else {
        input.value = uploadPageConfig.currentPage;
    }
}

// 文件处理和UI更新函数
function handleFileSelect(files, type) {
    if (!files || files.length === 0) return;
    const file = files[0];  // ✅ 修正语法错误
    updateFileInfoUI(file, type, 'selected');
    
    if (type === 'table') {
        // 显示目标列选择器
        showColumnInput();
        
        // 如果是CSV文件，尝试解析列名
        if (file.name.endsWith('.csv')) {
            parseCSVColumns(file);
        } else if (file.name.endsWith('.json')) {
            parseJSONColumns(file);
        } else {
            // 对于其他格式（Excel等），清空现有选项
            const select = document.getElementById('targetColumnSelect');
            if (select) {
                select.innerHTML = '<option value="">-- Select target column --</option>';
            }
        }
    }
}

function parseCSVColumns(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const content = e.target.result;
            const lines = content.split('\n');
            if (lines.length > 0) {
                const headers = lines[0].split(',').map(h => h.trim());
                populateColumnSelect(headers);
            }
        } catch (error) {
            console.error('解析CSV列名失败:', error);
            showColumnInput();
        }
    };
    reader.readAsText(file);
}

function parseJSONColumns(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const content = e.target.result;
            const data = JSON.parse(content);
            let columns = [];
            
            if (Array.isArray(data) && data.length > 0) {
                // 如果是数组形式的JSON
                columns = Object.keys(data[0]);
            } else if (typeof data === 'object') {
                // 如果是对象形式的JSON
                columns = Object.keys(data);
            }
            
            populateColumnSelect(columns);
        } catch (error) {
            console.error('解析JSON列名失败:', error);
            showColumnInput();
        }
    };
    reader.readAsText(file);
}
function handleFileDrop(event, type) {
    event.preventDefault();
    event.stopPropagation();
    const zone = event.currentTarget;
    zone.classList.remove('border-primary', 'bg-blue-100');
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        updateFileInfoUI(file, type, 'selected');
        const inputId = type === 'table' ? 'tableFileInput' : 'archiveFileInput';
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        document.getElementById(inputId).files = dataTransfer.files;
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.add('border-primary', 'bg-blue-100');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('border-primary', 'bg-blue-100');
}

function updateFileInfoUI(file, type, status) {
    const prefix = type === 'table' ? 'table' : 'archive';
    const zone = document.getElementById(`${prefix}UploadZone`);
    const initialDiv = document.getElementById(`${prefix}UploadInitial`);
    const infoDiv = document.getElementById(`${prefix}FileInfo`);
    const nameSpan = document.getElementById(`${prefix}FileName`);
    const sizeSpan = document.getElementById(`${prefix}FileSize`);
    const statusSpan = document.getElementById(`${prefix}FileStatus`);
    const progressContainer = document.getElementById(`${prefix}ProgressContainer`);
    const progressBar = document.getElementById(`${prefix}ProgressBar`);
    
    if (!zone) {
        console.error(`错误: 未找到上传区域元素 #${prefix}UploadZone`);
        return;
    }
    zone.classList.remove('upload-success', 'upload-loading', 'upload-error', 'border-primary', 'bg-blue-100');
    if (status === 'selected') {
        if (initialDiv) initialDiv.classList.add('hidden');
        if (infoDiv) infoDiv.classList.remove('hidden');
        if (nameSpan) nameSpan.textContent = file.name;
        if (sizeSpan) sizeSpan.textContent = formatFileSize(file.size);
        if (statusSpan) {
            statusSpan.textContent = '已选择，点击下方按钮开始上传';
            statusSpan.className = 'text-xs text-gray-dark';
        }
        if (progressContainer) progressContainer.classList.add('hidden');
        if (progressBar) progressBar.style.width = '0%';
    } else if (status === 'uploading') {
        zone.classList.add('upload-loading');
        if (statusSpan) {
            statusSpan.textContent = '上传中...';
            statusSpan.className = 'text-xs text-primary';
        }
        if (progressContainer) progressContainer.classList.add('hidden');
        if (progressBar) progressBar.style.width = '0%';
    } else if (status === 'success') {
        zone.classList.add('upload-success');
        if (statusSpan) {
            statusSpan.textContent = '上传成功！';
            statusSpan.className = 'text-xs text-success';
        }
        if (progressBar) progressBar.style.width = '100%';
        setTimeout(() => resetUploadZone(type), 3000);
    } else if (status === 'error') {
        zone.classList.add('upload-error');
        if (statusSpan) {
            statusSpan.textContent = '上传失败，请重试';
            statusSpan.className = 'text-xs text-danger';
        }
        if (progressBar) progressBar.style.width = '0%';
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function resetUploadZone(type) {
    const prefix = type === 'table' ? 'table' : 'archive';
    const zone = document.getElementById(`${prefix}UploadZone`);
    const initialDiv = document.getElementById(`${prefix}UploadInitial`);
    const infoDiv = document.getElementById(`${prefix}FileInfo`);
    const fileInput = document.getElementById(`${prefix}FileInput`);
    if (zone) zone.classList.remove('upload-success', 'upload-loading', 'upload-error');
    if (initialDiv) initialDiv.classList.remove('hidden');
    if (infoDiv) infoDiv.classList.add('hidden');
    if (fileInput) fileInput.value = '';
}

// 打开图片放大预览
function openImgModal(imgUrl) {
    const modal = document.getElementById('imgModal');
    const modalImg = document.getElementById('modalImg');
    if (imgUrl && !imgUrl.startsWith('http') && !imgUrl.startsWith('data:') && !imgUrl.startsWith('/')) {
        imgUrl = '/' + imgUrl;
    }
    modalImg.src = imgUrl;
    modal.classList.add('active');
    modalImg.onerror = function() {
        console.error('图片加载失败:', imgUrl);
        modalImg.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjYwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjBmMGYwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXNpemU9IjE2IiBmaWxsPSIjZmYwMDAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+SW1hZ2UgTG9hZGluZyBGYWlsZWQ8L3RleHQ+PHRleHQgeD0iNTAlIiB5PSI2MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyIiBmaWxsPSIjNjY2IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iMS4zZW0iPiR7aW1nVXJsfTwvdGV4dD48L3N2Zz4=';
    };
    modal.onclick = function(e) {
        if (e.target === modal) closeImgModal();
    };
}

// 关闭图片放大预览
function closeImgModal() {
    const modal = document.getElementById('imgModal');
    modal.classList.remove('active');
    document.getElementById('modalImg').src = '';
}

// 自动执行决策切换逻辑
async function toggleAutoProcess() {
    isAutoProcess = document.getElementById('autoProcessToggle').checked;
    if (isAutoProcess) {
        try {
            await sendDecisionSignal('a');
            alert('已开启自动执行决策，并发送信号 a');
        } catch (error) {
            alert(`自动执行信号发送失败: ${error.message}`);
        }
    } else {
        alert('已关闭自动执行决策（仅前端展示开关关闭）');
    }
}

// 全选功能
function toggleCheckAll() {
    const checkAll = document.getElementById('checkAllCheckbox');
    const checkboxes = document.querySelectorAll('.row-checkbox');
    checkboxes.forEach(cb => cb.checked = checkAll.checked);
    updateSelectedCount();
}

// 更新选中数量
function updateSelectedCount() {
    const checkboxes = document.querySelectorAll('.row-checkbox:checked');
    document.getElementById('selectedCount').textContent = checkboxes.length;
}

// 批量处理
function batchProcess(action) {
    const selectedCount = document.querySelectorAll('.row-checkbox:checked').length;
    if (selectedCount === 0) {
        alert('Please select at least one item to process.');
        return;
    }
    alert(`已${action === 'accept' ? '接受' : '拒绝'}选中的 ${selectedCount} 条决策`);
}

// 单个处理
function processSingle(action, id) {
    alert(`Decision for ID ${id} has been ${action === 'accept' ? 'accepted' : 'rejected'}.`);
}

// 打开上传侧边栏
function openUploadSidebar() {
    document.getElementById('uploadSidebar').classList.add('active');
    document.getElementById('sidebarOverlay').classList.add('active');
    switchUploadType('table');
    const targetDatasetButtons = document.querySelectorAll('.target-dataset-btn');
    targetDatasetButtons.forEach(btn => {
        if (currentTargetDataset === 'test' && btn.textContent.includes('Test Set')) {
            btn.classList.add('active');
        } else if (currentTargetDataset === 'val' && btn.textContent.includes('Validation Set')) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// 关闭上传侧边栏
function closeUploadSidebar() {
    document.getElementById('uploadSidebar').classList.remove('active');
    document.getElementById('sidebarOverlay').classList.remove('active');
}

// 下载结果功能
async function downloadResults() {
    try {
        // 使用固定的数据集名称
        const datasetName = 'adult';
        
        console.log(`开始下载 ${datasetName} 数据集的最佳清洗结果...`);
        
        // 1. 先检查文件是否存在
        const checkResponse = await fetch(`/api/check-best-data?dataset=${datasetName}`);
        const checkResult = await checkResponse.json();
        
        console.log("文件检查结果:", checkResult);
        
        if (!checkResult.exists) {
            alert('最佳数据文件不存在，请先运行"Data Cleaning"任务生成清洗结果。');
            return;
        }
        
        // 2. 显示确认对话框
        if (!confirm(`确定要下载 ${datasetName} 数据集的最佳清洗结果吗？\n文件大小: ${checkResult.file_info?.file_size_kb || '未知'} KB`)) {
            return;
        }
        
        // 3. 创建下载链接
        const downloadUrl = `/api/download-best-data?dataset=${datasetName}`;
        console.log("下载链接:", downloadUrl);
        
        // 4. 触发下载
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `best_data.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // 5. 显示成功消息
        setTimeout(() => {
            alert('下载已开始，请查看浏览器下载列表。\n如果下载没有自动开始，请检查浏览器设置。');
        }, 500);
        
    } catch (error) {
        console.error("下载失败:", error);
        alert(`下载失败: ${error.message}\n请检查控制台获取更多信息。`);
    }
}

// 运行质量检测并加载结果
// 质量检测功能
async function runQualityCheckAndLoad(triggerButton = null) {
    const button = triggerButton;
    if (!button) {
        alert('未找到触发按钮，无法开始质量检测。');
        return;
    }
    const originalText = button.innerHTML;
    const originalClass = button.className;
    
    try {
        // 1. 按钮状态反馈：禁用并显示加载中
        button.disabled = true;
        button.className = originalClass.replace('btn-warning', 'btn-primary') + ' opacity-70 cursor-not-allowed';
        button.innerHTML = '<i class="fa fa-spinner fa-spin"></i> 检测中...';
        
        console.log("开始质量检测...");
        
        const response = await fetch('/api/run-quality-check-and-load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ label_col: currentTargetColumn || 'income' })
        });
        
        // 检查响应内容类型
        const contentType = response.headers.get('content-type');
        console.log("响应类型:", contentType);
        
        if (!contentType || !contentType.includes('application/json')) {
            // 如果不是JSON，读取原始文本查看问题
            const text = await response.text();
            console.error("返回的不是JSON，而是:", text.substring(0, 500));
            
            // 恢复按钮状态
            button.disabled = false;
            button.className = originalClass;
            button.innerHTML = originalText;
            
            alert(`质量检测失败：服务器返回错误页面\n请查看Flask控制台错误信息`);
            return;
        }
        
        const result = await response.json();
        
        // 恢复按钮状态
        button.disabled = false;
        button.className = originalClass;
        button.innerHTML = originalText;
        
        if (!response.ok) {
            throw new Error(result.error || '质量检测失败');
        }
        
        console.log("质量检测结果:", result);
        
        if (result.success) {
            // 更新统计信息
            if (result.quality_stats) {
                updateStatisticsFromResult(result.quality_stats);
            }
            
            // 简化的成功提示
            alert('质量检测已完成');
            
            // 处理可视化结果
            if (result.visualization && result.visualization.success && result.visualization.images.length > 0) {
                window.generatedChartImages = result.visualization.images || [];
                console.log("生成的图表图片:", window.generatedChartImages);
                
                // 自动显示第一张图表
                if (window.generatedChartImages.length > 0) {
                    displayGeneratedChart('distribution');
                }
            } else if (result.visualization && !result.visualization.success) {
                console.warn("可视化生成失败:", result.visualization.message);
            }
            
        } else {
            alert('质量检测失败: ' + (result.error || '未知错误'));
        }
        
    } catch (error) {
        console.error('质量检测失败:', error);
        alert(`质量检测失败: ${error.message}`);
        
        // 恢复按钮状态
        if (button) {
            button.disabled = false;
            button.className = originalClass;
            button.innerHTML = originalText;
        }
    }
}

// 新增：从质量检测结果更新统计数据
function updateStatisticsFromResult(qualityStats) {
    const metricValues = document.querySelectorAll('.metric-value');
    if (metricValues.length >= 6) {
        metricValues[0].textContent = qualityStats.n_samples ? qualityStats.n_samples.toLocaleString() : '0';
        metricValues[1].textContent = qualityStats.n_features || '0';
        metricValues[2].textContent = qualityStats.n_classes || '0';
        metricValues[3].textContent = qualityStats.missing_rate || '0%';
        metricValues[4].textContent = qualityStats.feature_noise_ratio || '0%';
        metricValues[5].textContent = qualityStats.label_noise_ratio || '0%';
    }
    console.log('已更新质量统计数据:', qualityStats);
}

// 修改后的 updateStatisticsFromResult 函数
function updateStatisticsFromResult(stats) {
    const metricValues = document.querySelectorAll('.metric-value');
    if (metricValues.length >= 6) {
        metricValues[0].textContent = stats.n_samples ? stats.n_samples.toLocaleString() : 'N/A';
        metricValues[1].textContent = stats.n_features || 'N/A';
        metricValues[2].textContent = stats.n_classes || 'N/A';
        metricValues[3].textContent = stats.missing_rate || 'N/A';
        metricValues[4].textContent = stats.feature_noise_ratio || 'N/A';
        metricValues[5].textContent = stats.label_noise_ratio || 'N/A';
    }
}

// 组合功能：质量检测 + 数据增强
async function runQualityCheckAndAugmentation(triggerButton) {
    await runDataAugmentation(triggerButton);
    await runQualityCheckAndLoad(triggerButton);
}

// 数据增强功能
async function runDataAugmentation(triggerButton = null) {
    const button = triggerButton;
    const originalText = button ? button.innerHTML : '';
    
    try {
        // 按钮状态反馈
        if (button) {
            button.disabled = true;
            button.innerHTML = '<i class="fa fa-spinner fa-spin"></i> 增强中...';
        }
        
        console.log("开始数据增强...");
        
        const response = await fetch('/api/run-data-augmentation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const result = await response.json();
        
        console.log('数据增强API响应:', result);
        
        // 在main.js的runDataAugmentation函数中修改
        if (result.success) {
            // 显示成功消息
            alert(`数据增强完成！生成了 ${result.generated_count} 个样本，已添加到候选池。`);
            
            // 自动切换到候选池查看生成的数据
            if (typeof switchDisplayDataset === 'function') {
                switchDisplayDataset('pool');
            }
            
            // 重新加载候选池数据
            await loadPoolData();
            
            // 重新渲染数据展示区域
            if (typeof renderDisplayData === 'function') {
                setTimeout(() => {
                    renderDisplayData();
                }, 100);
            }
            
            // 更新统计数据
            updateStatistics();
        } else {
            alert('数据增强失败: ' + (result.error || '未知错误'));
        }
        
    } catch (error) {
        console.error('数据增强请求失败:', error);
        alert('网络请求失败: ' + error.message);
    } finally {
        // 恢复按钮状态
        if (button) {
            button.disabled = false;
            button.innerHTML = originalText;
        }
    }
}

// 加载候选池数据
async function loadPoolData() {
    try {
        const response = await fetch('/api/pool-data');
        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                console.log(`加载了 ${result.count} 条候选池数据`);
                // 将数据存储到前端全局变量
                poolData = result.data || [];
                console.log("候选池第一条数据:", poolData[0]);
            }
        }
    } catch (error) {
        console.error('加载候选池数据失败:', error);
    }
}

/**
 * 更新目标列选择下拉框
 */
function updateLabelColumnSelect(dataSample) {
    const select = document.getElementById('labelColumnSelect');
    if (!select) return;
    
    select.innerHTML = '<option value="">-- Select a column --</option>';
    select.disabled = true;
    
    if (!dataSample || Object.keys(dataSample).length === 0) {
        return;
    }
    
    // 假设dataSample是一个对象，其键是列名
    const columns = Object.keys(dataSample);
    columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        select.appendChild(option);
    });
    
    select.disabled = false;
    // 可选：尝试自动选择名为'label', 'target', 'y', 'class', 'income'的列
    const autoSelectKeys = ['income', 'label', 'target', 'y', 'class'];
    for (const key of autoSelectKeys) {
        if (columns.includes(key)) {
            select.value = key;
            break;
        }
    }
}

