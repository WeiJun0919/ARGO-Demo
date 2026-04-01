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
let currentVisualizationData = null;


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
const CLEANING_POLL_INTERVAL_MS = 1000; // 状态轮询间隔
let isFetchingCleaningStatus = false; // 防止轮询重入
let isCleaningRunning = false;  // 标记训练是否正在进行
let backendTrainingRunning = false; // 后端返回的训练运行状态
let cleaningFinishStableCount = 0; // 训练结束稳定计数，避免瞬时误判
let pendingDecisionData = null; // pending_decision.json 当前内容
let pendingSampleDecisions = {}; // { sample_id: 'accept' | 'reject' }
let currentPendingDecisionKey = null; // `${iteration}-${step}` 用于切换任务时重置状态
let timelineCollapsedIterations = {}; // 时间线折叠状态
let lastTimelineIteration = null; // 用于控制是否自动滚动
let timelineHistoryStore = {}; // { [iteration]: { iteration, time, steps: [] } }
let cleaningSessionStartMs = null; // 当前一次 Data Governance 启动时间（用于过滤旧 pending）

// function resetCleaningPanelDisplay() {
//     updateStatisticsFromResult({
//         n_samples: 0,
//         n_features: 0,
//         n_classes: 0,
//         missing_rate: '0%',
//         feature_noise_ratio: '0%',
//         label_noise_ratio: '0%',
//         classification_accuracy: '0%'
//     });
//     currentVisualizationData = null;
//     showEmptyChartHint('Processing...', 'Statistics update after data governance completes');
// }

function setDataCleaningButtonState(isRunning) {
    const btn = document.getElementById('dataCleaningBtn');
    if (!btn) return;

    if (isRunning) {
        btn.disabled = true;
        btn.classList.add('opacity-70', 'cursor-not-allowed');
        btn.innerHTML = '<i class="fa fa-spinner fa-spin mr-1"></i>Cleaning...';
    } else {
        btn.disabled = false;
        btn.classList.remove('opacity-70', 'cursor-not-allowed');
        btn.innerHTML = '<i class="fa fa-shower mr-1"></i>Data Governance';
    }
}
// ****************************************************************

// 页面加载初始化
window.onload = function() {
    // 页面刷新后重置时间线内存池与会话起点
    timelineHistoryStore = {};
    timelineCollapsedIterations = {};
    lastTimelineIteration = null;
    cleaningSessionStartMs = null;

    initChart();
    initUploadTablePagination();
    initDecisionTablePagination();
    changeUploadPage(1);
    changeDecisionPage(1);
    renderDisplayData();
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
        if (btn.textContent.includes('Training Set')) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    window.addEventListener('resize', function() {
        if (dataChart) dataChart.resize();
    });
    
    bindFileSelectionEvents();

    // 页面初始化仅拉取一次状态用于渲染，不自动开启轮询
    // fetchAndUpdateCleaningStatus();
    
    // ****************** 新增：页面卸载时清理轮询定时器 ******************
    window.addEventListener('beforeunload', function() {
        stopCleaningStatusPolling();
    });
    // ****************************************************************
};

// ****************** 新增：数据清洗任务核心函数 ******************

async function runDataCleaning() {
    if (isCleaningRunning) {
        alert('Data Governance is already running. Please do not start it again.');
        return;
    }

    if (!confirm('Are you sure you want to start data cleaning? This will start the RL training process.')) {
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
            throw new Error(result.error || 'Failed to start');
        }

        alert(result.message || 'Data Governance has started!');
        isCleaningRunning = true;
        cleaningFinishStableCount = 0;
        setDataCleaningButtonState(true);

        // 新会话开始：记录起点并清空旧时间线，避免显示历史 pending
        cleaningSessionStartMs = Date.now();
        timelineHistoryStore = {};
        timelineCollapsedIterations = {};
        lastTimelineIteration = null;
        renderEmptyPendingDecisionTable('Training started. Waiting for new pending samples...');
        updateTimelineWithEmptyStatus();
        // resetCleaningPanelDisplay();

        // 启动状态轮询
        startCleaningStatusPolling();
        // 更新右侧时间线标题，提示任务开始
        updateTimelineWithStart();

    } catch (error) {
        console.error('Failed to start data cleaning:', error);
        alert(`Failed to start: ${error.message}`);
    }
}

/**
 * 开始轮询获取数据清洗状态
 */
function startCleaningStatusPolling() {
    if (cleaningStatusTimer) {
        clearInterval(cleaningStatusTimer);
    }
    // 更快的轮询频率
    cleaningStatusTimer = setInterval(fetchAndUpdateCleaningStatus, CLEANING_POLL_INTERVAL_MS);
    // 立即获取一次
    fetchAndUpdateCleaningStatus();
}

/**
 * 获取并更新数据清洗状态
 */
async function fetchAndUpdateCleaningStatus() {
    if (isFetchingCleaningStatus) return;
    isFetchingCleaningStatus = true;

    try {
        const ts = Date.now();
        const labelCol = currentTargetColumn || 'income';
        const [statusResp, pendingResp] = await Promise.all([
            fetch(`/api/get-cleaning-status?dataset=adult&label_col=${encodeURIComponent(labelCol)}&_ts=${ts}`, { cache: 'no-store' }),
            fetch(`/api/pending-decision?dataset=adult&_ts=${ts}`, { cache: 'no-store' })
        ]);

        // pending 优先：确保决策表不被 status 接口失败牵连
        let pendingResult = null;
        try {
            pendingResult = await pendingResp.json();
        } catch (e) {
            pendingResult = { success: false, error: 'Pending response is not valid JSON' };
        }

        if (!pendingResp.ok) {
            console.warn('Failed to fetch pending decisions:', pendingResult?.error || pendingResp.statusText);
        }

        // status 先解析，用于判断后端是否真的在训练中
        let statusResult = null;
        try {
            statusResult = await statusResp.json();
        } catch (e) {
            statusResult = { success: false, error: 'Status response is not valid JSON' };
        }
        backendTrainingRunning = Boolean(statusResult && statusResult.running);

        // 判定 pending 是否属于“本次清洗会话”
        const parseDateTimeToMs = (dt) => {
            if (!dt || typeof dt !== 'string') return NaN;
            const ms = Date.parse(dt.replace(' ', 'T'));
            return Number.isFinite(ms) ? ms : NaN;
        };
        const pendingFileMtimeMs = parseDateTimeToMs(statusResult?.status?.file_mtime || '');
        const isPendingFreshForCurrentSession =
            !cleaningSessionStartMs ||
            !Number.isFinite(pendingFileMtimeMs) ||
            pendingFileMtimeMs >= (cleaningSessionStartMs - 1000);

        // 先独立处理 pending 决策渲染：仅在训练运行中展示，避免旧文件残留
        try {
            const prevPendingData = pendingDecisionData;
            const nextPendingData = (pendingResult && pendingResult.success) ? pendingResult.data : null;
            const hasSamples = !!(nextPendingData && Array.isArray(nextPendingData.samples) && nextPendingData.samples.length > 0);
            const pendingFlag = (pendingResult && pendingResult.success && pendingResult.pending === true) || (nextPendingData && nextPendingData.pending === true);
            const pendingValid = !!(pendingFlag && hasSamples && nextPendingData.decision == null);

            // 允许在非训练中也展示有效 pending，避免“文件存在但表格不显示”
            const allowPendingDisplay = pendingValid && isPendingFreshForCurrentSession;

            if (allowPendingDisplay && pendingValid) {
                pendingDecisionData = nextPendingData;
                updateDecisionTableWithPending(pendingDecisionData);
            } else if (allowPendingDisplay && prevPendingData && Array.isArray(prevPendingData.samples) && prevPendingData.samples.length > 0) {
                // 训练进行中，文件短暂切换/写入时保留上一帧，避免“个别 step 表格空白闪烁”
                pendingDecisionData = prevPendingData;
                updateDecisionTableWithPending(pendingDecisionData);
            } else {
                pendingDecisionData = null;
                updateSelectedCount();
                renderEmptyPendingDecisionTable(backendTrainingRunning ? 'Training started. Waiting for new pending samples...' : 'No active training session or no pending samples at the moment.');
            }

            console.log('[pending-debug]', {
                backend_running: backendTrainingRunning,
                api_pending: pendingResult?.pending,
                file_pending: nextPendingData?.pending,
                decision: nextPendingData?.decision,
                sample_count: Array.isArray(nextPendingData?.samples) ? nextPendingData.samples.length : 0,
                pendingValid
            });
        } catch (err) {
            console.error('Failed to render pending decision table:', err);
        }

        // status 仅用于时间线与训练结束判断；失败时不影响决策表
        if (!statusResp.ok) {
            console.warn('Failed to fetch status:', statusResult?.error || statusResp.statusText);
            updateTimelineWithEmptyStatus();
        } else if (statusResult.success) {
            const status = statusResult.status;
            const decisionTable = statusResult.decision_table || [];
            const panelPayload = statusResult.panel || null;
            const backendRunning = Boolean(statusResult.running);

            // 页面刷新后若训练仍在进行，自动恢复轮询与按钮状态
            if (backendRunning && !isCleaningRunning) {
                isCleaningRunning = true;
                cleaningFinishStableCount = 0;
                setDataCleaningButtonState(true);
            }
            if (backendRunning && !cleaningStatusTimer) {
                cleaningStatusTimer = setInterval(fetchAndUpdateCleaningStatus, CLEANING_POLL_INTERVAL_MS);
            }

            // 仅在真正结束并稳定后统一刷新面板，避免中途/旧数据污染
            const hasStatus = !!status;
            const hasDecisionSteps = Array.isArray(decisionTable) && decisionTable.length > 0;

            // 只要 pending_decision 有 status 或 step 就渲染时间线
            if (hasStatus || hasDecisionSteps) {
                const parseDateTimeToMs = (dt) => {
                    if (!dt || typeof dt !== 'string') return NaN;
                    const ms = Date.parse(dt.replace(' ', 'T'));
                    return Number.isFinite(ms) ? ms : NaN;
                };

                const statusFileMtimeMs = parseDateTimeToMs(status?.file_mtime || '');
                const isStatusFreshForCurrentSession =
                    !cleaningSessionStartMs ||
                    !Number.isFinite(statusFileMtimeMs) ||
                    statusFileMtimeMs >= (cleaningSessionStartMs - 1000);

                if (isStatusFreshForCurrentSession) {
                    updateTimelineWithIteration(status || {}, decisionTable, pendingDecisionData);
                } else {
                    // 当前会话启动后，忽略旧的 joint_latest 快照，避免出现“第二轮先于第一轮”
                    updateTimelineWithEmptyStatus();
                }
            } else {
                updateTimelineWithEmptyStatus();
            }

            // 轮询停止条件：后端连续多次报告“已结束且无待确认决策”，避免瞬时误判
            const hasPendingToConfirm = !!(pendingResult && pendingResult.success && pendingResult.pending === true);
            if (!backendRunning && isCleaningRunning && !hasPendingToConfirm) {
                cleaningFinishStableCount += 1;
            } else {
                cleaningFinishStableCount = 0;
            }

            if (cleaningFinishStableCount >= 3 && !backendRunning && isCleaningRunning && !hasPendingToConfirm) {
                stopCleaningStatusPolling();
                isCleaningRunning = false;
                cleaningFinishStableCount = 0;
                setDataCleaningButtonState(false);
                pendingDecisionData = null;
                pendingSampleDecisions = {};
                renderEmptyPendingDecisionTable('Training finished. Historical pending decisions are hidden.');

                if (hasStatus && status.total_iterations > 0) {
                    const finalAcc = Number(status.best_accuracy ?? status.accuracy ?? 0);
                    alert(`Completed！Final Accuracy： ${(finalAcc * 100).toFixed(2)}%`);
                }

                if (panelPayload && panelPayload.quality_stats) {
                    updateStatisticsFromResult(panelPayload.quality_stats);
                }
                if (panelPayload && panelPayload.visualization && panelPayload.visualization.success && panelPayload.visualization.data) {
                    currentVisualizationData = panelPayload.visualization.data;
                    renderCurrentVisualizationChart();
                }

                if (currentDisplayDataset === 'result') {
                    loadResultData();
                }
            }
        } else {
            console.warn('Status fetched but data is abnormal:', statusResult?.message || statusResult?.error);
            updateTimelineWithEmptyStatus();
        }
    } catch (error) {
        console.error('Failed to fetch data cleaning status:', error);
    } finally {
        isFetchingCleaningStatus = false;
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
            'modify_features': 'Modify Features',
            'modify_labels': 'Modify Labels',
            'delete_samples': 'Delete Samples',
            'add_samples': 'Add Samples',
            'no_op': 'No Operation',
            'modlify_features': 'Modify Features',
            'modlify_labels': 'Modify Labels',
            'no-op': 'No Operation'
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
function renderEmptyPendingDecisionTable(message = 'No pending decisions') {
    const tbody = document.getElementById('decisionTableBody');
    if (!tbody) return;
    tbody.innerHTML = `<tr><td colspan="9" class="text-center py-4 text-gray-dark">${message}</td></tr>`;
    updateSelectedCount();
}

// 辅助函数：将动作映射为操作类型
function mapActionToOperation(action) {
    const map = {
        'modify_features': 'Repair',
        'modify_labels': 'Relabel',
        'delete_samples': 'Remove',
        'add_samples': 'Augment',
        'no_op': 'No-op',
        // 'modlify_features': 'Repair',
        // 'modlify_labels': 'Relabel',
        // 'modlify_label': 'Relabel',
        // 'no-op': 'No-Op',
        // 'Repair': 'Repair',
        // 'Relabel': 'Relabel',
        // 'Augment': 'Add',
        // 'Remove': 'Delete',
        // 'Add': 'Add',
        // 'Delete': 'Delete',
        // 'No-Op': 'No-Op'
    };
    const mapped = map[action] || action;
    return formatTimelineOperation(mapped);
}

function formatTimelineOperation(op) {
    const map = {
        'Repair': 'Repair',
        'Relabel': 'Relabel',
        'Add': 'Augment',
        'Augment': 'Augment',
        'Delete': 'Remove',
        'Remove': 'Remove',
        'No-Op': 'No-op'
    };
    return map[op] || op;
}

// 辅助函数：根据动作生成问题详情
// function getIssueDetailByAction(sample) {
//     switch (sample.action) {
//         case 'modlify_features':
//             return '特征异常值';
//         case 'modlify_labels':
//             return sample.label_changed === 1 ? '标签可能错误' : '标签检查通过';
//         case 'add_samples':
//             return '从候选池添加';
//         case 'delete_samples':
//             return '疑似噪声样本';
//         default:
//             return '无操作';
//     }
// }

// 辅助函数：获取原始值
// function getOriginalValueByAction(sample) {
//     switch (sample.action) {
//         case 'modlify_features':
//             return sample.original_features_preview || 'N/A';
//         case 'modlify_labels':
//             return `Label: ${sample.original_label}`;
//         case 'add_samples':
//             return `Label: ${sample.original_label}`;
//         default:
//             return 'N/A';
//     }
// }

// // 辅助函数：获取建议值
// function getSuggestedValueByAction(sample) {
//     switch (sample.action) {
//         case 'modlify_features':
//             return '特征已修正';
//         case 'modlify_labels':
//             return `Predicted: ${sample.predicted_label} (${sample.label_changed ? 'Changed' : 'Unchanged'})`;
//         case 'add_samples':
//             return '已加入训练集';
//         case 'delete_samples':
//             return '将删除';
//         default:
//             return 'N/A';
//     }
// }

// 辅助函数：根据操作类型获取颜色
function getColorForOperation(op) {
    const colorMap = {
        'Repair': '#FF7D00', // 橙色 -> 对应 .btn-warning
        'Relabel': '#1a73e8', // 蓝色 -> 对应 .btn-primary
        'Augment': '#00B42A', // 绿色 -> 对应 .btn-success
        'Add': '#00B42A', // 绿色 -> 同 Augment
        'Remove': '#F53F3F', // 红色 -> 对应 .btn-danger
        'Delete': '#F53F3F', // 红色 -> 同 Remove
        'No-Op': '#94A3B8' // 灰色
    };
    return colorMap[op] || '#94A3B8';
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

function updateTimelineWithIteration(status, decisionTable, _pendingDecisionData = null) {
    status = status || {};
    const timelineContainer = document.querySelector('.timeline-container');
    if (!timelineContainer) return;

    const safeNumber = (value, defaultValue = 0) => {
        const n = Number(value);
        return Number.isFinite(n) ? n : defaultValue;
    };

    const currentIteration = safeNumber(status.iteration);
    const currentTime = status.timestamp || status.file_mtime || '--';

    if (currentIteration > 0) {
        if (!timelineHistoryStore[currentIteration]) {
            timelineHistoryStore[currentIteration] = {
                iteration: currentIteration,
                time: currentTime,
                best_accuracy: safeNumber(status.best_accuracy, NaN),
                model_accuracy: safeNumber(status.accuracy, NaN),
                accuracy_improvement: safeNumber(status.accuracy_improvement, 0),
                action_distribution: status.action_distribution || {},
                steps: []
            };
        }

        timelineHistoryStore[currentIteration].time = currentTime;
        timelineHistoryStore[currentIteration].best_accuracy = safeNumber(status.best_accuracy, NaN);
        timelineHistoryStore[currentIteration].model_accuracy = safeNumber(status.accuracy, NaN);
        timelineHistoryStore[currentIteration].accuracy_improvement = safeNumber(status.accuracy_improvement, 0);
        timelineHistoryStore[currentIteration].action_distribution = status.action_distribution || {};

        if (Array.isArray(decisionTable) && decisionTable.length > 0) {
            decisionTable.forEach((stepData) => {
                const stepNo = safeNumber(stepData.step, -1);
                if (stepNo < 0) return;

                const stepPayload = {
                    step: stepNo,
                    action: stepData.action || '--',
                    n_selected: safeNumber(stepData.n_selected),
                    label_noise_ratio: safeNumber(stepData.label_noise_ratio, 0),
                    feature_noise_ratio: safeNumber(stepData.feature_noise_ratio, 0),
                    class_ratio: stepData.class_ratio || '--'
                };

                const existedIdx = timelineHistoryStore[currentIteration].steps.findIndex(s => safeNumber(s.step, -1) === stepNo);
                if (existedIdx >= 0) {
                    timelineHistoryStore[currentIteration].steps[existedIdx] = stepPayload;
                } else {
                    timelineHistoryStore[currentIteration].steps.push(stepPayload);
                }
            });
        }
    }

    const formatActionName = (action) => {
        const actionMap = {
            'modify_features': 'Repair',
            'modify_labels': 'Relabel',
            'delete_samples': 'Remove',
            'add_samples': 'Augment',
            'no_op': 'No-op',
            'modlify_features': 'Repair',
            'modlify_labels': 'Relabel',
            'modlify_label': 'Relabel',
            'Add': 'Augment',
            'Delete': 'Remove',
        };
        return actionMap[action] || action;
    };


    const buildActionDistributionRows = (dist) => {
        const actionKeys = ['add_samples', 'delete_samples', 'modify_features', 'modify_labels', 'no_op'];
        const badges = actionKeys.map((rawKey) => {
            const displayName = formatActionName(rawKey);
            const v = dist && typeof dist === 'object'
                ? safeNumber(dist[rawKey], safeNumber(dist[displayName], 0))
                : 0;
            const color = getColorForOperation(displayName);
            return `
                <span
                    class="diff-badge"
                    style="
                        background:${color};
                        color:#fff;
                        border-radius:10px;
                        padding:4px 8px;
                        font-weight:600;
                        font-size:0.63rem;
                        letter-spacing:0.1px;
                        box-shadow:0 3px 10px rgba(15, 23, 42, 0.14);
                        display:inline-flex;
                        align-items:center;
                        gap:4px;
                    "
                >${displayName} ${Math.round(v * 100)}%</span>
            `;
        });

        return `
            <div class="timeline-content-row" style="margin-top:6px;">
                <div style="display:flex;flex-wrap:wrap;gap:8px 8px;align-items:center;">
                    ${badges.join('')}
                </div>
            </div>
        `;
    };

    let timelineHTML = `
        <div class="timeline-title">
            <i class="fa fa-history"></i>RL Timeline
        </div>
    `;

    const iterations = Object.values(timelineHistoryStore)
        .sort((a, b) => safeNumber(a.iteration) - safeNumber(b.iteration));

    if (iterations.length === 0) {
        timelineHTML += `
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-iteration-title">Waiting for timeline updates</div>
                <div class="timeline-content-row text-gray-dark"><i>The timeline will appear automatically after training begins.</i></div>
            </div>
        `;
    } else {
        iterations.forEach((iterItem, iterIndex) => {
            const iterationKey = `iter-${iterItem.iteration}`;
            const isCollapsed = timelineCollapsedIterations[iterationKey] !== undefined
                ? !!timelineCollapsedIterations[iterationKey]
                : true;
            const hasNextIteration = iterIndex < iterations.length - 1;

            const sortedSteps = Array.isArray(iterItem.steps)
                ? [...iterItem.steps].sort((a, b) => safeNumber(a.step) - safeNumber(b.step))
                : [];

            const accImprove = safeNumber(iterItem.accuracy_improvement, 0);
            const accImproveText = `${accImprove >= 0 ? '+' : ''}${(accImprove * 100).toFixed(2)}%`;
            const accImproveClass = accImprove > 0 ? 'accuracy-change-positive' : (accImprove < 0 ? 'accuracy-change-negative' : 'accuracy-change-neutral');

            timelineHTML += `
                <div class="timeline-item">
                    <div class="timeline-dot" style="background:#165DFF;"></div>
                    ${hasNextIteration ? '<div class="timeline-line"></div>' : ''}
                    <div class="timeline-iteration-title" style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;" onclick="toggleTimelineIteration('${iterationKey}')">
                        <span>Iteration ${iterItem.iteration}</span>
                        <span class="badge" style="font-size:10px;">${isCollapsed ? '▶ Expand Steps' : '▼ Collapse Steps'}</span>
                    </div>
                    <div class="timeline-content-row" style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                        <strong>Model Accuracy</strong>
                        <span style="display:flex;align-items:center;gap:8px;">
                            <span>${Number.isFinite(safeNumber(iterItem.model_accuracy, NaN)) ? `${(safeNumber(iterItem.model_accuracy, 0) * 100).toFixed(2)}%` : '--'}</span>
                            <span class="${accImproveClass}" style="display:inline-block;">${accImproveText}</span>
                        </span>
                    </div>
                    <div class="timeline-content-row" style="margin-top:8px;">
                        <strong>Distribution of Actions：</strong>
                    </div>
                    ${buildActionDistributionRows(iterItem.action_distribution)}
                </div>
            `;

            if (!isCollapsed) {
                if (sortedSteps.length > 0) {
                    sortedSteps.forEach((stepData, stepIndex) => {
                        const isLatestStep = stepIndex === sortedSteps.length - 1;

                        timelineHTML += `
                            <div class="timeline-item timeline-step-item">
                                <div class="timeline-dot" style="background: ${isLatestStep ? '#00B42A' : '#86909C'};"></div>
                                ${stepIndex < sortedSteps.length - 1 ? '<div class="timeline-line"></div>' : ''}
                                <div class="timeline-step-header">
                                    <div class="timeline-step-title">Step ${safeNumber(stepData.step)} ${isLatestStep ? '<span class="badge">Latest</span>' : ''}</div>
                                    <div class="timeline-step-action">${formatActionName(stepData.action || '--')}</div>
                                </div>
                                <div class="timeline-step-grid">
                                    <div class="timeline-content-row"><strong>Affected Samples:</strong> ${safeNumber(stepData.n_selected)}</div>
                                    <div class="timeline-content-row"><strong>Label Noise Ratio:</strong> ${(safeNumber(stepData.label_noise_ratio) * 100).toFixed(2)}%</div>
                                    <div class="timeline-content-row"><strong>Feature Noise Ratio:</strong> ${(safeNumber(stepData.feature_noise_ratio) * 100).toFixed(2)}%</div>
                                    <div class="timeline-content-row"><strong>Class Distribution:</strong> ${stepData.class_ratio || '--'}</div>
                                </div>
                            </div>
                        `;
                    });
                } else {
                    timelineHTML += `
                        <div class="timeline-item">
                            <div class="timeline-dot"></div>
                            <div class="timeline-iteration-title">No steps recorded for this iteration</div>
                        </div>
                    `;
                }
            }
        });
    }

    const shouldStickBottom = Math.abs((timelineContainer.scrollTop + timelineContainer.clientHeight) - timelineContainer.scrollHeight) < 24;
    const isNewIteration = lastTimelineIteration === null || currentIteration !== lastTimelineIteration;

    timelineContainer.innerHTML = timelineHTML;

    if (shouldStickBottom || isNewIteration) {
        timelineContainer.scrollTop = timelineContainer.scrollHeight;
    }

    lastTimelineIteration = currentIteration;
}

function updateTimelineWithStart() {
    // 仅在没有历史轮次时展示启动占位，避免覆盖真实时间线
    if (Object.keys(timelineHistoryStore || {}).length > 0) return;

    const timelineContainer = document.querySelector('.timeline-container');
    if (timelineContainer) {
        const startHTML = `
            <div class="timeline-title">
                <i class="fa fa-history"></i>RL Timeline
            </div>
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-line"></div>
                <div class="timeline-iteration-title">Training is starting</div>
                <div class="timeline-content-row">
                    <strong>Status:</strong> Initializing the RL cleaning pipeline...
                </div>
                <div class="timeline-content-row">
                    <strong>Info:</strong> Waiting for the first iteration update.
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
            <i class="fa fa-history"></i>RL Timeline
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-iteration-title">Waiting  for RL  timeline</div>
        </div>
    `;
}

function toggleTimelineIteration(iteration) {
    const currentVal = timelineCollapsedIterations[iteration] !== undefined ? !!timelineCollapsedIterations[iteration] : true;
    timelineCollapsedIterations[iteration] = !currentVal;
    fetchAndUpdateCleaningStatus();
}

function formatFeatureDisplay(displayValue) {
    if (displayValue == null) return '--';

    if (Array.isArray(displayValue)) {
        return displayValue
            .map(item => `${item?.name ?? ''}: ${item?.value ?? ''}`)
            .join(' | ');
    }

    if (typeof displayValue === 'object') {
        if ('name' in displayValue && ('new_value' in displayValue || 'value' in displayValue)) {
            return `${displayValue.name}: ${displayValue.new_value ?? displayValue.value ?? ''}`;
        }
        try {
            return JSON.stringify(displayValue);
        } catch (_) {
            return String(displayValue);
        }
    }

    return String(displayValue);
}

function getModifiedDisplayByAction(sample) {
    const opType = mapActionToOperation(sample.action);
    if (opType === 'Delete') return '--';

    if (opType === 'Add' || opType === 'Relabel' || opType === 'Repair') {
        return formatFeatureDisplay(sample.modified_features_display ?? '--');
    }

    return '--';
}

function updateDecisionTableWithPending(pendingData) {
    const tbody = document.getElementById('decisionTableBody');
    if (!tbody) return;

    if (!pendingData || !Array.isArray(pendingData.samples) || pendingData.samples.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-center py-4 text-gray-dark">No pending decisions (pending_decision.json).</td></tr>`;
        return;
    }

    const pendingKey = `${pendingData.iteration ?? 0}-${pendingData.step ?? 0}`;
    if (currentPendingDecisionKey !== pendingKey) {
        currentPendingDecisionKey = pendingKey;
        pendingSampleDecisions = {};
    }

    const actionName = pendingData.action || '--';

    const rows = pendingData.samples.map((sample) => {
        const sampleId = sample.sample_id ?? '--';
        const sampleIdKey = String(sampleId);
        const opType = formatTimelineOperation(mapActionToOperation(sample.action || actionName));
        const noiseScore = Number(sample.noise_score ?? 0).toFixed(6);
        const originalLabel = sample.original_label ?? '--';
        const originalDataDisplay = formatFeatureDisplay(sample.original_features_display);
        const modifiedDataDisplay = getModifiedDisplayByAction(sample);
        const modifiedLabel = sample.predicted_label ?? sample.modified_label ?? '--';

        const currentStatus = pendingSampleDecisions[sampleIdKey] || 'pending';
        const isLocked = currentStatus !== 'pending';
        const statusColor = currentStatus === 'accept' ? '#00B42A' : (currentStatus === 'reject' ? '#F53F3F' : '#FF7D00');
        const statusText = currentStatus === 'accept' ? 'Accepted' : (currentStatus === 'reject' ? 'Rejected' : 'Pending');

        return `
            <tr style="${isLocked ? 'opacity:0.8;' : ''}">
                <td>${sampleId}</td>
                <td><code>${originalDataDisplay}</code></td>
                <td>${originalLabel}</td>
                <td><span class="diff-badge" style="background: ${getColorForOperation(opType)};">${opType}</span></td>
                <td><code>${modifiedDataDisplay}</code></td>
                <td>${modifiedLabel}</td>
                <td>${noiseScore}</td>
                <td>
                    <div style="display:flex; gap:4px;">
                        <button class="action-btn btn-success btn-sm" ${isLocked ? 'disabled' : ''} onclick="setPendingSampleStatus('${sampleIdKey}','accept')">Accept</button>
                        <button class="action-btn btn-danger btn-sm" ${isLocked ? 'disabled' : ''} onclick="setPendingSampleStatus('${sampleIdKey}','reject')">Reject</button>
                    </div>
                </td>
                <td><span class="diff-badge" style="background:${statusColor};">${statusText}</span></td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = rows;
    updateSelectedCount();
}

async function sendDecisionSignal(signal, selectedSampleIds = []) {
    const response = await fetch('/api/submit-decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signal, selected_sample_ids: selectedSampleIds })
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
        throw new Error(result.error || 'Failed to send decision signal');
    }
}

function setPendingSampleStatus(sampleId, status) {
    if (!['accept', 'reject'].includes(status)) return;
    if (pendingSampleDecisions[sampleId]) return; // 已决策不可更改

    pendingSampleDecisions[sampleId] = status;
    if (pendingDecisionData) {
        updateDecisionTableWithPending(pendingDecisionData);
        autoConfirmIfAllDecided();
    }
}

async function persistPendingSelection() {
    if (!pendingDecisionData || !Array.isArray(pendingDecisionData.samples)) {
        throw new Error('No pending decision data');
    }

    const allSampleIds = pendingDecisionData.samples.map(s => String(s.sample_id));
    const undecided = allSampleIds.filter(id => !pendingSampleDecisions[id]);
    if (undecided.length > 0) {
        throw new Error(`${undecided.length} samples are still undecided`);
    }

    const acceptedIds = allSampleIds
        .filter(id => pendingSampleDecisions[id] === 'accept')
        .map(id => Number(id))
        .filter(v => Number.isFinite(v));

    const resp = await fetch('/api/update-pending-selection?dataset=adult', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ accepted_sample_ids: acceptedIds })
    });
    const result = await resp.json();
    if (!resp.ok || !result.success) {
        throw new Error(result.error || 'Failed to update pending_decision.json');
    }

    return acceptedIds;
}

async function confirmPendingDecision(showAlert = true) {
    try {
        const acceptedIds = await persistPendingSelection();
        await sendDecisionSignal('y', acceptedIds);
        if (showAlert) {
            alert(`Confirmed and sent signal y. Accepted ${acceptedIds.length}, rejected ${Object.keys(pendingSampleDecisions).length - acceptedIds.length}.`);
        }
        fetchAndUpdateCleaningStatus();
    } catch (error) {
        if (showAlert) {
            alert(`Failed to send: ${error.message}`);
        } else {
            console.error('自动确认失败:', error);
        }
    }
}

function autoConfirmIfAllDecided() {
    if (!pendingDecisionData || !Array.isArray(pendingDecisionData.samples) || pendingDecisionData.samples.length === 0) {
        return;
    }
    const undecidedCount = pendingDecisionData.samples.filter(s => !pendingSampleDecisions[String(s.sample_id)]).length;
    if (undecidedCount === 0) {
        confirmPendingDecision(false);
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
    setDataCleaningButtonState(false);
}

// 接受单个决策（示例函数）
function acceptDecision(sampleId, stepNum) {
    alert(`Accepted decision for sample ${sampleId} (step ${stepNum}).`);
    // 这里可以调用后端API确认此决策
}

// 拒绝单个决策（示例函数）
function rejectDecision(sampleId, stepNum) {
    alert(`Rejected decision for sample ${sampleId} (step ${stepNum}).`);
    // 这里可以调用后端API拒绝此决策
}

// ****************************************************************
// ****************** 以下是原有的main.js功能函数 ******************

// 更新数据质量统计信息 - 修改：从 ../results.json 读取真实数据
async function updateStatistics() {
    console.log("Fetching statistics from /api/statistics...");
    
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
        if (metricValues.length >= 7) {
            // 直接使用API返回的数据
            metricValues[0].textContent = stats.n_samples ? stats.n_samples.toLocaleString() : 'N/A';
            metricValues[1].textContent = stats.n_features || 'N/A';
            metricValues[2].textContent = stats.n_classes || 'N/A';
            metricValues[3].textContent = stats.missing_rate || 'N/A';
            metricValues[4].textContent = stats.feature_noise_ratio || 'N/A';
            metricValues[5].textContent = stats.label_noise_ratio || 'N/A';
            metricValues[6].textContent = stats.classification_accuracy || 'N/A';
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
    showEmptyChartHint('No visualization data', 'Run "Data Diagnostics" to generate charts');
}

function showEmptyChartHint(title, subTitle) {
    const chartContainer = document.getElementById('dataVisualChart');
    if (!chartContainer) return;

    const instance = dataChart || echarts.getInstanceByDom(chartContainer);
    if (instance) {
        instance.dispose();
    }
    dataChart = null;

    chartContainer.innerHTML = '';
    const noDataDiv = document.createElement('div');
    noDataDiv.className = 'w-full h-full flex flex-col items-center justify-center text-gray-500';
    noDataDiv.innerHTML = `
        <i class="fa fa-bar-chart text-4xl mb-3 opacity-30"></i>
        <p class="text-sm font-medium">${title}</p>
        <p class="text-xs mt-1">${subTitle}</p>
    `;
    chartContainer.appendChild(noDataDiv);
}

function renderClassDistributionChart(distribution) {
    if (!distribution) {
        showEmptyChartHint('No class distribution data', 'Please run Data Diagnostics first');
        return;
    }

    const xAxis = distribution.xAxis || [];
    const series = distribution.series || [];

    if (!Array.isArray(xAxis) || !Array.isArray(series) || xAxis.length === 0) {
        showEmptyChartHint('No class distribution data', 'Please run Data Diagnostics first');
        return;
    }

    const chartDom = document.getElementById('dataVisualChart');
    if (!chartDom) return;

    if (!dataChart) {
        chartDom.innerHTML = '';
        dataChart = echarts.init(chartDom);
    }

    const option = {
        tooltip: { trigger: 'axis' },
        grid: { left: 15, right: 15, top: 40, bottom: 30, containLabel: true },
        xAxis: {
            type: 'category',
            data: xAxis,
            axisLabel: { interval: 0, rotate: xAxis.length > 8 ? 30 : 0 }
        },
        yAxis: { type: 'value', name: 'Count' },
        series: [
            {
                type: 'bar',
                data: series,
                itemStyle: { color: '#165DFF' },
                barMaxWidth: 36
            }
        ]
    };

    dataChart.setOption(option, true);
}

function renderTsneChart(tsneData) {
    if (!tsneData) {
        showEmptyChartHint('No TSNE data', 'Please run Data Diagnostics first');
        return;
    }

    const points = Array.isArray(tsneData.points) ? tsneData.points : [];
    if (points.length === 0) {
        showEmptyChartHint('No TSNE data available', tsneData.message || 'Please run quality diagnostics first');
        return;
    }

    const grouped = {};
    points.forEach((p) => {
        const label = String(p.label ?? 'unknown');
        if (!grouped[label]) grouped[label] = [];
        grouped[label].push([Number(p.x), Number(p.y)]);
    });

    const chartDom = document.getElementById('dataVisualChart');
    if (!chartDom) return;

    if (!dataChart) {
        chartDom.innerHTML = '';
        dataChart = echarts.init(chartDom);
    }

    const series = Object.keys(grouped).map((label) => ({
        name: label,
        type: 'scatter',
        symbolSize: 6,
        data: grouped[label],
    }));

    const option = {
        tooltip: {
            trigger: 'item',
            formatter: (params) => {
                const val = params.value || [];
                return `${params.seriesName}<br/>x: ${Number(val[0]).toFixed(3)}<br/>y: ${Number(val[1]).toFixed(3)}`;
            }
        },
        legend: {
            type: 'scroll',
            top: 2,
            left: 'center',
            textStyle: { fontSize: 10 }
        },
        grid: { left: 25, right: 15, top: 40, bottom: 30, containLabel: true },
        xAxis: {
            type: 'value',
            name: 'TSNE-1',
            nameLocation: 'middle',
            nameGap: 28,
            scale: true
        },
        yAxis: {
            type: 'value',
            name: 'TSNE-2',
            nameLocation: 'middle',
            nameRotate: 90,
            nameGap: 32,
            scale: true
        },
        series
    };

    dataChart.setOption(option, true);
}

function renderCurrentVisualizationChart() {
    if (!currentVisualizationData) {
        showEmptyChartHint('No visualization data', 'Run "Data Diagnostics" to generate charts');
        return;
    }

    // 兼容两种结构：直接返回 payload.data 或 payload.data.frontend_viz
    const viz = currentVisualizationData.frontend_viz || currentVisualizationData;

    if (currentChartType === 'class') {
        renderClassDistributionChart(viz.class_distribution);
    } else {
        renderTsneChart(viz.tsne);
    }
}
// 切换上传类型
function switchUploadType(type) {
    currentUploadType = type;
    const buttons = document.querySelectorAll('.upload-type-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
        if ((type === 'table' && btn.textContent.includes('Tabular Data')) ||
            (type === 'archive' && btn.textContent.includes('Image Data'))) {
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
    
    const targetName = currentTargetDataset === 'test' ? 'Training Set' : 'Validation Set';
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
        
        // updateStatistics();
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
    const targetName = currentTargetDataset === 'test' ? 'Training Set' : 'Validation Set';
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
        // updateStatistics(); // 数据更新后，重新获取统计信息
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
        tab.classList.toggle('active', tab.textContent.includes(type === 'class' ? 'Class Distribution' : 'Sample Distribution'));
    });
    currentChartType = type;
    renderCurrentVisualizationChart();
}


// 初始化上传表格分页控件
function initUploadTablePagination() {
    updatePaginationButtons();
}

// 初始化决策表格分页控件
function initDecisionTablePagination() {
    updateDecisionPaginationButtons();
}

function updateDecisionPaginationButtons() {
    const pageNumbers = document.getElementById('decisionPageNumbers');
    if (!pageNumbers) return;

    const totalPages = decisionPageConfig.totalPages;
    const currentPage = decisionPageConfig.currentPage;
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
            button.onclick = () => changeDecisionPage(page);
        }
        pageNumbers.appendChild(button);
    });

    const prevBtn = document.getElementById('decisionPrevBtn');
    const nextBtn = document.getElementById('decisionNextBtn');
    const pageInput = document.getElementById('decisionPageInput');
    if (prevBtn) prevBtn.disabled = currentPage === 1;
    if (nextBtn) nextBtn.disabled = currentPage === totalPages;
    if (pageInput) pageInput.value = currentPage;
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
    // 保持当前决策表格内容，不做覆盖，避免出现空白
    if (pendingDecisionData && Array.isArray(pendingDecisionData.samples) && pendingDecisionData.samples.length > 0) {
        updateDecisionTableWithPending(pendingDecisionData);
    } else {
        renderEmptyPendingDecisionTable('No pending decisions.');
    }
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
                message = 'No cleaned result data yet. Please run Data Governance first.';
                icon = 'fa-filter';
                break;
            case 'pool':
                message = 'Augmentation Pool is empty. Please run "Data Diagnostics" first.';
                icon = 'fa-cube';
                break;
            default:
                message = 'No data available. Please click the "Data Upload" button to upload your data.';
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
                         alt="Preview" 
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
            console.log(`Result dataset loaded successfully, ${resultSetData.length} records in total.`);
            
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
                <p style="margin-top: 10px;">Loading data...</p>
            </div>
        `;
    }
    
    // 使用requestAnimationFrame让UI有机会更新
    requestAnimationFrame(() => {
        currentDisplayDataset = dataset;
        
        // 更新按钮激活状态
        const buttons = document.querySelectorAll('.upload-card .tab-btn');
        buttons.forEach(btn => {
            if ((btn.textContent === 'Training Set' && dataset === 'test') ||
                (btn.textContent === 'Validation Set' && dataset === 'val') ||
                (btn.textContent === 'Augmentation Pool' && dataset === 'pool') ||
                (btn.textContent === 'Curated Data' && dataset === 'result')) {
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
        console.error(`Error: upload zone element not found #${prefix}UploadZone`);
        return;
    }
    zone.classList.remove('upload-success', 'upload-loading', 'upload-error', 'border-primary', 'bg-blue-100');
    if (status === 'selected') {
        if (initialDiv) initialDiv.classList.add('hidden');
        if (infoDiv) infoDiv.classList.remove('hidden');
        if (nameSpan) nameSpan.textContent = file.name;
        if (sizeSpan) sizeSpan.textContent = formatFileSize(file.size);
        if (statusSpan) {
            statusSpan.textContent = 'Selected. Click the button below to begin uploading.';
            statusSpan.className = 'text-xs text-gray-dark';
        }
        if (progressContainer) progressContainer.classList.add('hidden');
        if (progressBar) progressBar.style.width = '0%';
    } else if (status === 'uploading') {
        zone.classList.add('upload-loading');
        if (statusSpan) {
            statusSpan.textContent = 'Uploading...';
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
        console.error('Failed to load image:', imgUrl);
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

// 自动执行决策切换逻辑（按钮模式：点击后仅生效，不切换样式/文案）
async function toggleAutoProcess() {
    if (isAutoProcess) {
        return;
    }

    isAutoProcess = true;

    try {
        await sendDecisionSignal('a');
        if (pendingDecisionData && Array.isArray(pendingDecisionData.samples)) {
            pendingDecisionData.samples.forEach(s => {
                const id = String(s.sample_id);
                if (!pendingSampleDecisions[id]) pendingSampleDecisions[id] = 'accept';
            });
            updateDecisionTableWithPending(pendingDecisionData);
            autoConfirmIfAllDecided();
        }
        alert('Auto decision mode is enabled, and signal a has been sent.');
    } catch (error) {
        isAutoProcess = false;
        alert(`Failed to send auto-execute signal: ${error.message}`);
    }
}

// 更新选中数量
function updateSelectedCount() {
    const selectedEl = document.getElementById('selectedCount');
    const totalEl = document.getElementById('totalCount');
    const pendingEl = document.getElementById('pendingCount');

    if (pendingDecisionData && Array.isArray(pendingDecisionData.samples)) {
        const totalCount = pendingDecisionData.samples.length;
        const pendingCount = pendingDecisionData.samples.filter(s => !pendingSampleDecisions[String(s.sample_id)]).length;
        const decidedCount = totalCount - pendingCount;

        if (selectedEl) selectedEl.textContent = decidedCount;
        if (totalEl) totalEl.textContent = totalCount;
        if (pendingEl) pendingEl.textContent = pendingCount;
    } else {
        if (selectedEl) selectedEl.textContent = 0;
        if (totalEl) totalEl.textContent = 0;
        if (pendingEl) pendingEl.textContent = 0;
    }
}

// 批量处理：处理当前表格剩余未决策样本
function batchProcess(action) {
    if (!['accept', 'reject'].includes(action)) return;
    if (!pendingDecisionData || !Array.isArray(pendingDecisionData.samples)) {
        alert('There are no pending samples available for batch processing.');
        return;
    }

    const remainingIds = pendingDecisionData.samples
        .map(s => String(s.sample_id))
        .filter(id => !pendingSampleDecisions[id]);

    if (remainingIds.length === 0) {
        alert('There are no remaining undecided samples in the current table.');
        return;
    }

    remainingIds.forEach(id => {
        pendingSampleDecisions[id] = action;
    });

    updateDecisionTableWithPending(pendingDecisionData);
    autoConfirmIfAllDecided();
    alert(`Batch ${action === 'accept' ? 'accepted' : 'rejected'} the remaining ${remainingIds.length} samples.`);
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
        if (currentTargetDataset === 'test' && btn.textContent.includes('Training Set')) {
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
        
        console.log(`Starting download of best cleaned result for dataset ${datasetName}...`);
        
        // 1. 先检查文件是否存在
        const checkResponse = await fetch(`/api/check-best-data?dataset=${datasetName}`);
        const checkResult = await checkResponse.json();
        
        console.log("文件检查结果:", checkResult);
        
        if (!checkResult.exists) {
            alert('Best data file not found. Please run "Data Governance" first to generate cleaned results.');
            return;
        }
        
        // 2. 显示确认对话框
        if (!confirm(`Are you sure you want to download the best cleaned result for dataset ${datasetName}?\nFile size: ${checkResult.file_info?.file_size_kb || 'Unknown'} KB`)) {
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
            alert('Download has started. Please check your browser download list.\nIf the download does not start automatically, please check your browser settings.');
        }, 500);
        
    } catch (error) {
        console.error("下载失败:", error);
        alert(`Download failed: ${error.message}\nPlease check the console for more details.`);
    }
}

// 运行质量检测并加载结果
// 质量检测功能
async function runQualityCheckAndLoad(triggerButton = null) {
    const button = triggerButton;
    if (!button) {
        alert('Trigger button not found. Unable to start diagnostics.');
        return;
    }
    const originalText = button.innerHTML;
    const originalClass = button.className;
    
    try {
        // 1. 按钮状态反馈：禁用并显示加载中
        button.disabled = true;
        button.className = originalClass.replace('btn-warning', 'btn-primary') + ' opacity-70 cursor-not-allowed';
        button.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Running...';
        
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
            
            alert(`Quality diagnostics failed: server returned an error page\nPlease check Flask console logs`);
            return;
        }
        
        const result = await response.json();
        
        // 恢复按钮状态
        button.disabled = false;
        button.className = originalClass;
        button.innerHTML = originalText;
        
        if (!response.ok) {
            throw new Error(result.error || 'Quality diagnostics failed');
        }
        
        console.log("质量检测结果:", result);
        
        if (result.success) {
            // 更新统计信息
            if (result.quality_stats) {
                updateStatisticsFromResult(result.quality_stats);
            }
            
            
            // 处理可视化结果（前端组件直接渲染）
            if (result.visualization && result.visualization.success && result.visualization.data) {
                currentVisualizationData = result.visualization.data;
                renderCurrentVisualizationChart();
            } else if (result.visualization && !result.visualization.success) {
                currentVisualizationData = null;
                showEmptyChartHint('Failed to generate visualization data', result.visualization.message || 'Please check backend logs');
                console.warn("Visualization generation failed:", result.visualization.message);
            } else {
                currentVisualizationData = null;
                showEmptyChartHint('No visualization data', 'Please run Data Diagnostics first');
            }
            
        } else {
            alert('Quality diagnostics failed: ' + (result.error || 'Unknown error'));
        }
        
    } catch (error) {
        console.error('质量检测失败:', error);
        alert(`Quality diagnostics failed: ${error.message}`);
        
        // 恢复按钮状态
        if (button) {
            button.disabled = false;
            button.className = originalClass;
            button.innerHTML = originalText;
        }
    }
}

function updateStatisticsFromResult(stats) {
    const metricValues = document.querySelectorAll('.metric-value');
    if (metricValues.length >= 7) {
        metricValues[0].textContent = (stats.n_samples ?? 'N/A') === 'N/A' ? 'N/A' : Number(stats.n_samples).toLocaleString();
        metricValues[1].textContent = stats.n_features ?? 'N/A';
        metricValues[2].textContent = stats.n_classes ?? 'N/A';
        metricValues[3].textContent = stats.missing_rate ?? 'N/A';
        metricValues[4].textContent = stats.feature_noise_ratio ?? 'N/A';
        metricValues[5].textContent = stats.label_noise_ratio ?? 'N/A';
        metricValues[6].textContent = stats.classification_accuracy ?? 'N/A';
    }
}

// 组合功能：质量检测 + 数据增强
async function runQualityCheckAndAugmentation(triggerButton) {
    await runQualityCheckAndLoad(triggerButton);
    await runDataAugmentation(triggerButton);

}

// 数据增强功能
async function runDataAugmentation(triggerButton = null) {
    const button = triggerButton;
    const originalText = button ? button.innerHTML : '';
    
    try {
        // 按钮状态反馈
        if (button) {
            button.disabled = true;
            button.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Running...';
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
            alert(`Quality diagnostics completed! ${result.generated_count} samples were generated and added to the Augmentation Pool.`);
            
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
            // updateStatistics();
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
                console.log(`Loaded ${result.count} candidate pool records`);
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

