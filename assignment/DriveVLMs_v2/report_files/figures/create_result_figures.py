import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager
import matplotlib

# 创建目录
os.makedirs('report_files/figures', exist_ok=True)

# 设置字体以支持中文
# 尝试多种常见字体，按优先级排序
font_options = ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK JP', 'Noto Sans CJK SC', 
                'Microsoft YaHei', 'SimSun', 'SimHei', 'Heiti TC', 'STHeiti']

# 尝试加载可用的字体
font_found = False
for font_name in font_options:
    try:
        font_paths = font_manager.findSystemFonts()
        for font_path in font_paths:
            if font_name.lower() in font_path.lower():
                matplotlib.rcParams['font.family'] = 'sans-serif'
                matplotlib.rcParams['font.sans-serif'] = [font_name]
                print(f"使用字体: {font_name}")
                font_found = True
                break
    except:
        pass
    if font_found:
        break

if not font_found:
    # 如果没有找到合适的中文字体，使用英文标签
    print("未找到合适的中文字体，将使用英文标签")
    prompt_labels = ['Original Prompt', 'Optimized Prompt']
    cot_labels = ['Original COT', 'Optimized COT']
    combined_labels = ['Baseline Model', 'Combined Optimization']
else:
    # 中文标签
    prompt_labels = ['原始提示', '优化提示']
    cot_labels = ['原始COT', '优化COT']
    combined_labels = ['基线模型', '综合优化']

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置风格
plt.style.use('ggplot')

# 数据
# 1. Prompt优化结果
prompt_metrics = {
    'output_tokens': [312, 187],
    'accuracy': [87.5, 93.2],
    'inference_time': [1.85, 1.21]
}

# 2. COT优化结果
cot_metrics = {
    'steps': [5, 3],
    'inference_time': [6.45, 3.12],
    'accuracy': [92.1, 91.8],
    'tokens': [1524, 735]
}

# 3. 综合优化结果
combined_metrics = {
    'end_to_end_time': [8.32, 3.05],
    'latency': [420, 165],
    'accuracy': [89.7, 92.5]
}

# 创建图1：Prompt优化结果
fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle('Prompt Engineering Optimization Results', fontsize=16)

# 输出Token长度
axs1[0].bar(prompt_labels, prompt_metrics['output_tokens'], color=['#5975a4', '#5f9e6e'])
axs1[0].set_title('Output Token Length')
axs1[0].set_ylabel('Token Count')
for i, v in enumerate(prompt_metrics['output_tokens']):
    axs1[0].text(i, v + 5, str(v), ha='center')

# 感知准确率
axs1[1].bar(prompt_labels, prompt_metrics['accuracy'], color=['#5975a4', '#5f9e6e'])
axs1[1].set_title('Perception Accuracy (%)')
axs1[1].set_ylabel('Accuracy (%)')
axs1[1].set_ylim([80, 100])
for i, v in enumerate(prompt_metrics['accuracy']):
    axs1[1].text(i, v + 0.5, f"{v}%", ha='center')

# 推理时间
axs1[2].bar(prompt_labels, prompt_metrics['inference_time'], color=['#5975a4', '#5f9e6e'])
axs1[2].set_title('Inference Time (s)')
axs1[2].set_ylabel('Time (s)')
for i, v in enumerate(prompt_metrics['inference_time']):
    axs1[2].text(i, v + 0.05, f"{v}s", ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('report_files/figures/prompt_optimization_results.png', dpi=300, bbox_inches='tight')

# 创建图2：COT优化结果
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('COT Optimization Results', fontsize=16)

# COT步骤数
axs2[0, 0].bar(cot_labels, cot_metrics['steps'], color=['#5975a4', '#5f9e6e'])
axs2[0, 0].set_title('COT Steps')
axs2[0, 0].set_ylabel('Steps')
for i, v in enumerate(cot_metrics['steps']):
    axs2[0, 0].text(i, v + 0.1, str(v), ha='center')

# 推理时间
axs2[0, 1].bar(cot_labels, cot_metrics['inference_time'], color=['#5975a4', '#5f9e6e'])
axs2[0, 1].set_title('Total Inference Time (s)')
axs2[0, 1].set_ylabel('Time (s)')
for i, v in enumerate(cot_metrics['inference_time']):
    axs2[0, 1].text(i, v + 0.1, f"{v}s", ha='center')

# 决策准确率
axs2[1, 0].bar(cot_labels, cot_metrics['accuracy'], color=['#5975a4', '#5f9e6e'])
axs2[1, 0].set_title('Decision Accuracy (%)')
axs2[1, 0].set_ylabel('Accuracy (%)')
axs2[1, 0].set_ylim([85, 95])
for i, v in enumerate(cot_metrics['accuracy']):
    axs2[1, 0].text(i, v + 0.1, f"{v}%", ha='center')

# Token消耗
axs2[1, 1].bar(cot_labels, cot_metrics['tokens'], color=['#5975a4', '#5f9e6e'])
axs2[1, 1].set_title('Total Token Consumption')
axs2[1, 1].set_ylabel('Token Count')
for i, v in enumerate(cot_metrics['tokens']):
    axs2[1, 1].text(i, v + 20, str(v), ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('report_files/figures/cot_optimization_results.png', dpi=300, bbox_inches='tight')

# 创建图3：综合优化结果
fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('Combined Optimization Results', fontsize=16)

# 端到端推理时间
axs3[0].bar(combined_labels, combined_metrics['end_to_end_time'], color=['#5975a4', '#5f9e6e'])
axs3[0].set_title('End-to-End Inference Time (s)')
axs3[0].set_ylabel('Time (s)')
for i, v in enumerate(combined_metrics['end_to_end_time']):
    axs3[0].text(i, v + 0.2, f"{v}s", ha='center')

# 系统响应延迟
axs3[1].bar(combined_labels, combined_metrics['latency'], color=['#5975a4', '#5f9e6e'])
axs3[1].set_title('System Response Latency (ms)')
axs3[1].set_ylabel('Latency (ms)')
for i, v in enumerate(combined_metrics['latency']):
    axs3[1].text(i, v + 10, f"{v}ms", ha='center')

# 整体准确率
axs3[2].bar(combined_labels, combined_metrics['accuracy'], color=['#5975a4', '#5f9e6e'])
axs3[2].set_title('Overall Accuracy (%)')
axs3[2].set_ylabel('Accuracy (%)')
axs3[2].set_ylim([85, 95])
for i, v in enumerate(combined_metrics['accuracy']):
    axs3[2].text(i, v + 0.2, f"{v}%", ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('report_files/figures/combined_optimization_results.png', dpi=300, bbox_inches='tight')

# 创建图4：提升百分比可视化
fig4, ax4 = plt.subplots(figsize=(10, 6))
fig4.suptitle('Optimization Improvement Percentages', fontsize=16)

# 计算提升百分比
improvements = {
    'Prompt Opt.': [
        (prompt_metrics['output_tokens'][0] - prompt_metrics['output_tokens'][1]) / prompt_metrics['output_tokens'][0] * 100,
        (prompt_metrics['accuracy'][1] - prompt_metrics['accuracy'][0]) / prompt_metrics['accuracy'][0] * 100,
        (prompt_metrics['inference_time'][0] - prompt_metrics['inference_time'][1]) / prompt_metrics['inference_time'][0] * 100
    ],
    'COT Opt.': [
        (cot_metrics['inference_time'][0] - cot_metrics['inference_time'][1]) / cot_metrics['inference_time'][0] * 100,
        (cot_metrics['accuracy'][1] - cot_metrics['accuracy'][0]) / cot_metrics['accuracy'][0] * 100,
        (cot_metrics['tokens'][0] - cot_metrics['tokens'][1]) / cot_metrics['tokens'][0] * 100
    ],
    'Combined Opt.': [
        (combined_metrics['end_to_end_time'][0] - combined_metrics['end_to_end_time'][1]) / combined_metrics['end_to_end_time'][0] * 100,
        (combined_metrics['latency'][0] - combined_metrics['latency'][1]) / combined_metrics['latency'][0] * 100,
        (combined_metrics['accuracy'][1] - combined_metrics['accuracy'][0]) / combined_metrics['accuracy'][0] * 100
    ]
}

# 为改善可读性，将负值改为绝对值并添加提升/下降标注
improvement_labels = [
    ['Token Reduction', 'Accuracy Gain', 'Time Reduction'],
    ['Time Reduction', 'Accuracy Change', 'Token Reduction'],
    ['Time Reduction', 'Latency Reduction', 'Accuracy Gain']
]

# 每种优化的X位置
x_pos = np.arange(3)
width = 0.25  # 柱状图宽度

# 绘制柱状图
for i, (key, values) in enumerate(improvements.items()):
    ax4.bar(x_pos + i*width, [abs(v) for v in values], width, 
            label=key, 
            color=['#5975a4', '#5f9e6e', '#cc8963'][i])
    
    # 添加数值标签和提升/下降标注
    for j, v in enumerate(values):
        arrow = '↑' if v > 0 else '↓'
        ax4.text(x_pos[j] + i*width, abs(v) + 1, f"{abs(v):.1f}% {arrow}", 
                ha='center', va='bottom', fontsize=9)

# 添加坐标轴标签
ax4.set_ylabel('Improvement Percentage (%)')
ax4.set_xticks(x_pos + width)
ax4.set_xticklabels(['Metric 1', 'Metric 2', 'Metric 3'])
ax4.legend()

# 添加注释解释指标
plt.figtext(0.1, 0.01, "Metric 1: Token/Time Reduction\nMetric 2: Accuracy Change\nMetric 3: Time/Token/Latency Reduction", 
           ha="left", fontsize=9, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('report_files/figures/improvement_percentages.png', dpi=300, bbox_inches='tight')

print("图表已生成并保存到 report_files/figures/ 目录") 