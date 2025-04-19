import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Данные для сравнения моделей
data = {
    'Model': [
        'Сверточная сеть без обработки', 
        'Сверточная сеть с MFCC=13',
        'Сверточная сеть с MFCC=25',
        'Сверточная сеть с MFCC=30',
        'Сверточная сеть с MFCC=40'
    ],
    'Training Accuracy': [0.8412, 0.8695, 0.8947, 0.9173, 0.9117],
    'Validation Accuracy': [0.8297, 0.8760, 0.9141, 0.9377, 0.9343],
    'Training Loss': [None, None, 0.4412, 0.3894, 0.4476],
    'Validation Loss': [None, None, 0.3977, 0.3229, 0.3849]
}

df = pd.DataFrame(data)

# Создаем фигуру с улучшенным дизайном
plt.figure(figsize=(13, 7), facecolor='#f5f5f5')
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

# Современная цветовая палитра
colors = {
    'Training': '#4C72B0',  # Приятный синий
    'Validation': '#DD8452'  # Теплый оранжевый
}

# Параметры диаграммы
bar_width = 0.35
index = np.arange(len(df))

# Создаем столбцы
bars_train = plt.bar(index, df['Training Accuracy'], bar_width, 
                    label='Точность обучения', color=colors['Training'],
                    edgecolor='white', linewidth=1, alpha=0.9)

bars_val = plt.bar(index + bar_width, df['Validation Accuracy'], bar_width,
                   label='Точность валидации', color=colors['Validation'],
                   edgecolor='white', linewidth=1, alpha=0.9)

# Настройка осей и заголовка
plt.ylabel('Точность', fontsize=12, labelpad=10)
plt.xticks(index + bar_width/2, df['Model'], rotation=15, ha='right', fontsize=10)
plt.ylim(0.7, 1.0)
plt.yticks(fontsize=10)

# Добавляем значения на столбцы
for bar in bars_train:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 0.02,
            f'{height:.3f}',
            ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

for bar in bars_val:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 0.02,
            f'{height:.3f}',
            ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')


# Легенда и сетка
plt.legend(frameon=True, facecolor='white', framealpha=0.8, fontsize=10, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Убираем лишние линии рамки
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()