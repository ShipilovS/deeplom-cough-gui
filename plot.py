import pandas as pd
import matplotlib.pyplot as plt

# Данные для сравнения моделей
data = {
    'Model': ['Сверточная сеть без обработки', 'Сверточная сеть с MFCC'],
    'Training Accuracy': [0.8412, 0.8695],
    'Validation Accuracy': [0.8297, 0.8760]
}

df = pd.DataFrame(data)

# Создаем фигуру с улучшенным дизайном
plt.figure(figsize=(10, 6), facecolor='#f5f5f5')
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

# Современная цветовая палитра
colors = {
    'Training': '#4C72B0',  # Приятный синий
    'Validation': '#DD8452'  # Теплый оранжевый
}

# Параметры диаграммы
bar_width = 0.35
index = range(len(df))

# Создаем столбцы
bars_train = plt.bar(index, df['Training Accuracy'], bar_width, 
                    label='Точность обучения', color=colors['Training'],
                    edgecolor='white', linewidth=1, alpha=0.9)

bars_val = plt.bar([i + bar_width for i in index], df['Validation Accuracy'], bar_width,
                   label='Точность валидации', color=colors['Validation'],
                   edgecolor='white', linewidth=1, alpha=0.9)

# Настройка осей и заголовка
plt.xlabel('Модель', fontsize=12, labelpad=10)
plt.ylabel('Точность', fontsize=12, labelpad=10)
plt.title('Сравнение моделей\n', fontsize=14, fontweight='bold', pad=20)
plt.xticks([i + bar_width/2 for i in index], df['Model'], fontsize=11)
plt.ylim(0.7, 0.9)
plt.yticks(fontsize=10)

# Добавляем значения на столбцы
for bar in bars_train + bars_val:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 0.02,
            f'{height:.3f}',
            ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

# Легенда и сетка
plt.legend(frameon=True, facecolor='white', framealpha=0.8, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Убираем лишние линии рамки
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('fig.png')
plt.show()