import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. ЗАГРУЗКА ДАННЫХ
df = pd.read_csv('NVDA_yfinance_clean.csv', delimiter=';')
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

print("=" * 50)
print("АНАЛИЗ NVDA")
print("=" * 50)
print(f"Период: {df.index.min().date()} - {df.index.max().date()}")
print(f"Дней: {len(df)}")
print(f"Пропуски: {df.isnull().sum().sum()}")
print()

# 2. ГРАФИКИ ВРЕМЕННЫХ РЯДОВ
fig, axes = plt.subplots(5, 1, figsize=(12, 10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
    axes[i].plot(df[col], linewidth=1, color='blue')
    axes[i].set_ylabel(col, fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=df[col].mean(), color='red', linestyle='--', alpha=0.7, label=f'Среднее: {df[col].mean():.2f}')
    axes[i].legend(fontsize=8)
plt.suptitle('Акции NVDA - Временные ряды', fontsize=14)
plt.tight_layout()
plt.savefig('nvda_series.png', dpi=150)
plt.show()

# 3. СТАТИСТИКА + ГРАФИКИ РАСПРЕДЕЛЕНИЯ
print("=" * 50)
print("СТАТИСТИКА:")
print(df.describe().round(2))
print()

# Графики распределения
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
    axes[i].hist(df[col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {df[col].mean():.2f}')
    axes[i].axvline(df[col].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Медиана: {df[col].median():.2f}')
    axes[i].set_title(f'Распределение {col}', fontsize=10)
    axes[i].set_xlabel('Значение')
    axes[i].set_ylabel('Частота')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
# Убираем пустой подграфик
axes[5].axis('off')
plt.suptitle('Гистограммы распределения цен и объема', fontsize=14)
plt.tight_layout()
plt.savefig('nvda_distributions.png', dpi=150)
plt.show()

# 4. ВЫБРОСЫ (3 сигмы) + ГРАФИКИ
print("=" * 50)
print("ВЫБРОСЫ (правило 3 сигм):")
outlier_data = {}
for col in df.columns:
    mean, std = df[col].mean(), df[col].std()
    lower = mean - 3 * std
    upper = mean + 3 * std
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_data[col] = outliers
    print(f"{col}: {len(outliers)} выбросов ({len(outliers) / len(df) * 100:.1f}%)")

# Графики выбросов
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
    mean, std = df[col].mean(), df[col].std()
    lower = mean - 3 * std
    upper = mean + 3 * std

    # Точечный график
    axes[i].scatter(range(len(df)), df[col], alpha=0.5, s=1, c='blue', label='Нормальные')

    # Выделяем выбросы
    outliers = outlier_data[col]
    if len(outliers) > 0:
        outlier_indices = [df.index.get_loc(idx) for idx in outliers.index]
        axes[i].scatter(outlier_indices, outliers[col], color='red', s=20, label=f'Выбросы ({len(outliers)})')

    axes[i].axhline(y=mean, color='green', linestyle='-', linewidth=1.5, label=f'Среднее: {mean:.2f}')
    axes[i].axhline(y=lower, color='orange', linestyle='--', linewidth=1, label=f'-3σ: {lower:.2f}')
    axes[i].axhline(y=upper, color='orange', linestyle='--', linewidth=1, label=f'+3σ: {upper:.2f}')
    axes[i].set_title(f'{col} - Выбросы', fontsize=10)
    axes[i].set_xlabel('Индекс')
    axes[i].set_ylabel('Значение')
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)
axes[5].axis('off')
plt.suptitle('Обнаружение выбросов (правило 3 сигм)', fontsize=14)
plt.tight_layout()
plt.savefig('nvda_outliers.png', dpi=150)
plt.show()

# 5. ДИАПАЗОНЫ
print("=" * 50)
print("ДИАПАЗОНЫ ЗНАЧЕНИЙ:")
print(f"Средний дневной размах (High-Low): ${(df['High'] - df['Low']).mean():.2f}")
print(f"Макс. дневной размах: ${(df['High'] - df['Low']).max():.2f}")
print(f"Среднее изменение Close: {df['Close'].pct_change().mean() * 100:.2f}%")

plt.figure(figsize=(10, 5))
df[['Open', 'High', 'Low', 'Close']].boxplot()
plt.title('Диапазоны цен (Boxplot)', fontsize=12)
plt.ylabel('Цена ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nvda_boxplots.png', dpi=150)
plt.show()

# 6. КОРРЕЛЯЦИЯ
print("=" * 50)
print("КОРРЕЛЯЦИЯ С CLOSE:")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица')
plt.tight_layout()
plt.savefig('nvda_corr.png')
plt.show()
print(df.corr()['Close'].round(3))

# 7. ШУМ (SNR)
print("=" * 50)
print("АНАЛИЗ ШУМА:")

decomp = seasonal_decompose(df['Close'], model='additive', period=252)
signal = decomp.trend + decomp.seasonal
resid = decomp.resid
valid = ~(signal.isna() | resid.isna())
snr = 10 * np.log10(signal[valid].var() / resid[valid].var())

print(f"SNR = {snr:.1f} дБ ({['Плохое','Удовл.','Хорошее','Отличное'][min(3, max(0, int(snr/10)))]})")

plt.figure(figsize=(12, 8))
plt.subplot(411); plt.plot(df.index, df['Close']); plt.title('Исходный')
plt.subplot(412); plt.plot(df.index, decomp.trend); plt.title('Тренд')
plt.subplot(413); plt.plot(df.index, decomp.seasonal); plt.title('Сезонность')
plt.subplot(414); plt.plot(df.index, decomp.resid); plt.title('Шум')
plt.tight_layout()
plt.savefig('nvda_decomp.png')
plt.show()

print("\n" + "=" * 50)
print("ВСЕ ГРАФИКИ СОХРАНЕНЫ:")
print("1. nvda_series.png - временные ряды")
print("2. nvda_distributions.png - гистограммы распределения")
print("3. nvda_outliers.png - графики выбросов")
print("4. nvda_boxplots.png - boxplot диапазонов")
print("5. nvda_corr.png - корреляционная матрица")
print("6. nvda_decomp.png - декомпозиция ряда")
print("=" * 50)
print("ГОТОВО!")