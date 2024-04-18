import numpy as np
import matplotlib.pyplot as plt
import imageio

# Открываем изображение
image_architecture = imageio.imread('7.jpg', pilmode='L')

# Преобразуем изображение в одномерный массив
image_flat = image_architecture.ravel()

# Инициализируем параметры модели
n_components = 3
n_iterations = 100  # Количество итераций алгоритма EM

# Инициализируем параметры смеси гауссовых распределений
mu = np.random.randint(0, 255, size=n_components)  # Средние значения для каждой компоненты
sigma = np.random.rand(n_components) * 100  # Стандартные отклонения для каждой компоненты
pi = np.ones(n_components) / n_components  # Веса компонент (в начале равномерные)

# Разделение изображения на компоненты
image_components = np.array_split(image_flat, n_components)

# EM алгоритм
for _ in range(n_iterations):
    # Expectation step
    # Вычисляем апостериорные вероятности принадлежности к каждой компоненте для каждого пикселя
    probs = np.zeros((len(image_flat), n_components))
    for i, component in enumerate(image_components):
        probs[:, i] = pi[i] * np.exp(-0.5 * ((image_flat - mu[i]) / sigma[i])**2) / (np.sqrt(2 * np.pi) * sigma[i])

    probs /= probs.sum(axis=1)[:, np.newaxis]

    # Maximization step
    # Обновляем параметры модели
    for i in range(n_components):
        # Проверяем наличие NaN и деления на ноль
        if np.isnan(sigma[i]) or sigma[i] == 0:
            continue
        mu[i] = np.sum(probs[:, i] * image_flat) / np.sum(probs[:, i])
        sigma[i] = np.sqrt(np.sum(probs[:, i] * (image_flat - mu[i])**2) / np.sum(probs[:, i]))
        pi[i] = np.mean(probs[:, i])

# Находим индекс компоненты с максимальной оценкой апостериорной вероятности для каждого пикселя
max_prob_idx = np.argmax(probs, axis=1)

# Построим сегментированное изображение, используя компоненты смеси с наибольшей вероятностью
segmented_image = np.zeros_like(image_flat)
for i in range(n_components):
    segmented_image[max_prob_idx == i] = mu[i]

# Визуализируем исходное изображение и сегментированное изображение
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_architecture, cmap='gray')
plt.title('Исходное изображение')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image.reshape(image_architecture.shape), cmap='gray')
plt.title('Сегментированное изображение')

plt.show()
