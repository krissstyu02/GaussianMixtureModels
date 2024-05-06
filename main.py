import numpy as np
from sklearn.mixture import GaussianMixture
import tkinter as tk
from PIL import ImageTk, Image
import imageio
import threading
from tkinter import filedialog, ttk
import cv2
import time

selected_image = None  # Исходное изображение
segmented_image1 = None  # Сегментированное изображение
segmented_image2 = None  # Сегментированное изображение


# Функция для загрузки изображения с компьютера
def load_image():
    global selected_image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Загружаем изображение и изменяем цветовую модель на RGB
        selected_image = imageio.imread(file_path, pilmode='RGB')
        return selected_image
    return None


# Функция для сегментации изображения
def segment_image():
    # Получаем выбранное изображение
    image = selected_image
    if image is not None:
        start_time_1 = time.time()
        # Получаем значения входных данных
        covariance_type = covariance_type_var.get()  # Тип ковариационной матрицы
        n_components = int(n_components_slider.get())  # Количество компонент смеси GMM
        max_iter = int(max_iter_entry.get())  # Максимальное количество итераций
        n_init = int(n_init_entry.get())  # Количество инициализаций
        methods = methods_var.get()  # Выбранный метод сегментации


        # Преобразуем изображение в одномерный массив
        image_flat = image.reshape(-1, 3)

        # Подготовим данные для обучения смешанной гауссовой модели
        X = image_flat

        # Инициализируем смешанную гауссову модель
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter,
                              n_init=n_init)

        # Обучим модель на данных
        gmm.fit(X)

        # Посчитаем оценки апостериорной вероятности для каждого пикселя
        probs = gmm.predict_proba(X)

        # Найдем индекс компоненты с максимальной оценкой апостериорной вероятности для каждого пикселя
        max_prob_idx = np.argmax(probs, axis=1)

        # Построим сегментированное изображение, используя компоненты смеси с наибольшей вероятностью
        segmented_image = np.zeros_like(image_flat)
        for i in range(n_components):
            segmented_image[max_prob_idx == i] = gmm.means_[i]

        segmented_image = segmented_image.reshape(image.shape)

        global segmented_image1
        segmented_image1 = segmented_image

        segmented_img = Image.fromarray(segmented_image).resize((300, 300))
        segmented_img_tk = ImageTk.PhotoImage(segmented_img)

        segmented_text = tk.Label(root, text="Сегментированное изображение")
        segmented_text.grid(row=2, column=1)
        segmented_label.config(image=segmented_img_tk)
        # Обновляем отображение
        segmented_label.image = segmented_img_tk

        end_time_1 = time.time()  # Запоминаем время окончания сегментации
        segmentation_time_1 = end_time_1 - start_time_1  # Вычисляем время сегментации
        time_label_1.config(text=f"Время сегментации: {segmentation_time_1:.2f} сек")  # Обновляем метку с временем

    global segmented_image2
    if (methods == 'k-means'):
        start_time_2 = time.time()
        data = image.reshape((-1, 3)).astype(np.float32)

        # Задание критериев останова
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)

        # Применение k-means
        _, labels, centers = cv2.kmeans(data, n_components, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Преобразование центров к целочисленному типу данных
        centers = np.uint8(centers)

        # Получение сегментированного изображения
        segmented_image = centers[labels.flatten()]

        # Возврат сегментированного изображения в формате, соответствующем исходному
        segmented_image = segmented_image.reshape(image.shape)

        segmented_image2 = segmented_image

        segmented_img = Image.fromarray(segmented_image).resize((300, 300))
        segmented_img_tk = ImageTk.PhotoImage(segmented_img)

        segmented_text = tk.Label(root, text="Сегментированное изображение2")
        segmented_text.grid(row=2, column=2)
        k_means_label.config(image=segmented_img_tk)
        # Обновляем отображение
        k_means_label.image = segmented_img_tk

        end_time_2 = time.time()  # Запоминаем время окончания сегментации
        segmentation_time_2 = end_time_2 - start_time_2  # Вычисляем время сегментации
        time_label_2.config(text=f"Время сегментации: {segmentation_time_2:.2f} сек")

    elif (methods == 'Метод на основе порога'):
        start_time_2 = time.time()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        threshold_value = 128
        # Применение пороговой обработки
        _, segmented_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        segmented_img = Image.fromarray(segmented_image).resize((300, 300))
        segmented_img_tk = ImageTk.PhotoImage(segmented_img)

        segmented_image2 = segmented_image

        # segmented_text = tk.Label(root, text="Сегментированное изображение2")
        # segmented_text.grid(row=2, column=2)
        k_means_label.config(image=segmented_img_tk)
        # Обновляем отображение
        k_means_label.image = segmented_img_tk

        end_time_2 = time.time()  # Запоминаем время окончания сегментации
        segmentation_time_2 = end_time_2 - start_time_2  # Вычисляем время сегментации
        time_label_2.config(text=f"Время сегментации: {segmentation_time_2:.2f} сек")


    elif (methods == 'Сегментация краев'):
        start_time_2 = time.time()
        low_threshold = 50  # Пример нижнего порога (можно настроить)
        high_threshold = 150  # Пример верхнего порога (можно настроить)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применение оператора Кэнни
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)

        segmented_image2 = edges

        segmented_img = Image.fromarray(edges).resize((300, 300))
        segmented_img_tk = ImageTk.PhotoImage(segmented_img)

        # segmented_text = tk.Label(root, text="Сегментированное изображение2")
        # segmented_text.grid(row=2, column=2)
        k_means_label.config(image=segmented_img_tk)
        # Обновляем отображение
        k_means_label.image = segmented_img_tk

        end_time_2 = time.time()  # Запоминаем время окончания сегментации
        segmentation_time_2 = end_time_2 - start_time_2  # Вычисляем время сегментации
        time_label_2.config(text=f"Время сегментации: {segmentation_time_2:.2f} сек")


def show_image():
    # Загружаем изображение
    image = load_image()
    if image is not None:
        # Отображаем изображения в окне tkinter
        original_img = Image.fromarray(image).resize((300, 300))

        # Преобразуем изображения в объекты ImageTk
        original_img_tk = ImageTk.PhotoImage(original_img)

        original_label.config(image=original_img_tk)
        # Обновляем отображение
        original_label.image = original_img_tk
        original_text = tk.Label(root, text="Исходное изображение")
        original_text.grid(row=2, column=0)


# Функция для сохранения сегментированного изображения
def save_segmented_image():
    global segmented_image1
    if segmented_image1 is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            segmented_img = Image.fromarray(segmented_image1)
            segmented_img.save(file_path)
            tk.messagebox.showinfo("Сохранение", "Сегментированное изображение успешно сохранено!")

    if segmented_image2 is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            segmented_img = Image.fromarray(segmented_image2)
            segmented_img.save(file_path)
            tk.messagebox.showinfo("Сохранение", "Сегментированное изображение успешно сохранено!")


# Создаем окно tkinter для выбора изображения
root = tk.Tk()
# Создаем новое окно tkinter
root.title("Сегментированное изображение")
root.geometry("1300x470")  # Устанавливаем размеры окна: ширина x высота

# Создаем и размещаем метки с изображениями и подписями в окне
original_label = tk.Label(root)
original_label.grid(row=1, column=0, padx=10, pady=10)

segmented_label = tk.Label(root)
segmented_label.grid(row=1, column=1, padx=10, pady=10)

k_means_label = tk.Label(root)
k_means_label.grid(row=1, column=2, padx=10, pady=10)

hirecal_label = tk.Label(root)
hirecal_label.grid(row=1, column=3, padx=10, pady=10)

# Фрейм для параметров GMM
gmm_frame = tk.Frame(root, bd=2, relief="groove")
gmm_frame.grid(row=0, column=100, rowspan=10, padx=10, pady=10, sticky="nsew")

# Создаем метку и ползунок для выбора количества компонент смеси GMM
n_components_label = tk.Label(gmm_frame, text="Количество компонент смеси GMM:")
n_components_label.grid(row=0, column=0, padx=10, pady=10)

n_components_slider = tk.Scale(gmm_frame, from_=1, to=10, orient=tk.HORIZONTAL)
n_components_slider.grid(row=1, column=0, padx=10, pady=10)

# Создаем переменную для хранения выбранного типа ковариационной матрицы
covariance_type_var = tk.StringVar(root)
covariance_type_var.set('tied')  # Устанавливаем значение по умолчанию

# Создаем меню для выбора типа ковариационной матрицы
covariance_type_menu = tk.OptionMenu(gmm_frame, covariance_type_var, 'full', 'tied', 'diag', 'spherical')
covariance_type_menu.grid(row=3, column=0, padx=10, pady=10)
covariance_type_label = tk.Label(gmm_frame, text="Тип ковариационной матрицы:")
covariance_type_label.grid(row=2, column=0, padx=10, pady=10)

# Создаем метку и поле ввода для выбора максимального количества итераций
max_iter_label = tk.Label(gmm_frame, text="Максимальное количество итераций:")
max_iter_label.grid(row=4, column=0, padx=10, pady=10)

max_iter_var = tk.StringVar(root, value="100")
max_iter_entry = tk.Entry(gmm_frame, textvariable=max_iter_var)
max_iter_entry.grid(row=5, column=0, padx=10, pady=10)

# Создаем метку и поле ввода для выбора количества инициализаций
n_init_label = tk.Label(gmm_frame, text="Количество инициализаций:")
n_init_label.grid(row=6, column=0, padx=10, pady=10)

n_init_var = tk.StringVar(root, value="1")
n_init_entry = tk.Entry(gmm_frame, textvariable=n_init_var)
n_init_entry.grid(row=7, column=0, padx=10, pady=10)

# Создаем метку и выпадающий список для выбора метода инициализации параметров
methods_label = tk.Label(gmm_frame, text="Сравнительный метод:")
methods_label.grid(row=10, column=0, padx=10, pady=10)

methods_var = tk.StringVar(root)
methods_var.set('Не установлен')  # Устанавливаем значение по умолчанию

methods_menu = tk.OptionMenu(gmm_frame, methods_var, 'Не установлен', 'k-means', 'Метод на основе порога',
                             'Сегментация краев')
methods_menu.grid(row=11, column=0, padx=10, pady=10)

# Создаем кнопку "Выбрать изображение"
button = tk.Button(root, text="Выбрать изображение", command=show_image)
button.grid(row=0, column=0, padx=10, pady=10)

# Создаем кнопку "Сегментировать изображение"
segment_button = tk.Button(root, text="Сегментировать изображение", command=segment_image)
segment_button.grid(row=0, column=1, padx=10, pady=10)

# Создаем кнопку "Сохранить сегментированное изображение"
save_button = tk.Button(root, text="Сохранить результат", command=save_segmented_image)
save_button.grid(row=3, column=0, columnspan=1, padx=10, pady=10)

time_label_1 = tk.Label(root, text="Время сегментации: -", fg="blue")
time_label_1.grid(row=3, column=1, padx=10, pady=10)

time_label_2 = tk.Label(root, text="Время сегментации: -", fg="blue")
time_label_2.grid(row=3, column=2, padx=10, pady=10)

root.mainloop()
