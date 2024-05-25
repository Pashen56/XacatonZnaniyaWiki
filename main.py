import matplotlib.pyplot as plt
import os
import json
import re
import mwclient
import requests

from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

directory = 'xacaton'  # Папка с папками, в которых json файлы
blacklist_file = 'blacklist.txt'  # Файл с информацией о запрещенных ссылках

number_of_articles = 0  # Глобальная переменная для подсчета количества статей
number_of_blacklist_articles = 0  # Глобальная переменная для подсчета количества статей с запрещенными ссылками
number_of_links_in_blacklist = 0  # Глобальная переменная для # подсчета количества запрещенных ссылок в черном списке


def read_blacklist(file_path):
    black_list = []  # Создаем пустой список для хранения запрещенных выражений
    global number_of_links_in_blacklist  # Используем глобальное число запрещенных ссылок в черном списке

    with open(file_path, 'r') as f:
        for line in f:
            # Используем регулярное выражение для поиска шаблонов сайтов
            # print("line: ", line)
            if line.find('#') == -1:
                match = re.search(r'\\b(.*)', line)
            else:
                match = re.search(r'\\b(.*?) ', line)
            # print("match: ", match)
            if match:
                result = match.group(1).replace("\\", "")
                # Проверка на специальные символы
                if '*' in result:
                    result = result.split('*')[0]  # Удалить все, что идет после *
                if '[' in result:
                    result = result.split('[')[0]  # Удалить все, что идет после [
                if '(?:' in result:
                    base_pattern = result.split('(?:')[0].strip()  # Получаем базовый шаблон
                    if base_pattern == "":
                        base_pattern = result.split(')')[1].strip()
                    patterns = result.split('(?:')[1].split('|')  # Разбиваем на части по |
                    for pattern in patterns:
                        if ')' in pattern:  # Если закрывающая скобка, то добавляем только до нее
                            pattern = pattern.split(')')[0]
                            if result.endswith(')'):
                                black_list.append(base_pattern + pattern.strip())  # Добавляем в черный список
                                # print(base_pattern + pattern.strip())
                            else:
                                black_list.append(pattern.strip() + base_pattern)  # Добавляем в черный список
                                # print(pattern.strip() + base_pattern)
                        else:
                            if result.endswith(')'):
                                black_list.append(base_pattern + pattern.strip())  # Добавляем в черный список
                                # print(base_pattern + pattern.strip())
                            else:
                                black_list.append(pattern.strip() + base_pattern)  # Добавляем в черный список
                                # print(pattern.strip() + base_pattern)
                black_list.append(result)
        # print(black_list)
        number_of_links_in_blacklist = len(black_list)

    return black_list


def check_for_blacklist(text, black_list):
    for item in black_list:
        if re.search(item, text):
            # print(item)
            return True
    return False


def read_json_files_in_directory(in_directory):
    global number_of_articles  # Используем глобальное число количества статей
    global number_of_blacklist_articles  # Используем глобальное число статей с запрещенными ссылками
    datadict: dict[str, list] = {'text': [], 'label': []}  # Создаем словарь для хранения данных и меток
    for root, dirs, files in os.walk(in_directory):
        for file in files:
            if file.endswith('.json'):
                number_of_articles += 1
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    text = data.get('text', '')  # Получаем текст из данных JSON
                    datadict['text'].append(text)
                    if check_for_blacklist(text, blacklist):
                        datadict['label'].append(1)
                        number_of_blacklist_articles += 1
                    else:
                        datadict['label'].append(0)
    return datadict


# Загрузка черного списка и JSON-данных
blacklist = read_blacklist(blacklist_file)
datadict = read_json_files_in_directory(directory)

percentage_of_banned_articles = 100 * float(
    round((number_of_blacklist_articles / number_of_articles), 2))  # Процент статей с запрещенными ссылками
print(f"В черном списке было выделено: {number_of_links_in_blacklist} запрещенных ссылок.\n"
      f"Далее было проанализировано {number_of_articles} статей из дата-сета, "
      f"из которых в {number_of_blacklist_articles} "
      f"статьях было обнаружено упоминание запрещенных ссылок.\n"
      f"Итак, можно подвести итог, "
      f"что {percentage_of_banned_articles}% проверяемых статей содержали запрещенные ссылки.")

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(datadict['text'], datadict['label'], test_size=0.3,
                                                    random_state=42)

# Преобразование текста в числовые признаки
vectorizer: TfidfVectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Обучение модели (метод опорных векторов)
clf = LinearSVC(dual=True)
clf.fit(X_train_tfidf, y_train)

# Предсказание на тестовой выборке
y_pred = clf.predict(X_test_tfidf)

# Оценка производительности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# Функция для добавления пометки к статье
def mark_article_as_suspicious(site, page_title):
    page = site.pages[page_title]
    text = page.text()
    new_text = f"{{{{Может содержать запрещенные ссылки}}}}\n{text}"
    token = site.api('query', meta='tokens')['query']['tokens']['csrftoken']
    page.save(new_text, summary="Добавлена пометка о возможных запрещенных ссылках", token=token)


# Настройка клиента для взаимодействия с MediaWiki API
try:
    site = mwclient.Site('https://baza.znanierussia.ru', path='/w/')
    site.login('login', 'password')  # впишите свой логин и пароль

    # Получение CSRF-токена
    token = site.api('query', meta='tokens')['query']['tokens']['csrftoken']

    url = "https://znanierussia.ru/articles/15_июня"
    new_text = requests.get(url).text

    # Преобразуем новый текст в числовые признаки
    new_text_tfidf = vectorizer.transform([new_text])

    # Предсказание класса для нового текста
    prediction = clf.predict(new_text_tfidf)

    # Вывод результата предсказания и добавление пометки, если ссылки запрещены
    page_title = "15_июня"
    if prediction == 1:
        print("Обнаружены запрещенные ссылки в тексте.")
        mark_article_as_suspicious(site, page_title)
    else:
        print("Запрещенные ссылки в тексте не обнаружены.")
except requests.exceptions.HTTPError as e:
    print(f"Ошибка HTTP: {e}")
except requests.exceptions.RequestException as e:
    print(f"Ошибка запроса: {e}")
except mwclient.errors.APIError as e:
    print(f"Ошибка API MediaWiki: {e}")
except Exception as e:
    print(f"Произошла ошибка: {e}")

# print(datadict['label'])

# Построение ROC-кривой
fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(X_test_tfidf))
roc_auc = auc(fpr, tpr)

# Создание графика ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) curve')
plt.legend(loc='lower right')
plt.show()
