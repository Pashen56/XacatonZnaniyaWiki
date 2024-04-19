import matplotlib.pyplot as plt
import os
import json
import re

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

# Пример нового текста для вердикта о предсказании модели
new_text = ("""
'''Даниэла Клетте''' (полное имя '''Даниэла Мария Луиза Клетте''', {{lang-de|Daniela Marie Luise Klette}}; 
род. [[5 ноября]] [[1958]]) — член бывшей немецкой леворадикальной организации «[[Фракция Красной армии]]» 
(RAF). == Биография == Родилась 5 ноября 1958 года в городе Карлсруэ в семье Адельхайд Клетте (1931—2016)
<ref>[https://www.stadtkirche-karlsruhe.de/media/download/variant/84541/201604_gemeindebrief.pdf Gemeindebrief 
der Alt- und Mittelstadtgemeinde]{{ref-de}}</ref>. Росла в Карлсруэ. С 1975 года принимала активное участие 
в различных левых экстремистских группировках. В 1978 году она присоединилась к RAF, где познакомилась с 
Вольфгангом Грамсом и Биргитой Хогефельд, которые были ведущими фигурами этой организации в 1980-х годах. 
Участвовала в различных экстремистских акциях в конце 1970-х — начале 1980-х годов. Подозревается в причастности 
к большому количеству преступлений в 1990-х годах. С конца 1990-х годов, когда «Фракция Красной армии» прекратила 
своё существование, Даниэла Клетте находилась на нелегальном положении и разыскивалась полицией. Розыски Клетте и её 
сообщников велись по настоящее время.<ref>[https://web.archive.org/web/20200506104558if_/https://eumostwanted.eu/ 
Europeans Most Wanted Fugitives]{{ref-de}}</ref> Сначала для поиска использовались фотографии 1980-х годов. В 2000-х 
годах Федеральное управление уголовной полиции использовало их фотографий, на которых с помощью компьютера искусственно 
состаривали людей. За важную информацию об их нахождении было предложено вознаграждение в размере {{nobr|до 150 000 
евро.<ref>[https://www.faz.net/aktuell/politik/inland/ermittler-bitten-um-hinweise-zu-frueheren-raf-terroristen-19508753.html 
Ermittler bitten um Hinweise zu früheren RAF-Terroristen]{{ref-de}}</ref>}} Даниэла Клетте была арестована 26 февраля 
2024 года. Скрываясь более тридцати лет, жила на съёмной квартире в центре [[Берлин]]а, в районе Кройцберг. При обыске 
квартиры, в которой она проживала под именем Клаудии Ивоне около 20 лет, были обнаружены два пистолетных магазина и 
патроны к пистолету, а также изъят итальянский паспорт.<!-- ref>
[https://www.welt.de/politik/deutschland/article250295122/Daniela-Klette-RAF-Terroristin-in-U-Haft-Hinweis-kam-aus-der-Bevoelkerung.html 
RAF-Terroristin Klette in U-Haft — Entscheidender Hinweis kam aus der Bevölkerung]{{ref-de}}</ref --><ref>
[https://www.ndr.de/nachrichten/niedersachsen/oldenburg_ostfriesland/Ex-RAF-Terroristin-Daniela-Klette-Kriegswaffen-in-Wohnung-entdeckt,klette100.html 
Ex-RAF-Terroristin Daniela Klette: Kriegswaffen in Wohnung entdeckt]{{ref-de}}</ref> На следующий день после ареста 
при обыске квартиры была найдена граната. Во время проведения следственных действий были эвакуированы жильцы дома, где 
проживала Даниэла, а также дома напротив.<ref>
[https://www.tagesspiegel.de/berlin/weiteres-gebaude-wurde-teilweise-evakuiert-schusswaffen-und-granate-im-haus-von-ex-raf-terroristin-daniela-klette-in-berlin-gefunden-11287405.html 
Durchsuchung bei Ex-RAF-Terroristin in Berlin beendet — Evakuierte kehren zurück]{{ref-de}}</ref>
17 февраля 2024 года полиция ФРГ в результате спецоперации в Вуппертале задержала двух человек. Сообщалось, что один из 
них — 69-летний участник «Фракции Красной армии» Эрнст Фолькер Штаубе, его разыскивали 25 лет.<ref>
[https://news.rambler.ru/incidents/52340152-politsiya-berlina-posle-30-let-rozyska-zaderzhala-terroristku-danielu-klette/ 
Полиция Берлина после 30 лет розыска задержала террористку Даниэлу Клетте]</ref> Немецкие власти Германии обвиняют Клетте 
и двух ее сообщников — Буркхарда Гарвега и Эрнста Фолькера Штауба — в покушении на убийство и серии ограблений, 
совершенных в период с 1999 по 2016 год).<ref name=life.ru>[https://life.ru/p/1642478 В Германии после 30 лет розыска 
поймали известную террористку]</ref> == Примечания == {{примечания}} == Ссылки ==
* [https://www.theguardian.com/world/2016/jul/26/german-red-army-faction-militants-wanted-dutch-police German Red Army Faction trio wanted by Dutch police]{{ref-en}}
* [https://dzen.ru/a/ZeAuEaLbqnzP_pOw RAF — и взяли]
* [https://www.rbc.
ru/politics/03/03/2024/65e434999a7947679d74e847 Спецназ вступил в бой с «пенсионерами» «Фракции Красной армии» в Берлине]
[[Категория:Родившиеся в Карлсруэ]]
""")

# Преобразуем новый текст в числовые признаки
new_text_tfidf = vectorizer.transform([new_text])

# Предсказание класса для нового текста
prediction = clf.predict(new_text_tfidf)

# Вывод результата предсказания
if prediction == 1:
    print("Обнаружены запрещенные ссылки в тексте.")
else:
    print("Запрещенные ссылки в тексте не обнаружены.")

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
