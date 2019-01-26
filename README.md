# Search engine with Machine Learning
![alt text](https://github.com/Vivikar/jooble_se/blob/master/readme/pics/vloodlelogo.jpg)

Created during internship at Jooble and based on data provided by them this search system is capable of searching and showing results on job search queries. It consists of few services that run independently and exhange information between each other sending jsons. The services structure is next:

![alt text](https://github.com/Vivikar/jooble_se/blob/master/readme/pics/struct.png)

We implemented two step search system. 
``First step`` consists of usual boolean search, that finds which documents that have words from user's query. Then tf-df ranking is applied to find 100 most relevant documents.
``Second step`` is used only when user provided additional information along with his search query. When he filled "Skills" section we  apply an advanced sorting alghorithm to find the most relevant documents among that 100 previously choosed. Before adding a document to our inverted\forward index data storage we do an additional research on it. We use special Deep Learning alghorithm based on encoder – decoder model 
![alt text](https://github.com/Vivikar/jooble_se/blob/master/readme/pics/nnstr.jpg)

to detect "Requirements" and "Duties" fields in vacancy's text. Then we sort our list of results finding cosine similarity between vectors of those fields and user's skills he listed on search page. This gives us nuch better accuracy and helps to find the most relevant jobs for user, especially than there are a lot of good and possibly relevant vacancies.

And actually our neural network performed quite good
![alt text](https://github.com/Vivikar/jooble_se/blob/master/readme/pics/metrics.png)

Some examples of its work:
```
Курьер

пунктуальность четкий выполнение ставить задача опыт объяснить весь научить желание иметь стабильный заработок
работа закрепленный участок выбор оформление документ заполнение накладной
ооо премьеравто приглашать сотрудничество пеший курьер постоянный основа временный заработок выплата возможный ежедневный еженедельный условие работа закреплять участок выбор оплата проезд питание требование пунктуальность четкий выполнение ставить задача опыт объяснить весь научить желание иметь стабильный заработок обязанность доставка корреспонденция возможный зависимость закреплять территория оформление документ заполнение накладной прием оплата доставка прямой работодатель набирать большой штат сотрудник самый надежный человек готовый работать постоянный основа труд поощрять премия различный бонус

0 Expected: 0 Predicted 0 ооо
1 Expected: 0 Predicted 0 премьеравто
2 Expected: 0 Predicted 0 приглашать
3 Expected: 0 Predicted 0 сотрудничество
4 Expected: 0 Predicted 0 пеший
5 Expected: 0 Predicted 0 курьер
6 Expected: 0 Predicted 0 постоянный
7 Expected: 0 Predicted 0 основа
8 Expected: 0 Predicted 0 временный
9 Expected: 0 Predicted 0 заработок
10 Expected: 0 Predicted 0 выплата
11 Expected: 0 Predicted 0 возможный
12 Expected: 0 Predicted 0 ежедневный
13 Expected: 0 Predicted 0 еженедельный
14 Expected: 0 Predicted 0 условие
15 Expected: 0 Predicted 0 работа
16 Expected: 0 Predicted 0 закреплять
17 Expected: 0 Predicted 0 участок
18 Expected: 0 Predicted 0 выбор
19 Expected: 0 Predicted 0 оплата
20 Expected: 0 Predicted 0 проезд
21 Expected: 0 Predicted 0 питание
22 Expected: 0 Predicted 1 требование
23 Expected: 1 Predicted 1 пунктуальность
24 Expected: 1 Predicted 1 четкий
25 Expected: 1 Predicted 1 выполнение
26 Expected: 1 Predicted 1 ставить
27 Expected: 1 Predicted 1 задача
28 Expected: 1 Predicted 1 опыт
29 Expected: 1 Predicted 1 объяснить
30 Expected: 1 Predicted 1 весь
31 Expected: 1 Predicted 1 научить
32 Expected: 1 Predicted 1 желание
33 Expected: 1 Predicted 1 иметь
34 Expected: 1 Predicted 1 стабильный
35 Expected: 1 Predicted 0 заработок
36 Expected: 0 Predicted 1 обязанность
37 Expected: 0 Predicted 1 доставка
38 Expected: 0 Predicted 1 корреспонденция
39 Expected: 0 Predicted 1 возможный
40 Expected: 0 Predicted 1 зависимость
41 Expected: 0 Predicted 1 закреплять
42 Expected: 0 Predicted 1 территория
43 Expected: 0 Predicted 1 оформление
44 Expected: 0 Predicted 1 документ
45 Expected: 0 Predicted 0 заполнение
46 Expected: 0 Predicted 0 накладной
47 Expected: 0 Predicted 0 прием
48 Expected: 0 Predicted 0 оплата
49 Expected: 0 Predicted 0 доставка
50 Expected: 0 Predicted 0 прямой
51 Expected: 0 Predicted 0 работодатель
52 Expected: 0 Predicted 0 набирать
53 Expected: 0 Predicted 0 большой
54 Expected: 0 Predicted 0 штат
55 Expected: 0 Predicted 0 сотрудник
56 Expected: 0 Predicted 0 самый
57 Expected: 0 Predicted 0 надежный
58 Expected: 0 Predicted 0 человек
59 Expected: 0 Predicted 0 готовый
60 Expected: 0 Predicted 0 работать
61 Expected: 0 Predicted 0 постоянный
62 Expected: 0 Predicted 0 основа
63 Expected: 0 Predicted 0 труд
64 Expected: 0 Predicted 0 поощрять
65 Expected: 0 Predicted 0 премия
66 Expected: 0 Predicted 0 различный
67 Expected: 0 Predicted 0 бонус
```

```
Продавец-кассир
Обязанности:
~касса, продажа кофе, чай, помощь повару в нарезке, уборка рабочего места.
Требования:
~сан.книжка
Условия:
~работа в мини-кафе с 10 до 23. Два через два .официальное оформление по трудовому договору.
Мини кафе находится на ул. Калиновского.

0 Predicted 1  -  обязанность
1 Predicted 1  -  продажа
2 Predicted 1  -  кофе
3 Predicted 1  -  чай
4 Predicted 1  -  помощь
5 Predicted 1  -  повар
6 Predicted 1  -  нарезка
7 Predicted 1  -  уборка
8 Predicted 1  -  рабочий
9 Predicted 0  -  место
10 Predicted 1  -  требование
11 Predicted 1  -  условие
12 Predicted 0  -  оформление
13 Predicted 0  -  трудовой
14 Predicted 0  -  договор
15 Predicted 0  -  миня
16 Predicted 0  -  кафе
17 Predicted 0  -  находиться
18 Predicted 0  -  улица
19 Predicted 0  -  калиновский

```


## Setup
### setup.py scripts
Loads all necessary data to project from google drive:
- initial datasets and index
- doc_to_vec vectorizer

### Default services adresses
Check them to be avialebe on your machine
- server_indexer: port=13500
- server_text_processing: port=13501
- server_ranking: port=13502
- server_snippets: port=13503
- server_result_page_form: port=13504
- server_manager: port=13505

