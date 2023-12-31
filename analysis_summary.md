
### Выполненные Действия и Результаты

1. **Предобработка данных и базовая линейная регрессия:**
   - Я провела базовую обработку данных для линейной регрессии, удалив некоторые категориальные признаки (`'name', 'fuel', 'seller_type', 'transmission', 'owner'`) и оставив числовые.
   - Была обучена и оценена простая модель линейной регрессии.
   - В результате модель показала самые худшие результаты.

2. **Использование Lasso и GridSearchCV:**
   - Я применила Lasso регрессию на стандартизированных данных.
   - Также использовала `GridSearchCV` для нахождения оптимального значения параметра `alpha` в Lasso.
   - Провела аналогичный поиск лучших параметров для ElasticNet и Ridge регрессий.
   - Ridge показала лучший R2 среди остальных моделей.


3. **Написание веб-сервиса**
   - После всех преобразований,была обучена модель Ridge регрессии с оптимальными параметрами, веса и scaler сохранены в файл.

4. **Feature Engineering:**
   - Попробовала выполнить дополнительно преобразование и добавление новых признаков, таких как логарифмирование целевой переменной, разделение на бренды, добавление бинарных признаков о владельцах и пропусках данных.
