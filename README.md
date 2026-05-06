🛡️ AI Health Guardian
Система інтелектуальної діагностики серцево-судинних ризиків.

Проєкт розроблено в межах переддипломної практики.  

🎯 Основна мета
Автоматизація аналізу клінічних показників для виявлення ризиків захворювань за допомогою машинного навчання.  

🛠 Технології
Мова: Python  

ML: Scikit-learn (SVM, RobustScaler, IterativeImputer)  

Інтерфейс: CustomTkinter (Modern Dark Mode)  

📊 Результати
Обрано модель SVM (RBF Kernel) через пріоритетну для медицини метрику Recall:  

Accuracy: 0.689  

Recall: 0.727  

📂 Структура проєкту
main_app.py — графічний інтерфейс користувача.  

train_model.py — скрипт навчання та аналізу моделей.  

/data — набір даних (Heart Disease Dataset).  

/model — збережені артефакти моделей (.pkl).  

🚀 Швидкий запуск
Встановіть залежності: pip install -r requirements.txt

Запустіть програму: python main_app.py