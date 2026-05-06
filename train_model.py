import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# --- 1. ПІДГОТОВКА (ADVANCED) ---
df = pd.read_csv('data/heart.csv').drop_duplicates()

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
X = df[features]
y = df['target']

# Розумне заповнення пропусків (прогнозує пропуски на основі кореляцій)
imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X)

# Надійне масштабування (стійке до викидів)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression

# Додаємо Логістичну регресію у словник моделей
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, kernel='rbf', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

# --- ПОРІВНЯННЯ ---
results = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    results.append({"Model": name, "CV Accuracy": np.mean(cv_scores), "Test Accuracy": test_acc})

# Вивід у вигляді таблиці
results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
print("\n=== Порівняльна таблиця алгоритмів ===")
print(results_df.to_string(index=False))

# Виведення коефіцієнтів для регресії (щоб розуміти вплив факторів)
lr_model = models["Logistic Regression"]
importance = lr_model.coef_[0]
print("\n=== Вплив факторів (за Логістичною регресією) ===")
for i, feat in enumerate(features):
    print(f"{feat}: {importance[i]:.4f}")

from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score

# --- ПОРІВНЯННЯ ТА ДЕТАЛЬНИЙ АНАЛІЗ ---
detailed_results = []
confusion_matrices = {}

print("\n" + "="*60)
print("ДЕТАЛЬНИЙ АНАЛІЗ МЕТРИК КЛАСИФІКАЦІЇ")
print("="*60)

for name, model in models.items():
    # Прогноз на тестових даних
    y_pred = model.predict(X_test)
    
    # Розрахунок метрик
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) # Чутливість (важливо для медицини!)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    confusion_matrices[name] = cm
    detailed_results.append({
        "Model": name, 
        "Accuracy": acc, 
        "Recall (Чуйність)": rec, 
        "F1-Score": f1
    })
    
    print(f"\n>>> Модель: {name}")
    print(classification_report(y_test, y_pred, target_names=['Здоровий', 'Ризик']))
    print(f"Матриця помилок:\n{cm}")

# Вивід фінальної порівняльної таблиці
final_df = pd.DataFrame(detailed_results).sort_values(by="Accuracy", ascending=False)
print("\n" + "="*60)
print("ПІДСУМКОВА ТАБЛИЦЯ МЕТРИК")
print("="*60)
print(final_df.to_string(index=False))

# Автоматичний вибір найкращої моделі за Recall (для медицини це пріоритет)
# або за Accuracy, якщо Recall однаковий
best_model_name = final_df.sort_values(by=["Recall (Чуйність)", "Accuracy"], ascending=False).iloc[0]['Model']
best_model_obj = models[best_model_name]

print(f"\nРЕКОМЕНДОВАНО ДЛЯ ВПРОВАДЖЕННЯ: {best_model_name}")

# --- 4. ЗБЕРЕЖЕННЯ НАЙКРАЩОЇ МОДЕЛІ ---
joblib.dump(best_model_obj, 'models/health_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(imputer, 'models/imputer.pkl')
print(f"\nАртефакти найкращої моделі ({best_model_name}) збережено!")