import customtkinter as ctk
import joblib
import pandas as pd
from tkinter import messagebox

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AIHealthAppV2(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Health Guardian - Localization Edition")
        self.geometry("560x850")
        self.resizable(False, False)

        # Завантаження артефактів
        self.model = joblib.load('models/health_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.imputer = joblib.load('models/imputer.pkl')

        # Заголовок
        self.header = ctk.CTkLabel(self, text="КЛІНІЧНИЙ АНАЛІЗ РИЗИКІВ", font=("Urbanist", 24, "bold"), text_color="#00E676")
        self.header.pack(pady=(20, 10))

        self.scroll_frame = ctk.CTkScrollableFrame(self, width=500, height=550, fg_color="#1A1A1A")
        self.scroll_frame.pack(padx=20, pady=10)

        # Словники для випадаючих списків (Текст -> Значення для моделі)
        self.mapping = {
            "sex": {"Чоловік": 1.0, "Жінка": 0.0},
            "cp": {
                "Стискаючий біль (виникає при русі)": 0.0, 
                "Дискомфорт (у щелепі, спині або плечі)": 1.0, 
                "Гострий біль (при вдиху або повороті)": 2.0, 
                "Болю немає": 3.0
            },
            "fbs": {"Менше 120 мг/дл (6.7 ммоль)": 0.0, "Більше 120 мг/дл (6.7 ммоль)": 1.0},
            "restecg": {"Норма": 0.0, "Аномалія ST-T": 1.0, "Гіпертрофія лівого шлуночка": 2.0},
            "exang": {"Ні": 0.0, "Так": 1.0}
        }

        self.inputs = {}

        # Додавання полів
        self.add_entry("Вік", "age", "напр. 45")
        self.add_dropdown("Стать", "sex", list(self.mapping["sex"].keys()))
        self.add_dropdown("Тип болю в грудях", "cp", list(self.mapping["cp"].keys()))
        self.add_entry("Артеріальний тиск (мм рт.ст.)", "trestbps", "напр. 120")
        self.add_entry("Холестерин (ммоль/л)", "chol", "напр. 5.2")
        self.add_dropdown("Рівень цукру натще", "fbs", list(self.mapping["fbs"].keys()))
        self.add_dropdown("Результат ЕКГ спокою", "restecg", list(self.mapping["restecg"].keys()))
        self.add_entry("Макс. пульс при навантаженні", "thalach", "напр. 155")
        self.add_dropdown("Стенокардія при фіз. навантаженні", "exang", list(self.mapping["exang"].keys()))

        # Кнопка
        self.btn = ctk.CTkButton(self, text="ОТРИМАТИ РЕЗУЛЬТАТ", command=self.predict, height=50, font=("Urbanist", 18, "bold"))
        self.btn.pack(pady=20, padx=40, fill="x")

        # Результат
        self.res_card = ctk.CTkFrame(self, height=80, fg_color="#111111")
        self.res_card.pack(fill="x", padx=30, pady=10)
        self.res_label = ctk.CTkLabel(self.res_card, text="Введіть дані пацієнта", font=("Urbanist", 16))
        self.res_label.pack(expand=True)

    def add_entry(self, label, key, placeholder):
        lbl = ctk.CTkLabel(self.scroll_frame, text=label, font=("Urbanist", 13, "bold"))
        lbl.pack(pady=(10, 0), anchor="w", padx=20)
        entry = ctk.CTkEntry(self.scroll_frame, placeholder_text=placeholder)
        entry.pack(fill="x", padx=20, pady=(0, 5))
        self.inputs[key] = entry

    def add_dropdown(self, label, key, options):
        lbl = ctk.CTkLabel(self.scroll_frame, text=label, font=("Urbanist", 13, "bold"))
        lbl.pack(pady=(10, 0), anchor="w", padx=20)
        dropdown = ctk.CTkOptionMenu(self.scroll_frame, values=options)
        dropdown.pack(fill="x", padx=20, pady=(0, 5))
        self.inputs[key] = dropdown

    def predict(self):
        try:
            # Збір та конвертація
            data = []
            
            # Числові значення
            age = float(self.inputs["age"].get())
            sex = self.mapping["sex"][self.inputs["sex"].get()]
            cp = self.mapping["cp"][self.inputs["cp"].get()]
            trestbps = float(self.inputs["trestbps"].get())
            
            # КОНВЕРТАЦІЯ ХОЛЕСТЕРИНУ: ммоль/л -> мг/дл (для моделі)
            chol_mmol = float(self.inputs["chol"].get())
            chol_mgdl = chol_mmol * 38.67 
            
            fbs = self.mapping["fbs"][self.inputs["fbs"].get()]
            restecg = self.mapping["restecg"][self.inputs["restecg"].get()]
            thalach = float(self.inputs["thalach"].get())
            exang = self.mapping["exang"][self.inputs["exang"].get()]

            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
            final_features = pd.DataFrame([[age, sex, cp, trestbps, chol_mgdl, fbs, restecg, thalach, exang]], 
                              columns=column_names)
            
            # Pipeline
            scaled = self.scaler.transform(self.imputer.transform(final_features))
            prediction = self.model.predict(scaled)[0]
            prob = self.model.predict_proba(scaled)[0][1]

            if prediction == 1:
                self.res_label.configure(text=f"РИЗИК ВИСОКИЙ ({prob:.1%})\nРекомендована консультація кардіолога", text_color="#FF5252")
            else:
                self.res_label.configure(text=f"РИЗИК НИЗЬКИЙ ({prob:.1%})\nПоказники в межах норми", text_color="#00E676")

        except Exception as e:
            messagebox.showwarning("Помилка", "Перевірте правильність заповнення всіх полів")

if __name__ == "__main__":
    app = AIHealthAppV2()
    app.mainloop()