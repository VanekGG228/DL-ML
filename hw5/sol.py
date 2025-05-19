import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv("ml-task-cosmos/train.csv")
test = pd.read_csv("ml-task-cosmos/test.csv")


X = train.drop(columns=["target"])
y = train["target"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = CatBoostClassifier(iterations=500, # количество итераций
                           depth=6,       # глубина деревьев
                           learning_rate=0.05, # скорость обучения
                           loss_function='Logloss', # функция потерь для бинарной классификации
                           verbose=200)    # выводим прогресс каждые 200 итераций

model.fit(X_train, y_train)


y_val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")


test_preds = model.predict(test_scaled)


pd.DataFrame(test_preds, columns=["target"]).to_csv("answers.csv", index=False)
