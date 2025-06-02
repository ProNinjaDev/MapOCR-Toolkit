from config import cleaning_image_config as config
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

print("[INFO] Загрузка датасета...")
features= []
targets=[]

for row in open(config.FEATURES_PATH):
    row = row.strip().split(",")
    row = [float(x) for x in row]
    target = row[0]
    pixels = row[1:]

    features.append(pixels)
    targets.append(target)

features = np.array(features,dtype = "float")
target = np.array(targets,dtype = "float")

(trainX,testX,trainY,testY) = train_test_split(features, target,
                                             test_size = 0.25, random_state = 42)

print("[INFO] Тренировка модели...")
model = RandomForestRegressor(n_estimators=10)
model.fit(trainX, trainY)

print("[INFO] Оценка модели...")
preds = model.predict(testX)
rmse = np.sqrt(mean_squared_error(testY, preds))
print(f"[INFO] RMSE: {rmse}")

with open(config.MODEL_PATH, "wb") as f:
    f.write(pickle.dumps(model))