import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Caminhos
dataset_path = "dataset"
modelo_path = "modelo/modelo_libras.h5"

# Parâmetros
img_width, img_height = 200, 200
batch_size = 32
epochs = 10

# Geração de dados com aumento (data augmentation)
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

# Geradores de imagens
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Definir modelo CNN
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback para salvar o melhor modelo
checkpoint = ModelCheckpoint(
    modelo_path, monitor='val_accuracy', save_best_only=True, verbose=1
)

# Treinar o modelo
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[checkpoint]
)

print(f"\n✅ Modelo treinado e melhor versão salva em: {modelo_path}")