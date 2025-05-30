Emirhan AK 2106A020

# 🧠 Deep Learning Network Builder GUI – Full Feature Documentation

This document explains each key feature of the GUI-based deep learning network builder. Each section includes a feature description and the corresponding Python code snippet from the application.

---

## 🔧 Dynamic Layer Configuration

### Description
Users can interactively add or remove Dense, Conv2D, Dropout, BatchNormalization, MaxPooling, and Flatten layers through the GUI. Each layer can be configured with parameters such as units, activation functions, kernel sizes, dropout rates, and L2 regularization.

### Code
```python
def add_layer_dialog(self, existing_layer=None, index=None):
    ...
    type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout", "BatchNormalization"])
    ...
    if layer_type == "Dense":
        units_input = QSpinBox()
        ...
        l2_checkbox = QCheckBox("Use L2 Regularization")
        l2_rate = QDoubleSpinBox()
        ...
    elif layer_type == "Conv2D":
        filters_input = QSpinBox()
        ...
    elif layer_type == "Dropout":
        rate_input = QDoubleSpinBox()
        ...
```
---

## 🧠 CNN Layer Support

### Description
The application allows adding convolutional neural network layers with configurable filters, kernel size, activation, and pooling. Designed for image datasets like MNIST.

### Code
```python
def create_cnn_controls(self):
    ...
    type_combo.addItems(["Conv2D", "MaxPooling2D", "Flatten", "Dropout", "BatchNormalization"])
    ...
    if layer_type == "Conv2D":
        filters_spin = QSpinBox()
        ...
    elif layer_type == "MaxPooling2D":
        pool_input = QLineEdit("2,2")
        ...
```

---

## 🧬 RNN Layer Support (LSTM/GRU)

### Description
Allows users to build recurrent neural networks using LSTM or GRU layers with options for setting the number of units and whether to return sequences.

### Code
```python
def add_rnn_layer_dialog(self):
    type_combo = QComboBox()
    type_combo.addItems(["LSTM", "GRU"])
    ...
    def save():
        self.rnn_layer_config.append({
            "type": type_combo.currentText(),
            "params": {
                "units": units_spin.value(),
                "return_sequences": return_seq_checkbox.isChecked()
            }
        })
```

---

## ⚙️ Optimizer Selection

### Description
The GUI provides options to select popular optimizers: Adam, SGD, and RMSprop.

### Code
```python

self.optimizer_combo = QComboBox()
self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])

            # 4. Optimizer selection
            optimizer_name = self.optimizer_combo.currentText()
            optimizer_class = {
                "Adam": tf.keras.optimizers.Adam,
                "SGD": tf.keras.optimizers.SGD,
                "RMSprop": tf.keras.optimizers.RMSprop
            }.get(optimizer_name, tf.keras.optimizers.Adam)
            optimizer = optimizer_class(learning_rate=learning_rate)
```

---

## 📉 Learning Rate Scheduling

### Description
Users can apply learning rate decay strategies such as step decay and exponential decay.

### Code
```python
self.lr_schedule_combo = QComboBox()
self.lr_schedule_combo.addItems(["None", "Step Decay", "Exponential Decay"])
```

---

## 🛡️ Regularization (Dropout & L2)

### Description
Dropout and L2 regularization can be enabled per layer to prevent overfitting.

### Code
```python
# In Dense/Conv2D layer dialog
l2_checkbox = QCheckBox("Use L2 Regularization")
dropout_input = QDoubleSpinBox()
...
params['kernel_regularizer'] = regularizers.l2(params.get('l2_rate', 0.001))
```

---

## ⏹️ Early Stopping

### Description
Stops training automatically when validation loss stops improving.

### Code
```python
if self.early_stop_checkbox.isChecked():
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))
```

---

## 📈 Gradient Histogram Visualization

### Description
Visualizes distribution of weight gradients after each epoch.

### Code
```python
class GradientHistogramCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        ...
        ax.hist(flat_grads.numpy(), bins=50)
```

---

## 💾 Save/Load Model

### Description
Models can be saved in `.h5` or `.json` format, and reloaded for future use.

### Code
```python
def save_model_architecture(self):
    ...
    model.save(file_path)

def load_model_architecture(self):
    ...
    model = load_model(file_path)
```

---

## 🧠 Pretrained Model Support (VGG16, ResNet)

### Description
Supports loading pretrained models and fine-tuning them on custom image datasets.

### Code
```python
from tensorflow.keras.applications import VGG16, ResNet50
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
...
model = Model(inputs=base_model.input, outputs=predictions)
```

---

## 🖼️ Data Augmentation

### Description
Applies image augmentations such as rotation, zoom, and horizontal flip to boost model generalization.

### Code
```python
self.rotation_spin = QSpinBox()
self.zoom_spin = QDoubleSpinBox()
self.flip_checkbox = QCheckBox("Enable Horizontal Flip")
```

---

## 📊 Final Evaluation Metrics

### Description
Accuracy and F1-score (classification) or MSE and MAE (regression) are calculated after training.

### Code
```python
accuracy = accuracy_score(y_test_labels, y_pred)
f1 = f1_score(y_test_labels, y_pred, average="macro")
```

---

> Developed using PyQt6 and TensorFlow for educational and experimental purposes.
