# models/customer_models.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class CustomerTrackingModel(Model):
    def __init__(self, num_classes=1):
        super(CustomerTrackingModel, self).__init__()
        
        self.backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(416, 416, 3)
        )
        
        self.neck = tf.keras.Sequential([
            layers.Conv2D(256, 1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 1, padding='same', activation='relu')
        ])
        
        self.detection_head = tf.keras.Sequential([
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(3 * (5 + num_classes), 1, activation='sigmoid')
        ])
    
    def call(self, inputs):
        features = self.backbone(inputs)
        enhanced_features = self.neck(features)
        detections = self.detection_head(enhanced_features)
        return detections

class BehaviorPredictionModel(Model):
    def __init__(self, sequence_length=10, num_features=8, num_classes=5):
        super(BehaviorPredictionModel, self).__init__()
        
        self.lstm_layers = tf.keras.Sequential([
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2)
        ])
        
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        
        self.classifier = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    def call(self, inputs):
        lstm_out = self.lstm_layers(inputs)
        
        attention_input = tf.expand_dims(lstm_out, axis=1)
        attention_output = self.attention(attention_input, attention_input)
        attention_output = tf.squeeze(attention_output, axis=1)
        
        combined = tf.concat([lstm_out, attention_output], axis=-1)
        
        return self.classifier(combined)