# models/product_models.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class ProductDetectionModel(Model):
    def __init__(self, num_classes=80):
        super(ProductDetectionModel, self).__init__()
        
        self.feature_extractor = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(640, 640, 3)
        )
        
        self.fpn = self._build_fpn()
        
        self.classification_head = tf.keras.Sequential([
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(num_classes, 1, activation='softmax')
        ])
        
        self.regression_head = tf.keras.Sequential([
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(4, 1, activation='sigmoid')
        ])
    
    def _build_fpn(self):
        c3, c4, c5 = [
            self.feature_extractor.get_layer('conv3_block4_out').output,
            self.feature_extractor.get_layer('conv4_block6_out').output,
            self.feature_extractor.get_layer('conv5_block3_out').output
        ]
        
        p5 = layers.Conv2D(256, 1, name='fpn_c5p5')(c5)
        p5_upsampled = layers.UpSampling2D(name='fpn_p5upsampled')(p5)
        
        p4 = layers.Conv2D(256, 1, name='fpn_c4p4')(c4)
        p4 = layers.Add(name='fpn_p4add')([p5_upsampled, p4])
        p4_upsampled = layers.UpSampling2D(name='fpn_p4upsampled')(p4)
        
        p3 = layers.Conv2D(256, 1, name='fpn_c3p3')(c3)
        p3 = layers.Add(name='fpn_p3add')([p4_upsampled, p3])
        
        p6 = layers.Conv2D(256, 3, strides=2, padding='same', name='fpn_p6')(c5)
        p7 = layers.Conv2D(256, 3, strides=2, padding='same', name='fpn_p7')(p6)
        
        return [p3, p4, p5, p6, p7]
    
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        fpn_features = self.fpn
        
        classifications = []
        regressions = []
        
        for feature in fpn_features:
            cls = self.classification_head(feature)
            reg = self.regression_head(feature)
            
            classifications.append(cls)
            regressions.append(reg)
        
        return classifications, regressions

class ShelfOccupancyModel(Model):
    def __init__(self):
        super(ShelfOccupancyModel, self).__init__()
        
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling2D()
        ])
        
        self.regressor = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, inputs):
        features = self.encoder(inputs)
        occupancy = self.regressor(features)
        return occupancy