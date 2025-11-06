# models/recommendation_models.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerRecommendationModel(Model):
    def __init__(self, vocab_size, embedding_dim=64, num_heads=4, num_layers=2):
        super(TransformerRecommendationModel, self).__init__()
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        self.encoder_layers = [
            TransformerEncoderLayer(embedding_dim, num_heads, 256)
            for _ in range(num_layers)
        ]
        
        self.final_layer = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(vocab_size, activation='softmax')
        ])
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        
        embeddings = self.embedding(inputs)
        embeddings = self.positional_encoding(embeddings, seq_len)
        
        for encoder_layer in self.encoder_layers:
            embeddings = encoder_layer(embeddings)
        
        last_output = embeddings[:, -1, :]
        
        return self.final_layer(last_output)

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def call(self, x, seq_len):
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.cast(positions, tf.float32)
        
        angle_rates = 1 / tf.pow(10000.0, 
                               (2 * (tf.range(self.d_model) // 2)) / tf.cast(self.d_model, tf.float32))
        
        angle_rads = tf.expand_dims(positions, 1) * tf.expand_dims(angle_rates, 0)
        
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.reshape(
            tf.stack([sines, cosines], axis=2),
            [seq_len, self.d_model]
        )
        
        return x + tf.expand_dims(pos_encoding, 0)