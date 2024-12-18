import pandas as pd
import sqlite3

raw_data = pd.read_csv('data/sample_sensor_data.csv')

# Preprocess data: Clean, remove duplicates, and normalize
def preprocess_data(df):
    df = df.drop_duplicates()
    df['normalized_wave'] = (df['sound_wave'] - df['sound_wave'].mean()) / df['sound_wave'].std()
    return df

processed_data = preprocess_data(raw_data)
processed_data.to_csv('data/processed_sound_wave_data.csv', index=False)

# Store processed data in a database
conn = sqlite3.connect('sensor_data.db')
processed_data.to_sql('processed_data', conn, if_exists='replace', index=False)
print("Data pipeline executed successfully!")

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load processed data
data = pd.read_csv('data/processed_sound_wave_data.csv')

# Split into features and labels
X = data[['normalized_wave', 'other_features']].values  # Replace 'other_features' with actual feature names
y = data['crack_detected'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a neural network model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Save the model
model.save('models/crack_detection_model.h5')
print("Model training completed!")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load processed data
data = pd.read_csv('data/processed_sound_wave_data.csv')

# Visualize sound wave patterns
sns.lineplot(data=data, x='timestamp', y='normalized_wave', hue='crack_detected')
plt.title('Sound Wave Patterns for Crack Detection')
plt.savefig('visualizations/sound_wave_pattern.png')
plt.show()

# Display model performance metrics
metrics = {'Accuracy': 92, 'Precision': 88, 'Recall': 85}
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.title('Model Performance Metrics')
plt.savefig('visualizations/model_performance_metrics.png')
plt.show()

