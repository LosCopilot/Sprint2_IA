import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import torch_directml  # Importación clave para AMD en Windows

# --- Configuración General ---
# ¡AJUSTA ESTO a la misma ruta absoluta que en Godot!
GODOT_PLAYER_DATA_DIR = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\JSTB_AI\\player_data_for_ai\\" 
MODEL_SAVE_DIR = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\JSTB_AI\\trained_models\\"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Parámetros del Transformer ---
SEQUENCE_LENGTH = 128
OVERLAP = 64
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
D_MODEL = 128
N_HEADS = 8
NUM_ENCODER_LAYERS = 4
DROPOUT = 0.1
SCREEN_WIDTH = 1280.0
SCREEN_HEIGHT = 720.0

# --- Modificación clave: Configuración del dispositivo ---
DEVICE = torch_directml.device() if torch_directml.is_available() else torch.device("cpu")
print(f"\n=== Información del Dispositivo ===")
print(f"Usando: {DEVICE}")
print(f"Nombre GPU: {torch_directml.device_name(0) if torch_directml.is_available() else 'CPU'}\n")


# --- 1. Carga y Preprocesamiento de Datos ---
def load_player_data(data_dir):
    """Carga todos los archivos JSON de datos del jugador del directorio especificado."""
    all_data = []
    if not os.path.exists(data_dir):
        print(f"Error: Directorio de datos del jugador no encontrado: {data_dir}")
        return []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and filename.startswith("player_path_data_"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
                print(f"Cargado: {filename}")
            except json.JSONDecodeError as e:
                print(f"Error al leer JSON en {filename}: {e}")
            except Exception as e:
                print(f"Error desconocido al cargar {filename}: {e}")
    return all_data

def preprocess_for_transformer(raw_player_data_list, sequence_length, overlap):
    """
    Preprocesa los datos crudos de la ruta del jugador en secuencias para un Transformer.
    Normaliza las coordenadas a un rango de 0-1.
    """
    input_sequences = []
    output_labels = [] # El siguiente punto (x,y)
    
    print(f"Procesando {len(raw_player_data_list)} archivos de juego...")

    for game_data in raw_player_data_list:
        player_path = game_data.get("player_path", [])
        if not player_path or len(player_path) < sequence_length + 1:
            # Necesitamos al menos (sequence_length + 1) puntos para formar una secuencia
            # y su etiqueta (el siguiente punto).
            continue
        
        path_array = np.array(player_path, dtype=np.float32)

        # Normalizar las coordenadas a 0-1
        normalized_path = path_array.copy()
        normalized_path[:, 0] = path_array[:, 0] / SCREEN_WIDTH
        normalized_path[:, 1] = path_array[:, 1] / SCREEN_HEIGHT
        
        step = sequence_length - overlap
        if step <= 0: # Asegurarse de que el paso sea al menos 1
            step = 1

        for i in range(0, len(normalized_path) - sequence_length, step):
            if i + sequence_length + 1 > len(normalized_path): # Asegurarse de que haya un siguiente punto para la etiqueta
                break
            
            input_seq = normalized_path[i : i + sequence_length]
            output_label = normalized_path[i + sequence_length] # El punto siguiente a la secuencia
            
            input_sequences.append(input_seq)
            output_labels.append(output_label)
            
    print(f"Generadas {len(input_sequences)} secuencias para el Transformer.")
    if not input_sequences:
        print("ADVERTENCIA: No se generaron suficientes secuencias. Asegúrate de tener suficientes datos y una SEQUENCE_LENGTH apropiada.")
    return input_sequences, output_labels

class PlayerPathDataset(Dataset):
    def __init__(self, input_sequences, output_labels):
        self.input_sequences = input_sequences
        self.output_labels = output_labels

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        # Convertir a tensores PyTorch
        input_tensor = torch.tensor(self.input_sequences[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.output_labels[idx], dtype=torch.float32)
        return input_tensor, label_tensor

# --- 2. Arquitectura del Modelo Transformer ---
class PositionalEncoding(nn.Module):
    """
    Agrega información posicional a las incrustaciones de entrada.
    Necesario para Transformers, ya que no tienen recursión o convoluciones.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x tiene forma (secuencia_largo, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class PlayerPathTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dropout):
        super().__init__()
        self.d_model = d_model
        
        # Capa de incrustación lineal para mapear las coordenadas (x,y) a d_model
        # input_dim es 2 (para x, y)
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=SEQUENCE_LENGTH)
        
        # Bloques del encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Capa de salida para predecir las coordenadas (x,y)
        # Tomamos la salida del último token o un promedio, aquí tomamos el último token
        self.fc_out = nn.Linear(d_model, input_dim) # input_dim = 2 (x, y)

    def forward(self, src):
        # src tiene forma (batch_size, sequence_length, input_dim)
        
        # Incrustar las entradas
        src = self.embedding(src) # Ahora (batch_size, sequence_length, d_model)
        
        # Añadir codificación posicional (PyTorch Transformer ya lo maneja bien con batch_first=True)
        # Si no usas batch_first=True, necesitarías transponer src a (sequence_length, batch_size, d_model)
        # para que PositionalEncoding funcione como se mostró en la clase.
        # Con batch_first=True, PositionalEncoding debe ajustarse o aplicarse antes del TransformerEncoder
        # En este caso, simplemente añadimos PE a las incrustaciones
        
        # src_with_pe = self.positional_encoding(src.permute(1, 0, 2)).permute(1, 0, 2) # Si PositionalEncoding espera (seq_len, batch, d_model)
        # Una forma más simple si ya tenemos batch_first=True:
        # src = src + self.positional_encoding.pe[:src.size(1), :].transpose(0,1) # No funciona directamente
        
        # Corregir la aplicación de PositionalEncoding para batch_first=True
        # Si positional_encoding.pe es (max_len, 1, d_model)
        # Queremos sumarlo a src (batch, seq_len, d_model)
        # Podemos expandir pe para que coincida con el tamaño del batch
        src = src + self.positional_encoding.pe[:src.size(1), :].squeeze(1) # Squeeze para quitar la dimensión 1 extra
        
        # Pasar por el encoder Transformer
        output = self.transformer_encoder(src) # (batch_size, sequence_length, d_model)
        
        # Tomar la salida del último token de la secuencia para la predicción
        # Esto asume que el último token encapsula la información de toda la secuencia
        # O podrías promediar: output.mean(dim=1)
        predicted_coords = self.fc_out(output[:, -1, :]) # (batch_size, input_dim)
        
        return predicted_coords

# --- 3. Bucle de Entrenamiento ---
def train_transformer_model(dataloader, model, criterion, optimizer, num_epochs, model_save_dir):
    model.train() # Poner el modelo en modo entrenamiento
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad() # Limpiar gradientes anteriores
            
            outputs = model(inputs) # Pasar las entradas al modelo
            
            loss = criterion(outputs, labels) # Calcular la pérdida (MSE entre predicción y etiqueta real)
            
            loss.backward() # Propagación hacia atrás
            optimizer.step() # Actualizar pesos del modelo
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Época {epoch+1}/{num_epochs}, Pérdida: {avg_epoch_loss:.6f}")

    end_time = time.time()
    print(f"\nEntrenamiento completado en {end_time - start_time:.2f} segundos.")

    # Guardar el modelo entrenado
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"player_path_transformer_epoch{num_epochs}_{timestamp}.pth"
    model_path = os.path.join(model_save_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en: {model_path}")

# --- 4. Función para Cargar un Modelo Existente ---
def load_trained_model(model_path, input_dim, d_model, n_heads, num_encoder_layers, dropout):
    model = PlayerPathTransformer(input_dim, d_model, n_heads, num_encoder_layers, dropout).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Poner el modelo en modo evaluación
    print(f"Modelo cargado desde: {model_path}")
    return model

# --- Ejecución Principal ---
if __name__ == "__main__":
    print("Iniciando preparación y entrenamiento del Transformer...")
    
    # 1. Cargar y preprocesar los datos
    raw_data = load_player_data(GODOT_PLAYER_DATA_DIR)
    
    if not raw_data:
        print("No se encontraron datos de jugador. Asegúrate de que Godot haya guardado archivos JSON en la ruta especificada.")
    else:
        inputs, labels = preprocess_for_transformer(raw_data, SEQUENCE_LENGTH, OVERLAP)
        
        if inputs and labels:
            dataset = PlayerPathDataset(inputs, labels)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            print(f"\nResumen de Datos para Entrenamiento:")
            print(f"Número total de secuencias: {len(dataset)}")
            print(f"Forma de una secuencia de entrada: {dataset[0][0].shape} (SEQUENCE_LENGTH, 2)")
            print(f"Forma de una etiqueta de salida: {dataset[0][1].shape} (2,)")
            print(f"Tamaño del lote (Batch Size): {BATCH_SIZE}")
            
            # 2. Inicializar el Modelo Transformer
            input_dimension = 2 # Coordenadas (x, y)
            model = PlayerPathTransformer(input_dimension, D_MODEL, N_HEADS, NUM_ENCODER_LAYERS, DROPOUT).to(DEVICE)
            
            # 3. Definir la Función de Pérdida y el Optimizador
            criterion = nn.MSELoss() # Mean Squared Error para regresión
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            # 4. Entrenar el Modelo
            print(f"\nIniciando entrenamiento por {NUM_EPOCHS} épocas...")
            train_transformer_model(dataloader, model, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_DIR)
            
            # --- Cómo Probar el Modelo (Ejemplo de Inferencia) ---
            print("\n--- Demostración de Inferencia del Modelo Entrenado ---")
            model.eval() # Poner el modelo en modo evaluación
            
            # Tomar un lote de datos de prueba (aquí, solo el primer lote)
            # En un escenario real, usarías un conjunto de datos de prueba separado
            for test_inputs, test_labels in dataloader:
                test_inputs = test_inputs.to(DEVICE)
                
                with torch.no_grad(): # Desactivar el cálculo de gradientes para la inferencia
                    predictions = model(test_inputs)
                
                print(f"Ejemplo de Entrada (normalizada, primeros 5 puntos):\n{test_inputs[0, :5].cpu().numpy()}")
                print(f"Predicción del siguiente punto (normalizada):\n{predictions[0].cpu().numpy()}")
                print(f"Etiqueta Real del siguiente punto (normalizada):\n{test_labels[0].cpu().numpy()}")

                # Desnormalizar para ver los valores reales en coordenadas de pantalla
                predicted_unnormalized = predictions[0].cpu().numpy() * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
                label_unnormalized = test_labels[0].cpu().numpy() * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
                
                print(f"Predicción del siguiente punto (desnormalizada):\n{predicted_unnormalized}")
                print(f"Etiqueta Real del siguiente punto (desnormalizada):\n{label_unnormalized}")
                
                # Puedes calcular la diferencia para ver qué tan cerca está la predicción
                diff = np.abs(predicted_unnormalized - label_unnormalized)
                print(f"Diferencia absoluta (en píxeles):\n{diff}")

                break # Solo procesar el primer lote para la demostración
            
            print("\nEntrenamiento y prueba del Transformer completados. ¡Ahora tienes un modelo para futuras inferencias!")
        else:
            print("No se pudieron generar secuencias de entrenamiento a partir de los datos cargados.")