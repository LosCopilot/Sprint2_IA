import json
import os
import numpy as np
import torch # O TensorFlow, dependiendo de tu framework de IA
from torch.utils.data import Dataset, DataLoader

# --- Configuración ---
# ¡AJUSTA ESTO a la misma ruta absoluta que en Godot!
# Asegúrate de que termine con una barra para que os.path.join funcione correctamente.
GODOT_PLAYER_DATA_DIR = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\JSTB_AI\\player_data_for_ai\\" 

SEQUENCE_LENGTH = 128 
OVERLAP = 64 

def load_player_data(data_dir):
    """
    Carga todos los archivos JSON de datos del jugador del directorio especificado.
    
    Args:
        data_dir (str): Directorio base donde se encuentran los archivos JSON.
        
    Returns:
        list: Una lista de diccionarios, donde cada diccionario es un archivo JSON cargado.
    """
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
    
    Args:
        raw_player_data_list (list): Lista de diccionarios de datos del jugador cargados.
        sequence_length (int): Longitud deseada de cada secuencia de entrada.
        overlap (int): Número de puntos de superposición entre secuencias consecutivas.
        
    Returns:
        tuple: (list de np.array de entradas, list de np.array de etiquetas)
               Cada entrada es una secuencia de posiciones (x, y).
               Cada etiqueta es el siguiente punto en la secuencia.
    """
    input_sequences = []
    output_labels = []

    for game_data in raw_player_data_list:
        player_path = game_data.get("player_path", [])
        if not player_path:
            continue
        
        path_array = np.array(player_path, dtype=np.float32)

        # Normalizar las coordenadas (asumiendo un tamaño de pantalla Godot de 1280x720)
        screen_width = 1280.0 
        screen_height = 720.0
        
        normalized_path = path_array.copy()
        normalized_path[:, 0] = path_array[:, 0] / screen_width
        normalized_path[:, 1] = path_array[:, 1] / screen_height
        
        step = sequence_length - overlap
        for i in range(0, len(normalized_path) - sequence_length, step):
            input_seq = normalized_path[i : i + sequence_length]
            output_label = normalized_path[i + sequence_length]
            
            input_sequences.append(input_seq)
            output_labels.append(output_label)
            
    print(f"Generadas {len(input_sequences)} secuencias para el Transformer.")
    return input_sequences, output_labels

class PlayerPathDataset(Dataset):
    def __init__(self, input_sequences, output_labels):
        self.input_sequences = input_sequences
        self.output_labels = output_labels

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.input_sequences[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.output_labels[idx], dtype=torch.float32)
        return input_tensor, label_tensor

if __name__ == "__main__":
    # La ruta de los datos ya es la absoluta y directa
    player_data_folder = GODOT_PLAYER_DATA_DIR 
    
    print(f"Buscando datos del jugador en: {player_data_folder}")

    raw_data = load_player_data(player_data_folder)
    
    if not raw_data:
        print("No se encontraron datos de jugador para procesar. Asegúrate de que Godot haya guardado archivos en la ruta especificada.")
    else:
        inputs, labels = preprocess_for_transformer(raw_data, SEQUENCE_LENGTH, OVERLAP)
        
        if inputs and labels:
            dataset = PlayerPathDataset(inputs, labels)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 
            
            print(f"\nDatos listos para el entrenamiento del Transformer:")
            print(f"Número total de secuencias: {len(dataset)}")
            print(f"Forma de una secuencia de entrada (primer elemento): {dataset[0][0].shape}") 
            print(f"Forma de una etiqueta de salida (primer elemento): {dataset[0][1].shape}") 
            
            print("\n¡Los datos están preparados! Ahora puedes usarlos para entrenar tu modelo Transformer.")
        else:
            print("No se pudieron generar secuencias de entrenamiento a partir de los datos cargados.")