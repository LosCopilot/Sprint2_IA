import json
import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.id3 import ID3NoHeaderError
import librosa
import librosa.display
import random
from sklearn.preprocessing import MinMaxScaler
import torch_directml # Importación clave para AMD en Windows

# --- Configuración General (debe coincidir con la de entrenamiento y Godot) ---
GODOT_PLAYER_DATA_DIR = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\JSTB_AI\\player_data_for_ai\\"
MODEL_SAVE_DIR = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\JSTB_AI\\trained_models\\"

# Directorio donde se guardarán los JSON FINALES generados por la IA (y donde Godot leerá los niveles)
AI_GENERATED_LEVELS_DIR = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\version4\\Levels\\"
os.makedirs(AI_GENERATED_LEVELS_DIR, exist_ok=True) # Asegurarse de que exista

# Donde están tus archivos de música de Godot (para obtener la duración y análisis)
GODOT_MUSIC_DIR = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\JSTB_AI\\Audio\\"

SEQUENCE_LENGTH = 128         # Debe coincidir con la longitud de secuencia de entrenamiento
D_MODEL = 128                 # Debe coincidir con el tamaño del modelo
N_HEADS = 8                   # Debe coincidir con el número de cabezas de atención
NUM_ENCODER_LAYERS = 4        # Debe coincidir con el número de capas del encoder
DROPOUT = 0.1                 # Debe coincidir con el dropout usado en entrenamiento

SCREEN_WIDTH = 1280.0         # Ancho de la pantalla de Godot para normalización
SCREEN_HEIGHT = 720.0         # Alto de la pantalla de Godot para normalización

# --- Modificación clave: Configuración del dispositivo unificada ---
DEVICE = torch_directml.device() if torch_directml.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# --- Configuración para la Generación de Obstáculos (ahora en este script) ---
OBSTACLE_TYPES = ["square", "triangle", "circle", "line"]
# Normalizador para las características de la música
scaler = MinMaxScaler()

# --- Arquitectura del Modelo Transformer (¡Copiar de tu script de entrenamiento!) ---
class PositionalEncoding(nn.Module):
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
        # x es (seq_len, batch_size, d_model) si batch_first=False
        # Pero si batch_first=True, x es (batch_size, seq_len, d_model)
        # Necesitamos que pe tenga el mismo segundo tamaño (seq_len)
        # Si pe es (max_len, 1, d_model), podemos hacerlo (max_len, d_model) con .squeeze(1)
        # y luego se hará broadcast al batch.
        return x + self.pe[:x.size(1), :].squeeze(1) # Ajustado para batch_first=True

class PlayerPathTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=SEQUENCE_LENGTH)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src) # Ya corregido en PositionalEncoding forward
        output = self.transformer_encoder(src)
        predicted_coords = self.fc_out(output[:, -1, :])
        return predicted_coords

# --- Función para Cargar un Modelo Existente ---
def load_trained_model(model_path, input_dim, d_model, n_heads, num_encoder_layers, dropout):
    model = PlayerPathTransformer(input_dim, d_model, n_heads, num_encoder_layers, dropout).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Poner el modelo en modo evaluación (desactiva dropout, etc.)
    print(f"Modelo cargado desde: {model_path}")
    return model

# --- Funciones de Análisis de Audio ---
def get_audio_duration(file_path):
    """
    Obtiene la duración en segundos de un archivo de audio usando mutagen.
    Soporta .mp3, .wav, .flac, .ogg.
    """
    print(f"DEBUG: Intentando obtener duración para: {file_path}")
    if not os.path.exists(file_path):
        print(f"DEBUG: Error: Archivo de audio NO encontrado en la ruta: {file_path}")
        return None
    try:
        if file_path.lower().endswith('.mp3'):
            print(f"DEBUG: Intentando con mutagen.MP3 para: {file_path}")
            audio = MP3(file_path)
        elif file_path.lower().endswith('.wav'):
            print(f"DEBUG: Intentando con mutagen.WAVE para: {file_path}")
            audio = WAVE(file_path)
        elif file_path.lower().endswith('.flac'):
            print(f"DEBUG: Intentando con mutagen.FLAC para: {file_path}")
            audio = FLAC(file_path)
        elif file_path.lower().endswith('.ogg'):
            print(f"DEBUG: Intentando con mutagen.OggVorbis para: {file_path}")
            audio = OggVorbis(file_path)
        else:
            print(f"DEBUG: Advertencia: Formato de audio no compatible con mutagen para duración: {file_path}. Intentando fallback a librosa.")
            # Fallback a librosa si mutagen no lo maneja
            try:
                y, sr = librosa.load(file_path, sr=None, duration=None)
                duration_librosa = librosa.get_duration(y=y, sr=sr)
                print(f"DEBUG: Duración obtenida con librosa (fallback): {duration_librosa:.2f} segundos.")
                return duration_librosa
            except Exception as e_librosa:
                print(f"DEBUG: Error al obtener duración para {file_path} (fallback librosa): {e_librosa}")
                return None
        
        # Si llegamos aquí, mutagen debería haber cargado el audio
        duration_mutagen = audio.info.length
        print(f"DEBUG: Duración obtenida con mutagen: {duration_mutagen:.2f} segundos.")
        return duration_mutagen
        
    except ID3NoHeaderError:
        print(f"DEBUG: Advertencia: MP3 sin encabezado ID3. Intentando obtener duración directamente con librosa: {file_path}")
        try:
            y, sr = librosa.load(file_path, sr=None, duration=None)
            duration_librosa = librosa.get_duration(y=y, sr=sr)
            print(f"DEBUG: Duración obtenida con librosa (después de ID3NoHeaderError): {duration_librosa:.2f} segundos.")
            return duration_librosa
        except Exception as e:
            print(f"DEBUG: Error al obtener duración para {file_path} (fallback librosa después de ID3): {e}")
            return None
    except Exception as e:
        print(f"DEBUG: Error al leer el archivo de audio {file_path}: {e}")
        return None

def get_audio_features(y, sr):
    """Extrae y normaliza características de audio."""
    try:
        rms = librosa.feature.rms(y=y)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        
        min_len = min(len(rms), len(spectral_centroid), len(spectral_bandwidth), len(spectral_rolloff), len(zcr))
        features = np.stack([
            rms[:min_len],
            spectral_centroid[:min_len],
            spectral_bandwidth[:min_len],
            spectral_rolloff[:min_len],
            zcr[:min_len]
        ], axis=1)
        
        if features.shape[0] > 1:
            return scaler.fit_transform(features)
        return np.zeros((1, 5))
        
    except Exception as e:
        print(f"Error al extraer características de audio: {e}")
        return np.zeros((1, 5))

# --- Función para obtener la posición de la IA en un tiempo dado ---
def get_ai_position_at_time(ai_path, time_in_seconds, total_duration, fps):
    """
    Obtiene la posición XY de la IA en un tiempo dado, interpolando si es necesario.
    ai_path: Lista de [x, y] posiciones generadas por la IA.
    """
    if not ai_path:
        return [SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2] # Posición central de fallback

    # Calcula el índice correspondiente en la lista de la trayectoria
    # Si la trayectoria tiene N puntos, cubre total_duration segundos.
    # Cada punto representa 1/fps segundos.
    index_float = time_in_seconds * fps
    
    # Manejar límites
    if index_float < 0:
        return ai_path[0]
    if index_float >= len(ai_path) - 1:
        return ai_path[-1]

    # Interpolar para una mayor precisión si no coincide con un índice entero
    idx1 = int(np.floor(index_float))
    idx2 = int(np.ceil(index_float))

    if idx1 == idx2: # Si es un índice entero o el último
        return ai_path[idx1]

    # Calcular el factor de interpolación
    t = index_float - idx1
    
    pos1 = np.array(ai_path[idx1])
    pos2 = np.array(ai_path[idx2])
    
    interpolated_pos = (1 - t) * pos1 + t * pos2
    return interpolated_pos.tolist()


# --- Función de Generación de Obstáculos (¡AHORA MÁS REACTIVA Y ERRÁTICA!) ---
def generate_reactive_obstacle(event_time, music_features_at_time, ai_path_unnormalized, total_duration_seconds, fps):
    """
    Genera un obstáculo con propiedades basadas en características musicales Y la posición de la IA.
    Ahora apunta desde cualquier lado de la pantalla hacia la posición de la IA.
    """
    if music_features_at_time is None or music_features_at_time.size == 0:
        music_features_at_time = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) # Valores medios por defecto

    intensity = music_features_at_time[0] # RMS (Energía)
    brightness = music_features_at_time[1] # Spectral Centroid (Brillo)
    richness = music_features_at_time[2] # Spectral Bandwidth (Riqueza armónica)
    rolloff = music_features_at_time[3] # Spectral Rolloff (Distribución de energía)
    percussiveness = music_features_at_time[4] # ZCR (Percusividad)

    # Obtener la posición de la IA en este momento del evento (coordenadas de pantalla)
    ai_current_pos = get_ai_position_at_time(ai_path_unnormalized, event_time, total_duration_seconds, fps)
    
    # --- Calcular un "Score de Animación/Alegría" (Liveliness Score) ---
    # Combina las características que indican qué tan "animada" o "fuerte" es la música.
    # Se usan pesos para dar más importancia a la intensidad y percusividad.
    liveliness_score = (intensity * 0.4 + percussiveness * 0.4 + brightness * 0.2 + richness * 0.1) 
    liveliness_score = np.clip(liveliness_score, 0.0, 1.0) # Asegurar que esté entre 0 y 1

    # --- Lógica de Tipo de Obstáculo (Más variada con mayor liveliness) ---
    if liveliness_score > 0.75: # Si la música es muy animada
        shape_type = random.choice(["square", "triangle", "circle"]) # Más variedad en formas "sólidas"
    elif intensity > 0.6 and percussiveness > 0.5:
        shape_type = random.choice(["square", "triangle"])
    elif brightness > 0.7:
        shape_type = "circle"
    elif rolloff < 0.4:
        shape_type = "line"
    else:
        shape_type = random.choice(OBSTACLE_TYPES)

    # --- Tamaño del Obstáculo (Más grande con mayor animación) ---
    base_size = 50.0
    extra_size = liveliness_score * 120.0 # Más grande, hasta 170
    size_val = base_size + extra_size
    size_val = np.clip(size_val, 50.0, 200.0) # Asegura un rango razonable

    if shape_type == "line":
        # Las líneas pueden ser más largas y gruesas en canciones animadas
        line_length = size_val * (1.5 + liveliness_score * 1.0) # Más largas con animación
        line_thickness = 15.0 + (liveliness_score * 10.0) # Más gruesas con animación
        size = [line_length, line_thickness] if random.random() > 0.5 else [line_thickness, line_length]
        size[0] = np.clip(size[0], 50.0, 500.0) # Asegurar longitud razonable
        size[1] = np.clip(size[1], 10.0, 50.0) # Asegurar grosor razonable
    else:
        size = [size_val, size_val]

    # --- Color del Obstáculo (¡Fijo a Rosa, RGB 0-1!) ---
    color = [218/255, 0.0/255, 85/255, 1.0] # Rosa vibrante: RGB(218, 0, 85) normalizado a 0-1

    # --- DIRECCIÓN Y POSICIÓN DE SPAWN: ATACAR DESDE CUALQUIER LADO APUNTANDO A LA IA ---
    spawn_x, spawn_y = 0.0, 0.0
    
    # Define los márgenes fuera de la pantalla
    margin_x = SCREEN_WIDTH * 0.15 # 15% del ancho fuera de pantalla
    margin_y = SCREEN_HEIGHT * 0.15 # 15% del alto fuera de pantalla

    # Elección aleatoria de un borde de spawn
    spawn_edge = random.choice(["top", "bottom", "left", "right"])

    if spawn_edge == "top":
        spawn_x = random.uniform(-margin_x, SCREEN_WIDTH + margin_x)
        spawn_y = -random.uniform(50, 150) # Asegurarse de que esté fuera
    elif spawn_edge == "bottom":
        spawn_x = random.uniform(-margin_x, SCREEN_WIDTH + margin_x)
        spawn_y = SCREEN_HEIGHT + random.uniform(50, 150)
    elif spawn_edge == "left":
        spawn_x = -random.uniform(50, 150)
        spawn_y = random.uniform(-margin_y, SCREEN_HEIGHT + margin_y)
    elif spawn_edge == "right":
        spawn_x = SCREEN_WIDTH + random.uniform(50, 150)
        spawn_y = random.uniform(-margin_y, SCREEN_HEIGHT + margin_y)

    # Vector desde el punto de spawn hacia la posición de la IA
    direction_vector_to_ai = np.array([ai_current_pos[0] - spawn_x, ai_current_pos[1] - spawn_y])
    
    # Normalizar el vector para obtener solo la dirección
    norm_to_ai = np.linalg.norm(direction_vector_to_ai)
    if norm_to_ai != 0:
        base_direction = (direction_vector_to_ai / norm_to_ai)
    else:
        # Fallback si el vector es cero (raro, pero posible si spawn_x/y es igual a ai_current_pos)
        base_direction = np.array([-1.0, 0.0]) 

    # Añadir un factor de "errancia" o "dispersión" basado en liveliness_score
    # Cuanto más animada la música, más impredecible puede ser la trayectoria.
    angle_rad = np.arctan2(base_direction[1], base_direction[0])
    
    # La desviación máxima es mayor con mayor liveliness_score (hasta 45 grados a cada lado para muy animado)
    max_angle_deviation_deg = liveliness_score * 45.0 
    angle_deviation_rad = random.uniform(
        -np.deg2rad(max_angle_deviation_deg),
        np.deg2rad(max_angle_deviation_deg)
    )
    angle_rad += angle_deviation_rad # Aplica la desviación

    # Vuelve a convertir el ángulo a un vector de dirección normalizado
    direction = [np.cos(angle_rad), np.sin(angle_rad)]
    
    # Asegurarse de que la dirección no sea un vector nulo por si acaso
    norm_final = np.linalg.norm(direction)
    if norm_final != 0:
        direction = (np.array(direction) / norm_final).tolist()
    else:
        direction = [-1.0, 0.0] # Fallback a ir a la izquierda

    # --- Velocidad del Obstáculo (Más rápido con mayor animación) ---
    base_speed = 200.0 # Aumentado ligeramente para mayor dinamismo
    extra_speed = liveliness_score * 500.0 # Hasta 700 de velocidad (200 + 500)
    speed = base_speed + extra_speed
    speed = np.clip(speed, 150.0, 800.0) # Asegurar límites razonables y permitir más velocidad

    # --- Offset de Posición de Spawning (Se usa directamente como la posición inicial del obstáculo) ---
    # Ya no es un "offset", sino la posición inicial calculada.
    # El "spawn_pos_offset" en el JSON puede confundir si no es un offset.
    # Podríamos llamarlo `initial_position` en el JSON para mayor claridad en Godot.
    initial_position = [spawn_x, spawn_y]

    return {
        "time": float(event_time),
        "type": shape_type,
        "size": size,
        "color": color,
        "direction": direction, # Dirección normalizada del movimiento del obstáculo
        "speed": speed,
        "initial_position": initial_position # La posición inicial donde aparece el obstáculo
    }

def generate_checkpoint(time, index):
    """Genera un evento de checkpoint."""
    return {
        "time": float(time),
        "type": "checkpoint",
        "message": f"Checkpoint {index + 1} Reached!"
    }


# --- 2. Generación de Trayectoria por la IA ---
def generate_ai_path(model, start_position_unnormalized, total_duration_seconds, steps_per_second):
    """
    Genera una trayectoria del jugador usando el modelo Transformer.
    """
    model.eval() # Asegurarse de que el modelo esté en modo evaluación
    
    current_position_normalized = np.array([
        start_position_unnormalized[0] / SCREEN_WIDTH,
        start_position_unnormalized[1] / SCREEN_HEIGHT
    ], dtype=np.float32)

    generated_path = []
    generated_path.append(start_position_unnormalized)

    # Inicializar la secuencia de entrada con la posición de inicio repetida
    input_sequence = np.tile(current_position_normalized, (SEQUENCE_LENGTH, 1))

    num_steps = int(total_duration_seconds * steps_per_second)
    print(f"Generando trayectoria para {total_duration_seconds:.2f} segundos ({num_steps} pasos)...")

    with torch.no_grad():
        for i in range(num_steps):
            # Crear tensor de entrada y mover al dispositivo
            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Realizar predicción
            predicted_next_point_normalized = model(input_tensor).squeeze(0).cpu().numpy()

            # Convertir de vuelta a coordenadas de pantalla (desnormalizar)
            predicted_next_point_unnormalized = predicted_next_point_normalized * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
            
            # Asegurarse de que la posición se mantenga dentro de los límites de la pantalla
            predicted_next_point_unnormalized[0] = np.clip(predicted_next_point_unnormalized[0], 0, SCREEN_WIDTH)
            predicted_next_point_unnormalized[1] = np.clip(predicted_next_point_unnormalized[1], 0, SCREEN_HEIGHT)

            generated_path.append(predicted_next_point_unnormalized.tolist())

            # Actualizar la secuencia de entrada para la próxima predicción
            # Mover todos los elementos un paso hacia atrás y añadir la nueva predicción al final
            input_sequence = np.roll(input_sequence, -1, axis=0)
            input_sequence[-1] = predicted_next_point_normalized

            if (i + 1) % (steps_per_second * 5) == 0: # Imprimir cada 5 segundos de juego
                print(f"Paso {i+1}/{num_steps} completado. Última pred.: [{predicted_next_point_unnormalized[0]:.2f}, {predicted_next_point_unnormalized[1]:.2f}]")
    
    print(f"Generación de trayectoria de IA completada. Total de puntos: {len(generated_path)}")
    return generated_path

# --- 3. Guardar la Trayectoria Generada y Obstáculos en JSON ---
def save_ai_level_to_json(ai_path, music_name, output_dir, obstacle_events_to_include, music_path_in_level_json): 
    file_name = f"{music_name}_level.json" 
    file_path = os.path.join(output_dir, file_name)

    data_to_save = {
        "music_name": music_name,
        "music_path": music_path_in_level_json,
        "status": "ai_generated_reactive", # Nuevo estado para indicar reactividad y que se generó con IA
        "total_duration_recorded": len(ai_path) / 60.0, # Aproximación, depende de FPS
        "player_path": ai_path,
        "obstacle_events": obstacle_events_to_include 
    }

    print(f"\nDEPURACIÓN: Guardando el JSON final en: {file_path}")
    print(f"DEPURACIÓN: El JSON final ¿contiene 'obstacle_events'?: {'obstacle_events' in data_to_save}")
    if 'obstacle_events' in data_to_save:
        print(f"DEPURACIÓN: Cantidad de obstáculos en el JSON final: {len(data_to_save['obstacle_events'])}")

    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    print(f"Nivel de IA con trayectoria y obstáculos reactivos guardados exitosamente en: {file_path}")
    print("¡IMPORTANTE! Si ya existe un archivo de nivel con ese nombre, ¡será sobrescrito!")


# --- Ejecución Principal ---
if __name__ == "__main__":
    print("--- INICIANDO GENERACIÓN DE NIVEL DE IA CON OBSTÁCULOS REACTIVOS ---")
    print(f"Ruta de guardado de niveles AI (Godot): {AI_GENERATED_LEVELS_DIR}")
    print(f"Ruta de archivos de música (mutagen/librosa): {GODOT_MUSIC_DIR}\n")

    # --- 1. Seleccionar el modelo entrenado para cargar ---
    list_of_models = [os.path.join(MODEL_SAVE_DIR, f) for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.pth')]
    if not list_of_models:
        print(f"ERROR: No se encontró ningún modelo entrenado en {MODEL_SAVE_DIR}. Entrena el modelo primero.")
        exit()
    
    latest_model_path = max(list_of_models, key=os.path.getctime)
    
    input_dimension = 2 # Coordenadas (x, y)
    model = load_trained_model(latest_model_path, input_dimension, D_MODEL, N_HEADS, NUM_ENCODER_LAYERS, DROPOUT)

    # --- 2. Definir parámetros para la generación y cargar música ---
    MUSIC_TO_GENERATE_FOR = "song1" # <<<<<<<< ¡AJUSTA ESTO AL NOMBRE DE TU ARCHIVO DE MÚSICA (sin extensión)! <<<<<<<<
    
    music_file_full_path = None
    music_file_basename = None
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        temp_path = os.path.join(GODOT_MUSIC_DIR, MUSIC_TO_GENERATE_FOR + ext)
        if os.path.exists(temp_path):
            music_file_full_path = temp_path
            music_file_basename = os.path.basename(temp_path)
            break
    
    if music_file_full_path is None:
        print(f"ERROR: Archivo de música '{MUSIC_TO_GENERATE_FOR}' no encontrado en {GODOT_MUSIC_DIR} con extensiones comunes.")
        print("Por favor, verifica la ruta 'GODOT_MUSIC_DIR' y el nombre de la música.")
        exit()

    TOTAL_SONG_DURATION_SECONDS = get_audio_duration(music_file_full_path)
    
    if TOTAL_SONG_DURATION_SECONDS is None:
        print("ERROR: No se pudo obtener la duración de la canción. Asegúrate de que el archivo no esté corrupto y que 'mutagen' esté instalado correctamente.")
        exit()
    else:
        print(f"Duración de la canción '{MUSIC_TO_GENERATE_FOR}': {TOTAL_SONG_DURATION_SECONDS:.2f} segundos.")

    AI_START_POSITION = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2) 
    FPS = 60 # Fotogramas por segundo, usado para interpolar la trayectoria de la IA

    # --- 3. Generar la trayectoria de la IA ---
    ai_generated_path_unnormalized = generate_ai_path(
        model, 
        AI_START_POSITION, 
        TOTAL_SONG_DURATION_SECONDS, 
        FPS
    )

    # --- 4. ANALIZAR AUDIO Y GENERAR OBSTÁCULOS REACTIVOS BASADOS EN LA TRAYECTORIA DE LA IA ---
    print("\n--- Iniciando generación de obstáculos reactivos ---")
    
    # Cargar el audio para análisis con librosa
    try:
        y_audio, sr_audio = librosa.load(music_file_full_path, sr=None) # sr=None para mantener el sample rate original
    except Exception as e:
        print(f"ERROR: No se pudo cargar el audio para análisis con librosa: {e}")
        exit()

    # Extraer características musicales
    music_features_all_frames = get_audio_features(y_audio, sr_audio)
    
    # Detección de beats (para los momentos en que se generan los obstáculos)
    tempo, beat_frames = librosa.beat.beat_track(y=y_audio, sr=sr_audio)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr_audio)
    
    # Filtrar beats muy cercanos (si se desea)
    # Valor más bajo = más obstáculos. Ajusta aquí si quieres más o menos.
    min_beat_interval = 0.25 # Segundos. Reducido de 0.35 para más densidad.
    filtered_beats = []
    if beat_times.size > 0:
        filtered_beats.append(beat_times[0])
        for t in beat_times[1:]:
            if t - filtered_beats[-1] >= min_beat_interval:
                filtered_beats.append(t)
    print(f"DEBUG: {len(beat_times)} beats detectados inicialmente.")
    print(f"DEBUG: {len(filtered_beats)} beats después de filtrar con intervalo de {min_beat_interval}s.")
    
    # Generar checkpoints (opcional, distribuidos a lo largo de la canción)
    checkpoint_times = [TOTAL_SONG_DURATION_SECONDS * (i+1)/4 for i in range(3)]
    
    # Combinar y ordenar todos los eventos (beats y checkpoints)
    all_level_event_times = sorted(list(set(filtered_beats + checkpoint_times)))
    
    # Lista para almacenar los eventos de obstáculos y checkpoints
    final_obstacle_events = []
    
    for time_point in all_level_event_times:
        if time_point in checkpoint_times:
            final_obstacle_events.append(generate_checkpoint(time_point, checkpoint_times.index(time_point)))
        else:
            # Obtener características musicales para este punto en el tiempo
            if len(music_features_all_frames) > 0:
                audio_duration_for_features = librosa.get_duration(y=y_audio, sr=sr_audio)
                feature_idx = min(int((time_point / audio_duration_for_features) * len(music_features_all_frames)), len(music_features_all_frames) - 1)
                features_for_time = music_features_all_frames[feature_idx]
            else:
                features_for_time = None

            # ¡Generar obstáculo REACTIVO usando la trayectoria de la IA!
            final_obstacle_events.append(generate_reactive_obstacle(
                time_point, 
                features_for_time, 
                ai_generated_path_unnormalized, 
                TOTAL_SONG_DURATION_SECONDS, 
                FPS
            ))
            
    print(f"Generación de {len(final_obstacle_events)} obstáculos reactivos completada.")

    # --- 5. Guardar el nivel completo (IA + Obstáculos Reactivos) ---
    save_ai_level_to_json(
        ai_generated_path_unnormalized, 
        MUSIC_TO_GENERATE_FOR, 
        AI_GENERATED_LEVELS_DIR, 
        final_obstacle_events,
        music_file_basename
    )
    
    print("\n--- PROCESO DE GENERACIÓN DE NIVEL COMPLETADO ---")
    print(f"El archivo JSON FINAL de nivel (con trayectoria de IA y obstáculos reactivos) está en: {AI_GENERATED_LEVELS_DIR}{MUSIC_TO_GENERATE_FOR}_level.json")
    print(f"Asegúrate de que Godot esté leyendo los niveles de la carpeta: {AI_GENERATED_LEVELS_DIR}")