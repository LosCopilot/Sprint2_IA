import librosa
import librosa.display
import numpy as np
import json
import os
import random
from sklearn.preprocessing import MinMaxScaler
# Importar mutagen para obtener la duración del audio, si aún no lo estás usando en analyze_audio
# from mutagen.mp3 import MP3
# from mutagen.wave import WAVE
# from mutagen.flac import FLAC
# from mutagen.oggvorbis import OggVorbis
# from mutagen.id3 import ID3NoHeaderError

# --- Configuración del Generador de Niveles ---
OBSTACLE_TYPES = ["square", "triangle", "circle", "line"]
NUM_OBSTACLE_TYPES = len(OBSTACLE_TYPES)

# Normalizador para las características de la música (lo mantienes)
scaler = MinMaxScaler()

# Dimensiones de la pantalla de Godot (asumidas, ajusta si es necesario)
SCREEN_WIDTH = 1280.0
SCREEN_HEIGHT = 720.0

# --- MEJORA: Parámetro para la sensibilidad del beat y densidad de obstáculos ---
# Reducir este valor permitirá que beats más cercanos entre sí generen obstáculos.
# Un valor de 0.1 significa que los beats deben estar al menos a 0.1 segundos de distancia.
# Un valor de 0.0 permite que todos los beats detectados generen obstáculos (máxima sensibilidad).
# Puedes experimentar con valores como 0.1, 0.2, 0.05, etc.
BEAT_DETECTION_SENSITIVITY = 0.1 # Valor ajustado para más sensibilidad

def get_audio_features(y, sr):
    """Extrae y normaliza características de audio."""
    try:
        # Energía del audio (RMS)
        rms = librosa.feature.rms(y=y)[0]
        # Centroid espectral (brillo percibido)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        # Ancho de banda espectral (riqueza armónica)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        # Roll-off espectral (indica la forma del espectro, si los agudos son fuertes o no)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        # Tasa de cruces por cero (Zero Crossing Rate - ZCR), indica qué tan "ruidoso" o percusivo es el sonido
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        
        # Asegurar longitudes iguales para todas las características
        min_len = min(len(rms), len(spectral_centroid), len(spectral_bandwidth), len(spectral_rolloff), len(zcr))
        features = np.stack([
            rms[:min_len],
            spectral_centroid[:min_len],
            spectral_bandwidth[:min_len],
            spectral_rolloff[:min_len],
            zcr[:min_len]
        ], axis=1)
        
        if features.shape[0] > 1:
            # Normalizar las características para que estén entre 0 y 1
            return scaler.fit_transform(features)
        return np.zeros((1, 5)) # Fallback seguro si no hay suficientes datos, 5 características
        
    except Exception as e:
        print(f"Error al extraer características de audio: {e}")
        return np.zeros((1, 5))

def generate_obstacle(event_time, music_features_at_time):
    """
    Genera un obstáculo con propiedades basadas en las características musicales.
    'music_features_at_time' es un array 1D de características normalizadas para ese instante.
    """
    # Valor por defecto si no hay características de música (ej. inicio/fin de canción)
    if music_features_at_time is None or music_features_at_time.size == 0:
        music_features_at_time = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) # Valores medios

    # Desempaquetar las características normalizadas (entre 0 y 1)
    intensity = music_features_at_time[0] # RMS
    brightness = music_features_at_time[1] # Spectral Centroid
    richness = music_features_at_time[2] # Spectral Bandwidth
    rolloff = music_features_at_time[3] # Spectral Rolloff
    percussiveness = music_features_at_time[4] # ZCR

    # --- Lógica de Tipo de Obstáculo (Más variada) ---
    # Combina características para decidir el tipo de obstáculo
    if intensity > 0.7 and percussiveness > 0.6:
        # Alta intensidad y percusividad: Obstáculos sólidos y agresivos
        shape_type = random.choice(["square", "triangle"]) 
    elif brightness > 0.7 and richness > 0.5:
        # Brillante y con riqueza armónica: Obstáculos más "suaves" o "fluidos"
        shape_type = "circle"
    elif rolloff < 0.4:
        # Bajo rolloff (más graves, menos agudos): Puede ser una línea que bloquea
        shape_type = "line"
    else:
        # Cualquier otro caso: Elige al azar o por defecto
        shape_type = random.choice(OBSTACLE_TYPES)

    # --- Tamaño del Obstáculo (Depende de la intensidad) ---
    # Si la intensidad es baja (0), tamaño mínimo (50). Si es alta (1), tamaño máximo (150).
    size_val = 50.0 + (intensity * 100.0) 
    if shape_type == "line":
        # Las líneas suelen ser más largas horizontal o verticalmente
        size = [size_val * 2.0, 15.0] if random.random() > 0.5 else [15.0, size_val * 2.0]
    else:
        size = [size_val, size_val]

    # --- Color del Obstáculo (¡Fijo a Rosa!) ---
    # Valores RGB para un color rosa (ej. (255, 191, 201) normalizado a 0-1)
    color = [0.99, 0.13, 0.41, 1.0] # Rosa

    # --- Dirección del Obstáculo (Simulando "persecución" o apunte al centro) ---
    # Esto asume que Godot spawnea los obstáculos en los bordes de la pantalla.
    # El objetivo es que se muevan hacia el centro.
    
    target_x = SCREEN_WIDTH / 2 
    target_y = SCREEN_HEIGHT * 0.6 

    dir_x = -1.0 # Por defecto, de derecha a izquierda

    if brightness > 0.7 or percussiveness > 0.7:
        if random.random() < 0.5: 
            dir_y = (target_y - SCREEN_HEIGHT) / SCREEN_HEIGHT 
        else:
            dir_y = (target_y - 0) / SCREEN_HEIGHT 
        
        dir_x = random.uniform(-1.0, -0.7) 

    else:
        dir_y = random.uniform(-0.1, 0.1) 
        dir_x = -1.0 

    direction_vector = np.array([dir_x, dir_y])
    norm = np.linalg.norm(direction_vector)
    if norm != 0:
        direction = (direction_vector / norm).tolist()
    else:
        direction = [-1.0, 0.0] # Fallback por si acaso

    # --- Velocidad del Obstáculo (Depende de la intensidad y percusividad) ---
    speed = 150.0 + (intensity * 200.0) + (percussiveness * 100.0) 
    speed = np.clip(speed, 100.0, 500.0) 

    # --- Offset de Posición de Spawning (para agrupar o desviar) ---
    spawn_pos_offset = [0.0, 0.0]
    if random.random() < 0.2: 
        spawn_pos_offset[1] = random.uniform(-100.0, 100.0) 

    return {
        "time": float(event_time),
        "type": shape_type,
        "size": size,
        "color": color,
        "direction": direction,
        "speed": speed,
        "spawn_pos_offset": spawn_pos_offset
    }

def generate_checkpoint(time, index):
    """Genera un evento de checkpoint."""
    return {
        "time": float(time),
        "type": "checkpoint",
        "message": f"Checkpoint {index + 1} Reached!"
    }

def analyze_audio(audio_path, output_dir="levels"):
    """
    Analiza audio y genera eventos de juego.
    Ajustado para mayor sensibilidad de beats.
    """
    try:
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Detección de beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # MEJORA: Filtrar beats con la nueva sensibilidad.
        # Un BEAT_DETECTION_SENSITIVITY más bajo significa que se filtran MENOS beats,
        # resultando en MÁS obstáculos.
        filtered_beats = []
        if beat_times.size > 0:
            filtered_beats.append(beat_times[0])
            for t in beat_times[1:]:
                if t - filtered_beats[-1] >= BEAT_DETECTION_SENSITIVITY:
                    filtered_beats.append(t)
        
        print(f"DEBUG: {len(beat_times)} beats detectados inicialmente.")
        print(f"DEBUG: {len(filtered_beats)} beats después de filtrar con intervalo de {BEAT_DETECTION_SENSITIVITY}s.")

        # Generar checkpoints distribuidos
        # Genera 3 checkpoints en 1/4, 1/2 y 3/4 de la canción
        checkpoint_times = [duration * (i+1)/4 for i in range(3)]
        
        # Combinar y ordenar todos los eventos (beats y checkpoints)
        all_events = sorted(list(set(filtered_beats + checkpoint_times))) 

        # Extraer características musicales en el tiempo
        music_features = get_audio_features(y, sr)
        
        # Generar eventos de obstáculos y checkpoints
        obstacle_events = []
        for time_point in all_events: 
            if time_point in checkpoint_times:
                obstacle_events.append(generate_checkpoint(time_point, checkpoint_times.index(time_point)))
            else:
                # Calcular el índice de la característica más cercana en el tiempo
                if len(music_features) > 0:
                    feature_idx = min(int((time_point / duration) * len(music_features)), len(music_features) - 1)
                    features_for_time = music_features[feature_idx]
                else:
                    features_for_time = None 

                # Aquí se llama a la función generate_obstacle (NO generate_reactive_obstacle)
                # ya que generate_level no tiene la trayectoria de la IA
                obstacle_events.append(generate_obstacle(
                    time_point, features_for_time
                ))
        
        # Guardar archivo JSON
        os.makedirs(output_dir, exist_ok=True)
        # Asegúrate de que esta ruta sea la que espera generate_ai_path.py para cargar los obstáculos
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_level.json")
        
        with open(output_path, 'w') as f:
            json.dump({
                "music_path": os.path.basename(audio_path),
                "obstacle_events": sorted(obstacle_events, key=lambda x: x["time"]) # Asegurarse de ordenar por tiempo
            }, f, indent=4)
        
        print(f"Nivel generado exitosamente en: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error al procesar audio y generar nivel: {e}")
        import traceback
        traceback.print_exc() 
        return None

if __name__ == "__main__":
    audio_file = "song1.wav" # <<<<<<< ¡AJUSTA EL NOMBRE DE TU MÚSICA AQUÍ!
    
    # Ruta de salida para los JSON de obstáculos.
    # Si Godot espera este archivo en 'DEFINITIVA VERSION/Levels',
    # DEBES CAMBIAR ESTA LÍNEA O LA CONFIGURACIÓN DE TU generate_ai_path.py
    # Para que sea consistente con la ruta de tu generate_ai_path.py,
    # debería ser la misma que OBSTACLE_LEVEL_SOURCE_DIR en ese script.
    
    # MUY IMPORTANTE: ESTA RUTA DEBE COINCIDIR EXACTAMENTE CON LA RUTA DE CARGA EN generate_ai_path.py
    # Si generate_ai_path.py espera los obstáculos en "C:\Users\TNTKing\Documents\DEFINITIVA VERSION\version4\Levels\"
    # entonces output_dir debe ser esa ruta.
    # Basado en tu comentario previo, tu generate_ai_path.py los espera en 'version4\Levels\'
    
    # Por lo tanto, aquí vamos a usar la ruta absoluta directamente:
    output_dir = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\version4\\Levels\\" 
    os.makedirs(output_dir, exist_ok=True) # Asegurarse de que el directorio exista
    
    # Ruta completa del archivo de audio
    # Asume que song1.wav está en la carpeta 'Audio' que usas en generate_ai_path.py
    # Si está en el mismo directorio que generate_level.py, ajusta.
    # Si no tienes una constante GODOT_MUSIC_DIR aquí, puedes poner la ruta completa del audio
    audio_input_path = os.path.join("C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\JSTB_AI\\Audio\\", audio_file)

    if os.path.exists(audio_input_path):
        print(f"DEBUG: Procesando archivo de audio: {audio_input_path}")
        analyze_audio(audio_input_path, output_dir)
    else:
        print(f"Error: Archivo de audio '{audio_input_path}' no encontrado.")
        print("Por favor, asegúrate de que el archivo de música esté en la ruta especificada.")