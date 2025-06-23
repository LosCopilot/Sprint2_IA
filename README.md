Grupo LosCopilot
## Configuración del Juego Godot

Sigue estos pasos para configurar y ejecutar el juego base en Godot:

1.  **Abrir el Proyecto Godot:**
    * Si ya tienes Godot Engine instalado, abre Godot y selecciona "Importar Proyecto". Navega hasta la carpeta `GodotProject/` (o `Version4/` si esa es la raíz de tu proyecto Godot) dentro de este repositorio y selecciona el archivo `project.godot`.
    * Si no tienes Godot, puedes descargar la versión correspondiente desde la página oficial y luego abrir la aplicación para importar el proyecto.

2.  **Modificar Ruta de Guardado de Datos del Jugador (CRÍTICO):**
    * Es **CRÍTICO** modificar una ruta dentro del código Godot para que el juego pueda guardar correctamente los datos de trayectoria del jugador para el entrenamiento de la IA.
    * Abre el script `GlobalData.gd`, que se encuentra en la ruta:
        `GodotProject/scripts/GlobalData.gd` (o `Version4/scripts/GlobalData.gd` si esa es la carpeta raíz de tu juego).
    * Busca la línea que define la constante `ABSOLUTE_PLAYER_DATA_DIR`:
        ```gdscript
        const ABSOLUTE_PLAYER_DATA_DIR = "C:\\Users\\TNTKing\\Documents\\DEFINITIVA VERSION\\JSTB_AI\\player_data_for_ai\\"
        ```
    * **Modifica esta constante** para que apunte a la ruta absoluta de la carpeta `player_data_for_ai/` en tu sistema.
        **Ejemplo:** `const ABSOLUTE_PLAYER_DATA_DIR = "C:/Users/TuUsuario/Descargas/Just_Shapes_AI/player_data_for_ai/"`
        *Asegúrate de usar barras inclinadas `/` o dobles barras invertidas `\\` y de que la ruta sea **absoluta** y correcta para tu sistema.*

3.  **Ejecutar el Juego:**
    * Una vez modificada la ruta, puedes ejecutar el juego desde Godot (presionando `F5`). Juega varias veces, moviendo activamente a tu personaje para generar datos de movimiento. Estos archivos se guardarán como `player_path_data_*.json` en la carpeta `player_data_for_ai/`. Sin estos datos, el entrenamiento de la IA no podrá comenzar.

## Configuración del Entorno Python (IA y Generación de Niveles)

Aquí se detalla cómo preparar tu entorno Python para entrenar el modelo de IA y generar niveles:

1.  **Crear y Activar un Entorno Virtual (Recomendado):**
    Abre tu terminal o línea de comandos y navega a la **raíz de este repositorio** (`/Just_Shapes_AI/`).
    ```bash
    python -m venv venv
    ```
    Activa el entorno virtual:
    * **Windows:** `.\venv\Scripts\activate`
    * **macOS/Linux:** `source venv/bin/activate`

2.  **Instalar Dependencias:**
    Con el entorno virtual activado, instala las librerías necesarias utilizando `pip`.

    * **Para Usuarios con GPU AMD (Windows):**
        ```bash
        pip install torch-directml numpy scikit-learn librosa mutagen
        ```
    * **Para Usuarios con GPU NVIDIA (CUDA):**
        Visita la página oficial de PyTorch ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) para obtener el comando de instalación correcto para tu versión de CUDA. Por ejemplo (para CUDA 12.1):
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        pip install numpy scikit-learn librosa mutagen
        ```
        *(Ajusta `cu121` según tu versión de CUDA, o busca la versión más reciente en la página de PyTorch)*
    * **Para Usuarios sin GPU (Solo CPU):**
        ```bash
        pip install torch numpy scikit-learn librosa mutagen
        ```
    * **Verificar Instalación:** Ejecuta el siguiente comando para confirmar que PyTorch y tu acelerador (si aplica) están configurados correctamente:
        ```bash
        python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); try: import torch_directml; print(f'DirectML available: {torch_directml.is_available()}') if torch_directml.is_available() else print('No DirectML available'); except ImportError: pass; if hasattr(torch, 'cuda'): print(f'CUDA available: {torch.cuda.is_available()}')"
        ```

3.  **Configuración de Rutas y Parámetros en los Scripts Python (ESENCIAL):**
    Es **ESENCIAL** que las rutas dentro de los scripts Python (`python_scripts/train_transformer_model.py` y `python_scripts/generate_ia_path.py`) sean correctas y apunten a las carpetas adecuadas en tu sistema.

    Abre ambos scripts con un editor de texto y **ajusta las siguientes variables a las rutas absolutas** de tu proyecto (se recomienda usar barras inclinadas `/` para mayor compatibilidad):

    * **En `python_scripts/train_transformer_model.py`:**
        * `GODOT_PLAYER_DATA_DIR = "C:/Users/TuUsuario/Descargas/Just_Shapes_AI/player_data_for_ai/"`
        * `MODEL_SAVE_DIR = "C:/Users/TuUsuario/Descargas/Just_Shapes_AI/trained_models/"`
        * Asegúrate de que `SCREEN_WIDTH` y `SCREEN_HEIGHT` coincidan con las dimensiones de tu juego en Godot.

    * **En `python_scripts/generate_ia_path.py`:**
        * `MODEL_SAVE_DIR = "C:/Users/TuUsuario/Descargas/Just_Shapes_AI/trained_models/"`
        * `AI_GENERATED_LEVELS_DIR = "C:/Users/TuUsuario/Descargas/Just_Shapes_AI/Levels/"`
        * `GODOT_MUSIC_DIR = "C:/Users/TuUsuario/Descargas/Just_Shapes_AI/Audio/"`
        * **`MUSIC_TO_GENERATE_FOR = "nombre_de_tu_cancion"`**: **CAMBIA** este valor al nombre exacto de tu archivo de música (sin la extensión, ej. `"song1"`) que quieres usar para generar el nivel. Este archivo debe estar previamente en la carpeta `Audio/`.
        * Asegúrate de que `SCREEN_WIDTH` y `SCREEN_HEIGHT` coincidan con las dimensiones de tu juego en Godot.
        * **Verifica la consistencia de parámetros:** Asegúrate de que los parámetros del modelo Transformer (`SEQUENCE_LENGTH`, `D_MODEL`, `N_HEADS`, `NUM_ENCODER_LAYERS`, `DROPOUT`) en este script **coincidan exactamente** con los valores utilizados durante el entrenamiento en `train_transformer_model.py`.

## Flujo de Trabajo: Reentrenamiento de la IA y Generación de Niveles

Sigue estos pasos en orden para reentrenar la IA y generar un nuevo nivel reactivo para tu juego:

1.  **Paso 1: Recolectar Datos de Trayectoria del Jugador (Usando Godot)**
    * Asegúrate de haber configurado Godot (ver sección "Configuración del Juego Godot").
    * Ejecuta el juego desde Godot y juega varias veces, moviendo activamente a tu personaje. Cada partida exitosa generará un archivo `player_path_data_*.json` en la carpeta `player_data_for_ai/` configurada. Cuantos más datos diversos recolectes, mejor será el entrenamiento de la IA.

2.  **Paso 2: Entrenar el Modelo Transformer**
    * Abre tu terminal o línea de comandos.
    * Navega a la carpeta `python_scripts/` dentro de tu proyecto.
    * Asegúrate de que tu entorno virtual Python esté activado (ver Sección "Configuración del Entorno Python").
    * Ejecuta el script de entrenamiento:
        ```bash
        python train_transformer_model.py
        ```
    * Este proceso puede tomar desde minutos hasta horas, dependiendo de la cantidad de datos disponibles y la potencia de tu hardware (CPU/GPU). Al finalizar, se guardará un archivo `.pth` (el modelo entrenado) con una marca de tiempo en la carpeta `trained_models/`.

3.  **Paso 3: Generar la Trayectoria de la IA y el Nivel Reactivo**
    * **Importante:** Asegúrate de haber completado el Paso 2 y tener al menos un modelo entrenado en la carpeta `trained_models/`.
    * Asegúrate de que el archivo de música que configuraste en `generate_ia_path.py` (variable `MUSIC_TO_GENERATE_FOR`) esté presente en la carpeta `Audio/`.
    * Desde la misma terminal (con el entorno virtual activado y en la carpeta `python_scripts/`), ejecuta el script de generación de nivel:
        ```bash
        python generate_ia_path.py
        ```
    * Este script cargará automáticamente el modelo entrenado más reciente, analizará la música y generará tanto la trayectoria predicha de la IA como los eventos de obstáculos reactivos basados en la música y la IA.
    * El resultado final será un archivo JSON (ej. `nombre_de_tu_cancion_level.json`) guardado en la carpeta `Levels/`.

## Cargar Niveles Generados en Godot

Una vez que el archivo JSON del nivel se ha generado en la carpeta `Levels/`, tu proyecto Godot está configurado para leer y cargar estos niveles. Simplemente inicia el juego en Godot, y debería detectar y usar el nivel generado por la IA para la canción especificada.

## Solución de Problemas Comunes

* **`FileNotFoundError` o `json.JSONDecodeError`:**
    * **Verifica las Rutas Absolutas:** Asegúrate de que todas las rutas absolutas configuradas en `GlobalData.gd` de Godot, y en `train_transformer_model.py` y `generate_ia_path.py` de Python, sean **correctas** para tu sistema.
    * **Formato de Rutas:** Usa barras inclinadas `/` o dobles barras invertidas `\\` consistentemente.
    * **Archivos Faltantes:** Confirma que existan archivos `player_path_data_*.json` en `player_data_for_ai/` antes de entrenar, y que el archivo de música especificado exista en `Audio/`.
* **`ModuleNotFoundError` (o librerías no encontradas):**
    * Asegúrate de que tu entorno virtual Python esté **activado** antes de ejecutar cualquier script.
    * Confirma que todas las dependencias se instalaron correctamente utilizando `pip install ...`.
* **Problemas de Dispositivo (GPU/CPU):**
    * Si ves advertencias o errores relacionados con CUDA o DirectML, verifica que tu instalación de PyTorch coincida con tu hardware (ver Paso 2.b en "Configuración del Entorno Python"). Asegúrate de que los drivers de tu GPU estén actualizados.
* **Advertencia "ADVERTENCIA: No se generaron suficientes secuencias" durante el entrenamiento:**
    * Esto indica que necesitas más datos de juego. Juega más partidas en Godot o considera ajustar el `OVERLAP` en `train_transformer_model.py` (aunque esto podría reducir la diversidad de secuencias). Asegúrate de que tus trayectorias de jugador sean lo suficientemente largas para el `SEQUENCE_LENGTH` configurado.
* **Modelo no encontrado al generar el nivel:**
    * Asegúrate de que el script `train_transformer_model.py` haya finalizado correctamente y haya guardado un archivo `.pth` en `trained_models/`.
* **El nivel generado no se carga en Godot:**
    * Verifica que el nombre del archivo JSON generado en `Levels/` (`nombre_de_tu_cancion_level.json`) coincida exactamente con el nombre de la canción que Godot espera para cargar el nivel.
    * Confirma que la lógica de carga de niveles en Godot apunte correctamente a la carpeta `Levels/`.

---
