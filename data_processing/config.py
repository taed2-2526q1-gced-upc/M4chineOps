import os

# Ruta base del proyecto (ajusta si es necesario)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio con los datos originales
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# Directorio donde se guardarán los datos procesados
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Crea las carpetas si no existen
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
