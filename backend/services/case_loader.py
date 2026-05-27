from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[2]
CASOS_DIR = BASE_DIR / "data" / "casos"

################# Despliegue básico para evaluación: Listar casos #################
def listar_casos():
    if not CASOS_DIR.exists():
        return []

    casos = []
    for archivo in sorted(CASOS_DIR.glob("*.json")):
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                data = json.load(f)
                casos.append({
                    "id": data.get("id", archivo.stem),
                    "titulo": data.get("titulo", archivo.stem),
                })
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")
    return casos
###################################################################################

def cargar_caso(caso_id: str):
    ruta = CASOS_DIR / f"{caso_id}.json"

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    with open(ruta, "r", encoding="utf-8") as archivo:
        return json.load(archivo)