from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
CASOS_DIR = BASE_DIR / "data" / "casos"
KNOWLEDGE_DIR = BASE_DIR / "data" / "docs" / "knowledge"
INDICE_PATH = BASE_DIR / "data" / "casos_index.json"


SECTION_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _normalizar(texto: str) -> str:
    texto = (texto or "").strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _leer_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"El archivo {path} no contiene un objeto JSON válido.")
    return data


def _leer_texto(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _cargar_indice() -> dict[str, dict[str, Any]]:
    if not INDICE_PATH.exists():
        return {}

    try:
        with open(INDICE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, list):
        return {}

    return {
        str(item.get("case_id", "")).strip(): item
        for item in data
        if isinstance(item, dict) and str(item.get("case_id", "")).strip()
    }


def _fusionar_listas(*listas: Any) -> list[str]:
    salida: list[str] = []
    vistos: set[str] = set()

    for lista in listas:
        if not lista:
            continue

        if isinstance(lista, str):
            candidatos = [lista]
        elif isinstance(lista, list):
            candidatos = lista
        else:
            candidatos = [str(lista)]

        for item in candidatos:
            texto = str(item).strip()
            if not texto:
                continue
            clave = _normalizar(texto)
            if clave in vistos:
                continue
            vistos.add(clave)
            salida.append(texto)

    return salida


def _agregar_texto(partes: list[str], valor: Any) -> None:
    if valor is None:
        return

    if isinstance(valor, str):
        texto = valor.strip()
        if texto:
            partes.append(texto)
        return

    if isinstance(valor, list):
        for item in valor:
            if item is None:
                continue
            texto = str(item).strip()
            if texto:
                partes.append(texto)
        return

    texto = str(valor).strip()
    if texto:
        partes.append(texto)


def _extraer_secciones_md(texto_md: str) -> tuple[str, dict[str, list[str]]]:
    titulo_documento = ""
    secciones: dict[str, list[str]] = {}

    seccion_actual: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, seccion_actual
        if not seccion_actual:
            buffer = []
            return

        lineas = [linea.strip() for linea in buffer if linea.strip()]
        if not lineas:
            buffer = []
            return

        secciones.setdefault(seccion_actual, []).extend(lineas)
        buffer = []

    for linea in texto_md.splitlines():
        texto = linea.strip()

        if not texto:
            if seccion_actual:
                buffer.append("")
            continue

        match = SECTION_RE.match(texto)
        if match:
            flush()
            nivel = len(match.group(1))
            heading = match.group(2).strip()

            if nivel == 1:
                titulo_documento = heading
                seccion_actual = None
                buffer = []
            else:
                seccion_actual = _normalizar(heading)
                buffer = []
            continue

        if seccion_actual:
            buffer.append(texto)

    flush()
    return titulo_documento, secciones


def _texto_parrafo(lines: list[str]) -> str:
    partes = [linea.strip() for linea in lines if linea and linea.strip()]
    return " ".join(partes).strip()


def _extraer_palabras_clave(texto: str, limite: int = 40) -> list[str]:
    texto_n = _normalizar(texto)
    if not texto_n:
        return []

    tokens = [
        t for t in texto_n.split()
        if len(t) >= 5 and t not in {
            "redacta", "respuesta", "orden", "logico", "lógico", "incluye", "evita", "usa",
            "premisas", "conclusion", "conclusión", "sustento", "juridico", "jurídico",
            "argumentativos", "argumentativo", "contenido", "claros", "claro", "caso",
            "texto", "argumenta", "argumentar",
        }
    ]

    frases: list[str] = []
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            frase = " ".join(tokens[i : i + n]).strip()
            if frase and frase not in frases:
                frases.append(frase)

    candidatos = list(dict.fromkeys(tokens + frases))
    return candidatos[:limite]


def _construir_perfil_juridico(
    caso_id: str,
    caso_data: dict[str, Any],
    perfil_indice: dict[str, Any] | None,
) -> dict[str, Any]:
    md_path = KNOWLEDGE_DIR / f"{caso_id}.md"
    md_texto = _leer_texto(md_path)
    titulo_md, secciones_md = _extraer_secciones_md(md_texto)

    hechos_md = _texto_parrafo(secciones_md.get("hechos relevantes", []))
    soporte_constitucional = _texto_parrafo(secciones_md.get("soporte constitucional sugerido", []))
    criterio_aplicacion = _texto_parrafo(secciones_md.get("criterio de aplicación", []))
    objetivo = _texto_parrafo(secciones_md.get("objetivo", []))
    nota_indice = _texto_parrafo(secciones_md.get("nota para el índice vectorial", []))

    # Enriquecimiento desde el índice auxiliar generado automáticamente.
    perfil_indice = perfil_indice if isinstance(perfil_indice, dict) else {}

    # Base conceptual del texto del markdown.
    texto_base = " ".join(
        part for part in [
            titulo_md,
            hechos_md,
            soporte_constitucional,
            criterio_aplicacion,
            objetivo,
            nota_indice,
            _texto_parrafo(
                secciones_md.get("caso asociado", [])
                + secciones_md.get("caja", [])
            ),
        ]
        if part
    ).strip()

    conceptos_esperados = _fusionar_listas(
        perfil_indice.get("conceptos_esperados", []),
        perfil_indice.get("palabras_clave", []),
        _extraer_palabras_clave(texto_base),
    )

    normas_clave = _fusionar_listas(
        perfil_indice.get("normas_clave", []),
        [soporte_constitucional] if soporte_constitucional else [],
        _extraer_palabras_clave(soporte_constitucional, limite=15),
    )

    hechos_clave = _fusionar_listas(
        perfil_indice.get("hechos_clave", []),
        [hechos_md] if hechos_md else [],
    )

    tesis_esperada = _fusionar_listas(
        perfil_indice.get("tesis_esperada", []),
        [criterio_aplicacion] if criterio_aplicacion else [],
    )

    palabras_clave = _fusionar_listas(
        perfil_indice.get("palabras_clave", []),
        _extraer_palabras_clave(" ".join([hechos_md, soporte_constitucional, criterio_aplicacion]), limite=30),
    )

    descripcion = perfil_indice.get("descripcion") or objetivo or titulo_md or caso_data.get("titulo", "")

    perfil_juridico = {
        "id": caso_id,
        "source": str(md_path.relative_to(BASE_DIR)) if md_path.exists() else "",
        "titulo_documento": titulo_md or caso_data.get("titulo", ""),
        "descripcion": descripcion,
        "hechos_clave": hechos_clave,
        "normas_clave": normas_clave,
        "conceptos_esperados": conceptos_esperados,
        "tesis_esperada": tesis_esperada,
        "palabras_clave": palabras_clave,
        "texto_completo": md_texto,
        "secciones": secciones_md,
    }

    # Si el JSON ya traía perfil_juridico, lo preservamos y enriquecemos.
    perfil_json = caso_data.get("perfil_juridico")
    if isinstance(perfil_json, dict):
        perfil_juridico["hechos_clave"] = _fusionar_listas(
            perfil_json.get("hechos_clave", []),
            perfil_juridico["hechos_clave"],
        )
        perfil_juridico["normas_clave"] = _fusionar_listas(
            perfil_json.get("normas_clave", []),
            perfil_juridico["normas_clave"],
        )
        perfil_juridico["conceptos_esperados"] = _fusionar_listas(
            perfil_json.get("conceptos_esperados", []),
            perfil_juridico["conceptos_esperados"],
        )
        perfil_juridico["tesis_esperada"] = _fusionar_listas(
            perfil_json.get("tesis_esperada", []),
            perfil_juridico["tesis_esperada"],
        )
        perfil_juridico["palabras_clave"] = _fusionar_listas(
            perfil_json.get("palabras_clave", []),
            perfil_juridico["palabras_clave"],
        )
        if not perfil_juridico["descripcion"]:
            perfil_juridico["descripcion"] = perfil_json.get("descripcion", "")

    return perfil_juridico


def listar_casos() -> list[dict[str, str]]:
    if not CASOS_DIR.exists():
        return []

    casos: list[dict[str, str]] = []

    for archivo in sorted(CASOS_DIR.glob("*.json")):
        try:
            data = _leer_json(archivo)
            casos.append(
                {
                    "id": str(data.get("id", archivo.stem)),
                    "titulo": str(data.get("titulo", archivo.stem)),
                }
            )
        except Exception:
            continue

    return casos


def cargar_caso(caso_id: str) -> dict[str, Any]:
    ruta = CASOS_DIR / f"{caso_id}.json"

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    data = _leer_json(ruta)

    indice = _cargar_indice()
    perfil_indice = indice.get(caso_id, {})

    perfil_juridico = _construir_perfil_juridico(
        caso_id=caso_id,
        caso_data=data,
        perfil_indice=perfil_indice,
    )

    data["perfil_juridico"] = perfil_juridico
    data["knowledge_path"] = perfil_juridico.get("source", "")
    return data