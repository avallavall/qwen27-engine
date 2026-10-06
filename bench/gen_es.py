# Spanish model outputs for the draft-vocabulary ranking and its test: 40 Spanish requests to a
# running server; each raw output (reasoning + answer) goes to bench\out\es_gen\NN.txt.
# Usage: $env:BENCH_KEY = "<key>"; .venv\Scripts\python.exe bench\gen_es.py
import json, os, sys, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8081")
KEY = os.environ.get("BENCH_KEY", "none")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "es_gen")
os.makedirs(OUT, exist_ok=True)
PROMPTS = [
    "Explícame qué es una tabla hash y cuándo conviene usarla.",
    "Escribe un correo formal para pedir un presupuesto de mantenimiento de una bomba industrial.",
    "Resume en dos párrafos las causas de la Revolución Francesa.",
    "¿Qué diferencias hay entre un proceso y un hilo en un sistema operativo?",
    "Dame una receta de paella valenciana con los pasos numerados.",
    "Explica cómo funciona la fotosíntesis a un niño de diez años.",
    "Escribe una función en Python que valide un DNI español y explica el código en español.",
    "¿Cuáles son las ventajas y desventajas del teletrabajo para una pyme?",
    "Redacta un plan de entrenamiento de tres semanas para correr 10 km.",
    "Explica qué es la inflación y cómo afecta a los ahorros.",
    "Describe el ciclo del agua con detalle.",
    "Haz una lista de comprobación para preparar una auditoría de seguridad informática.",
    "¿Qué es Git y para qué sirven las ramas? Pon ejemplos de comandos.",
    "Escribe un cuento corto sobre un faro en la costa gallega.",
    "Explica la diferencia entre SQL y NoSQL con ejemplos de uso.",
    "¿Cómo se calcula el interés compuesto? Pon un ejemplo con números.",
    "Redacta las instrucciones de uso de una cafetera italiana.",
    "Explica qué es una red neuronal y cómo aprende.",
    "Escribe un acta breve de una reunión de equipo sobre el retraso de un proyecto.",
    "¿Qué consejos darías a alguien que empieza a programar en JavaScript?",
    "Describe la arquitectura de una aplicación web con frontend, backend y base de datos.",
    "Explica las leyes de Newton con ejemplos de la vida diaria.",
    "Escribe una carta de presentación para un puesto de ingeniero de datos.",
    "¿Qué es Docker y qué problema resuelve? Explica con un Dockerfile de ejemplo.",
    "Resume la historia de la imprenta y su impacto en Europa.",
    "Explica cómo hacer una copia de seguridad en Windows paso a paso.",
    "¿Qué es el cambio climático y qué medidas se pueden tomar en casa?",
    "Escribe una función en C que invierta una cadena y explica los punteros que usa.",
    "Explica qué es una API REST y los verbos HTTP principales.",
    "Redacta una reseña de un restaurante imaginario en Barcelona.",
    "¿Cómo funciona un motor eléctrico? Explícalo sin fórmulas.",
    "Explica la diferencia entre machine learning supervisado y no supervisado.",
    "Escribe un poema de ocho versos sobre el otoño.",
    "Describe cómo preparar una entrevista de trabajo técnica.",
    "Explica qué son los microservicios y cuándo no conviene usarlos.",
    "¿Qué es el método científico? Pon un ejemplo de experimento.",
    "Escribe las bases de un concurso de fotografía para un ayuntamiento.",
    "Explica cómo se depura un programa que falla solo a veces.",
    "Resume las ventajas de la energía solar frente a la eólica.",
    "Explica cómo organizar un proyecto de software con Scrum.",
]
for i, p in enumerate(PROMPTS):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 900}
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    m = d["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + "\n\n" + (m.get("content") or "")
    with open(os.path.join(OUT, f"{i:02d}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(i, d["usage"]["completion_tokens"], flush=True)
