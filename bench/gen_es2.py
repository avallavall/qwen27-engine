# More Spanish (and some Catalan) model outputs for the draft-vocabulary ranking: topic x task prompts, thinking
# off (so the whole output is in the prompt's language). Writes bench\out\es_gen2\NNN.txt. These files are used for
# the ranking only; bench\out\es_gen odd files stay the held-out test.
# Usage: $env:BENCH_KEY = "<key>"; .venv\Scripts\python.exe bench\gen_es2.py
import json, os, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8081")
KEY = os.environ.get("BENCH_KEY", "none")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "es_gen2")
os.makedirs(OUT, exist_ok=True)
TEMAS = ["la gestión de un almacén", "la seguridad en redes wifi", "el aceite de oliva", "las bases de datos relacionales",
         "la contabilidad de una pyme", "el mantenimiento de una caldera", "el turismo rural", "la programación en Rust",
         "la energía hidráulica", "los contratos de alquiler", "la nutrición deportiva", "la historia de Roma",
         "los sensores de temperatura", "la logística del transporte por carretera", "la cocina mediterránea",
         "las pruebas unitarias de software", "la gestión de equipos remotos", "la fotografía nocturna",
         "el reciclaje de plásticos", "la atención al cliente por teléfono", "el mercado inmobiliario",
         "la inteligencia artificial en la medicina", "el cultivo del tomate", "las hojas de cálculo",
         "la arquitectura modernista", "los derechos del consumidor", "la impresión 3D", "el ciclismo de montaña",
         "la planificación de un viaje a Japón", "el diseño de interfaces web"]
TAREAS = ["Escribe una explicación detallada sobre {t} para un principiante.",
          "Redacta un informe breve con recomendaciones prácticas sobre {t}.",
          "Escribe preguntas frecuentes con sus respuestas sobre {t}.",
          "Explica los errores más comunes relacionados con {t} y cómo evitarlos."]
CATALAN = ["Explica què és una base de dades i per a què serveix.", "Escriu un correu per demanar vacances a l'empresa.",
           "Resumeix la història de Barcelona en tres paràgrafs.", "Explica com funciona un motor de combustió.",
           "Dona consells per estudiar millor per als exàmens.", "Descriu com es prepara un pa amb tomàquet.",
           "Explica què és la intel·ligència artificial a una persona gran.", "Escriu un conte curt sobre un drac i un pastor.",
           "Explica com fer una còpia de seguretat de les fotos del mòbil.", "Quins avantatges té anar en bicicleta a la feina?",
           "Explica la diferència entre un virus i un bacteri.", "Escriu un informe breu sobre el consum d'aigua a casa.",
           "Explica què és Git i com es fa un commit.", "Redacta les normes d'una biblioteca municipal.",
           "Explica com es calcula el percentatge d'un descompte.", "Descriu el clima de la costa mediterrània.",
           "Escriu una carta de reclamació per un paquet perdut.", "Explica què són els microplàstics.",
           "Dona una recepta de crema catalana.", "Explica com organitzar una reunió de feina eficaç."]
prompts = [tarea.format(t=t) for t in TEMAS for tarea in TAREAS] + CATALAN
for i, p in enumerate(prompts):
    dst = os.path.join(OUT, f"{i:03d}.txt")
    if os.path.exists(dst):
        continue
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 700,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    with open(dst, "w", encoding="utf-8") as f:
        f.write(d["choices"][0]["message"].get("content") or "")
    if i % 20 == 0:
        print(i, len(prompts), flush=True)
