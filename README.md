# Cuadernos de comprobación de proyectos de IA en Google Colab

Este repositorio reúne tres cuadernos de Google Colab. Cada uno toma un proyecto de código abierto relacionado con los
modelos de lenguaje (LLM), explica qué es y comprueba si funciona de verdad. Lo que no se puede comprobar se indica
expresamente.

| Cuaderno | Proyecto que comprueba | Qué es el proyecto | Resultado en Colab |
|---|---|---|---|
| `test_LLMs_from_scratch.ipynb` | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | Código del libro *Build a Large Language Model (From Scratch)*: un GPT programado paso a paso con fines didácticos | ✅ 33 de 33 pruebas correctas |
| `ds4_dwarfstar_explicado_y_probado.ipynb` | [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) | Programa en C para ejecutar modelos gigantes (DeepSeek V4, GLM 5, Qwen3.8) en ordenadores potentes de uso personal | ✅ 24 de 24 pruebas correctas (sin poder cargar el modelo) |
| `colibri_que_es_si_funciona_y_como_probarlo.ipynb` | [JustVugg/colibri](https://github.com/JustVugg/colibri) (colibrí) | Programa en C que ejecuta modelos de expertos más grandes que la memoria del ordenador, leyendo del disco las partes que necesita | ✅ 8 promesas ciertas · ⚠️ 2 ciertas con matices · ❌ 0 falsas |

Los tres se han ejecutado en Colab en noviembre de 2026. Los resultados de este documento salen de esas ejecuciones.

---

## Cómo usarlos

1. Abre [Google Colab](https://colab.research.google.com) → **Archivo → Subir cuaderno** y elige el cuaderno.
2. Elige el tipo de entorno (**Entorno de ejecución → Cambiar tipo de entorno**):
   - `test_LLMs_from_scratch`: **GPU T4** (recomendado).
   - `ds4_dwarfstar`: **GPU T4** (solo hace falta para la parte de CUDA).
   - `colibri`: **CPU normal**; no necesita tarjeta gráfica.
3. **Entorno de ejecución → Ejecutar todo**. La última celda de cada cuaderno muestra una tabla resumen con
   ✅ / ⚠️ / ❌.

| Cuaderno | Tiempo aproximado | Descargas |
|---|---|---|
| LLMs-from-scratch | 15–20 min con T4 | ~500 MB (pesos de GPT-2) |
| ds4 | ~12 min (de ellos, ~9 compilando CUDA) | Solo el código |
| colibrí | 25–35 min | 23 GB (modelo Qwen3.6-35B-A3B) |

---

## 1. LLMs-from-scratch

### Qué es el proyecto

Es el código que acompaña al libro de Sebastian Raschka. Construye un modelo tipo GPT desde cero con PyTorch, capítulo
a capítulo: cómo se trocea el texto en tokens, el mecanismo de atención, la arquitectura GPT, el entrenamiento y el
ajuste para tareas concretas (clasificar spam, seguir instrucciones). Incluye material extra con versiones desde cero
de modelos actuales como Llama 3.2, Qwen3, Gemma 3 y OLMo 3.

### Qué hace el cuaderno

1. Clona el proyecto e instala lo que falta en Colab.
2. Comprueba las versiones de las librerías con el script oficial del proyecto.
3. Ejecuta las pruebas automáticas de los capítulos 4, 5 y 6, las mismas que usa el autor en GitHub.
4. Vuelve a ejecutar los cuadernos de los capítulos 2 y 3 y compara sus resultados con los publicados.
5. Ejecuta las pruebas del material extra: caché de atención, atención eficiente, tokenizador BPE desde cero y
   utilidades del capítulo 7.
6. **Prueba de funcionamiento real:** carga los pesos oficiales de GPT-2 (124 millones de números) en el modelo del
   libro, genera texto y mide su error sobre una frase en inglés.
7. Ejecuta las pruebas del paquete del libro y de los modelos extra (Llama 3.2, Qwen3, Gemma 3 y OLMo 3), que se
   comparan con la librería `transformers` de Hugging Face.

### Resultados en Colab (GPU T4, PyTorch 2.11)

- **33 de 33 pruebas correctas.**
- GPT-2 cargado en el código del libro escribe texto coherente en inglés:
  *"Every effort moves you as far as the hand can go until the end of your turn…"*. Su error sobre el texto de control
  es 3,80; con pesos aleatorios sería en torno a 11. Por tanto, la arquitectura está bien programada.
- Las versiones desde cero de Llama 3.2, Qwen3, Gemma 3 y OLMo 3 coinciden con `transformers`.

### Limitaciones del proyecto

- **Es material para aprender, no para producción.** Los modelos que se entrenan en el libro son pequeños, y el código
  prioriza que se entienda, no la velocidad.
- **GPT-2 de 124M es un modelo muy limitado.** Sirve para comprobar que todo funciona, pero sus respuestas son pobres
  comparadas con las de un modelo actual.
- **Entrenar un modelo grande desde cero no es viable** con este código ni en Colab: el libro entrena modelos pequeños
  y, para los grandes, carga pesos ya entrenados.
- Necesita **PyTorch**, y algunas partes extra necesitan `transformers` y descargas de Hugging Face.
- Las pruebas extensas (Llama 3, Qwen3) piden varios GB de memoria; en Colab gratuito caben, pero van justas.

### Limitaciones de la comprobación

- Las pruebas del autor comprueban que el código hace lo que el autor espera. La prueba independiente más fuerte es
  la de GPT-2, junto con las comparaciones con `transformers`.
- No se comprueban las explicaciones teóricas del libro ni los resultados de entrenamientos largos.

---

## 2. ds4 (DwarfStar)

### Qué es el proyecto

Es un programa escrito en C por Salvatore Sanfilippo (antirez, el creador de Redis) para **ejecutar modelos gigantes
en ordenadores de uso personal muy potentes**: Mac con mucha memoria, NVIDIA DGX Spark o PCs AMD Strix Halo. Está
especializado en unos pocos modelos: DeepSeek V4 Flash, V4.1 Flash y V4 PRO, GLM 5.x y Qwen3.8. Para que quepan, los
guarda muy comprimidos (unos 2 bits por número en las partes que lo toleran). Incluye:

- conversación en la terminal;
- un servidor compatible con las APIs de OpenAI y Anthropic;
- un agente de programación;
- pruebas de calidad con preguntas de exámenes conocidos;
- medición de velocidad.

### Qué hace el cuaderno

1. Explica los conceptos clave: modelos de expertos, compresión de los pesos, caché de la conversación y por qué la
   velocidad depende de la memoria.
2. Calcula cuánto ocupa DeepSeek V4 Flash según el nivel de compresión, y por qué la memoria limita la velocidad.
3. Compila el proyecto en Linux sin GPU (`make cpu`) y comprueba que los programas arrancan y fallan de forma
   controlada si no hay modelo.
4. Ejecuta 15 pruebas automáticas que no necesitan el modelo:
   - compresión Q4_K y MXFP4;
   - elección del siguiente token;
   - gestión de la conversación;
   - reparto entre varias GPUs;
   - imágenes;
   - agente;
   - comunicación por red entre equipos.
5. Valida el banco de preguntas de examen de `ds4-eval` (GPQA, AIME 2025, SuperGPQA).
6. Compila la versión para GPU NVIDIA (CUDA) para la T4 de Colab y comprueba que arranca.

### Resultados en Colab (GPU T4, 12,7 GB de memoria)

- **24 de 24 pruebas correctas**, incluida la compilación para la T4 (unos 9 minutos).
- La prueba de tiempos de espera de red, que fallaba en otros entornos por cómo redondea el tiempo el sistema
  operativo, en Colab ha pasado.

### Limitaciones del proyecto

- **Necesita mucha memoria.** El modelo más pequeño (DeepSeek V4 Flash comprimido) ocupa unos **81 GB**, y el autor
  recomienda equipos con **96–128 GB**. Qwen3.8 funciona con 64 GB en Mac.
- **Hardware concreto.** Está pensado sobre todo para Mac con chip de Apple; también funciona en NVIDIA (DGX Spark,
  Blackwell, varias GPUs Ada) y AMD Strix Halo. La versión de solo procesador es **lentísima**: existe solo para
  comprobar que las demás calculan bien.
- **Pocos modelos y solo sus propios archivos.** No carga cualquier modelo; solo los que publica el proyecto.
- **Velocidad publicada** (DeepSeek V4 Flash, contexto corto): unos 39 tokens/s en un MacBook Pro M5 Max y unos
  18 tokens/s en una DGX Spark.
- Está en **fase beta** y cambia muy rápido.

### Limitaciones de la comprobación

- **No se ha ejecutado ningún modelo**, porque no caben en Colab (81 GB frente a 12,7 GB de memoria y 15 GB de
  tarjeta gráfica). Se ha comprobado que el código compila, arranca y pasa sus pruebas, pero no la calidad ni la
  velocidad reales.
- La versión CUDA se ha compilado y arrancado, pero no ha hecho cálculos con un modelo.

---

## 3. colibrí

### Qué es el proyecto

Es un programa escrito en C que permite usar **modelos de expertos más grandes que la memoria del ordenador**. Estos
modelos solo usan una pequeña parte de sí mismos para cada token: unos pocos "expertos" de entre cientos. Colibrí deja
en memoria la parte que se usa siempre, lee del disco los expertos que hacen falta en cada momento y guarda en memoria
los que más se usan. Su caso más llamativo es **GLM-5.2 (744.000 millones de números, unos 400 GB) con unos 25 GB de
memoria**. Se puede usar desde la terminal, desde el navegador o como si fuera la API de ChatGPT o de Claude.

### Qué hace el cuaderno

Comprueba nueve promesas del proyecto:

| # | Promesa | Cómo se comprueba |
|---|---|---|
| P1 | Hecho en C, sin instalar nada más ni tarjeta gráfica | Compilación y revisión de lo que necesita para funcionar |
| P2 | Responde exactamente lo mismo que la implementación de referencia | Modelos en miniatura de GLM-5.2 y Qwen3.6 comparados token a token con `transformers` |
| P3 | La memoria disponible cambia la velocidad, no la respuesta | Repetición con 8, 2 y 1 expertos en memoria, y con el modelo de verdad |
| P4 | Se instala con un solo comando | `coli setup` |
| P5 | Funciona con modelos más grandes que la memoria | Qwen3.6-35B-A3B (23 GB) en un Colab con 12,7 GB |
| P6 | 6 tokens/s con ese modelo en un procesador de 8 núcleos | Medición en Colab (2 núcleos) |
| P7 | Se puede usar como la API de ChatGPT o de Claude | Servidor `coli serve` y preguntas con los dos formatos |
| P8 | GLM-5.2 con unos 25 GB de memoria, a 0,05–0,1 tokens/s | Cálculos con los datos del modelo y medición del disco |
| P9 | Cálculos internos exactos | Prueba del proyecto que compara sus funciones rápidas con otras de referencia |

### Resultados en Colab (CPU, 2 núcleos, 12,7 GB de memoria)

- **P1, P2, P3, P4, P5, P7 y P9: ✅ ciertas.**
  - Los modelos en miniatura coinciden token a token con `transformers` (20/20, 16/16 y 32/32), tengan en memoria 8, 2
    o 1 expertos.
  - La instalación descargó el modelo (21,4 GB) en **4,7 minutos**.
  - El modelo de 21,4 GB funcionó usando como máximo **3,0 GB** de memoria (con 16 expertos por capa) y **5,2 GB**
    (con 48).
  - Con las dos configuraciones, el texto generado es **idéntico**: *"La capital de Francia es París, conocida
    mundialmente por sus monumentos icónicos como la Torre Eiffel y el Museo del Louvre…"*. Solo cambia cuántas veces
    encontró el experto ya en memoria (58 % frente a 76 %).
  - Respondió correctamente con el formato de ChatGPT (saludar en cinco idiomas) y con el de Claude (17 × 23 = 391).
  - 11.312 comprobaciones de cálculo sin ningún error.
- **P6: ⚠️.** En Colab fue a **0,4–0,5 tokens/s** (unos 2–3 segundos por token). Por núcleo es más lento que el dato
  publicado; la velocidad depende mucho del procesador y del disco.
- **P8: ⚠️.** Las cuentas de GLM-5.2 cuadran (unos 10 GB en memoria y 367 GB de expertos en disco) y la velocidad
  anunciada es coherente con la velocidad del disco. Con el disco de Colab (0,74 GB/s) el máximo sería de unos
  0,07 tokens/s. El modelo ocupa 429 GB, así que no se pudo ejecutar.

### Limitaciones del proyecto

- **La velocidad es el precio.** Leer del disco es mucho más lento que leer de la memoria:
  - Qwen3.6-35B en Colab: 0,4–0,5 tokens/s. La primera pregunta tardó **225 segundos** incluyendo el arranque, y cada
    petición a la API, entre 1 y 1,5 minutos.
  - GLM-5.2 en un portátil con 25 GB y un disco normal: **0,05–0,1 tokens/s**, es decir, 10–20 segundos por palabra.
    Con 128 GB de memoria o un Mac M5 Max mejora hasta 1–2 tokens/s.
- **Solo sirve para modelos de expertos.** En un modelo normal, que usa todo para cada token, no hay nada que dejar en
  el disco.
- **El disco importa mucho.** Hace falta un disco rápido (NVMe); con discos lentos la velocidad cae en picado.
- **Ocupa mucho espacio en disco**: 19–23 GB los modelos medianos y unos 430 GB GLM-5.2.
- **La calidad depende del modelo, no de colibrí.** Colibrí reproduce exactamente lo que haría el modelo, incluidos
  sus fallos. Por ejemplo, al preguntarle a Qwen3.6 *"qué es un modelo de inteligencia artificial de expertos"*,
  contestó describiendo los **sistemas expertos** clásicos, que son otra cosa distinta a los modelos de expertos.
- Para conversar con fluidez sigue siendo mejor un modelo que quepa entero en memoria, o un servicio en la nube. La
  ventaja de colibrí es poder usar modelos que de otra forma no cabrían, con total privacidad.

### Limitaciones de la comprobación

- La comparación exacta con `transformers` se hace con **modelos en miniatura** con números aleatorios, que tienen la
  misma estructura que los reales. Con el modelo real se ha comprobado que la respuesta no cambia según la memoria,
  pero no se ha comparado con `transformers`, porque eso exigiría cargar el modelo completo.
- La **velocidad** medida es la de Colab (2 núcleos y un disco de red de 0,74 GB/s). En un ordenador con más núcleos y
  un disco NVMe será bastante mayor.
- **GLM-5.2 no se ha ejecutado**: solo se han comprobado sus cuentas y la coherencia de la velocidad anunciada.
- No se ha probado la decodificación especulativa (escribir varios tokens de golpe) ni el uso con tarjeta gráfica.

---

## Comparación rápida

| | LLMs-from-scratch | ds4 | colibrí |
|---|---|---|---|
| Para qué sirve | Aprender cómo funciona un LLM por dentro | Usar modelos gigantes rápido en equipos personales muy potentes | Usar modelos de expertos que no caben en tu memoria |
| Lenguaje | Python + PyTorch | C (+ Metal, CUDA o ROCm) | C |
| Ordenador necesario | Cualquiera; mejor con GPU | 64–128 GB de memoria y hardware concreto | Uno normal con disco rápido; mejor con más memoria |
| Velocidad | No es su objetivo | Alta (18–39 tokens/s con DeepSeek V4 Flash) | Baja con modelos enormes (0,05–2 tokens/s) |
| ¿Se pudo ejecutar un modelo en Colab? | Sí (GPT-2) | No (no cabe) | Sí (Qwen3.6-35B, 23 GB) |
| Resultado en Colab | 33/33 ✅ | 24/24 ✅ | 8 ✅ · 2 ⚠️ |

## Versiones probadas

| Proyecto | Versión (commit) | Fecha |
|---|---|---|
| LLMs-from-scratch | `b5a5165` | 6 de noviembre de 2026 |
| ds4 | `b3f5018` | Último commit disponible el 6 de noviembre de 2026 |
| colibrí | `0a5504a1` (v2.1.0) | 31 de octubre de 2026 |

Los tres proyectos cambian a menudo. Con versiones más nuevas, algunos resultados pueden variar.
