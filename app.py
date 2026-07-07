import cv2
import mediapipe as mp
import numpy as np
import base64
import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
import bcrypt
from groq import Groq
groq_client = Groq(api_key="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")

# ======================  IMPORTS ML ======================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# ============================================================
# MONGODB ATLAS
# ============================================================
MONGO_URI = "mongodb+srv://prolife_admin:s4q37OT3oYhGOSlf@prolife.6yw9rnd.mongodb.net/?retryWrites=true&w=majority&appName=Prolife"
MONGO_DB = "prolife_db"

client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
try:
    client.admin.command("ping")
    print("✅ Conexión exitosa a MongoDB Atlas")
except Exception as e:
    print(f"❌ Error de conexión: {e}")

db = client[MONGO_DB]
usuarios_col = db.usuarios
logs_energia_col = db.logs_energia
perfiles_col = db.perfiles_salud
tokens_col = db.session_tokens

# ============================================================
# MEDIAPIPE 
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)
OJO_IZQ = [362, 385, 387, 263, 373, 380]
OJO_DER = [33, 160, 158, 133, 153, 144]


def calcular_ear(landmarks, indices):
    import math

    def d(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    v1 = d(landmarks[indices[1]], landmarks[indices[5]])
    v2 = d(landmarks[indices[2]], landmarks[indices[4]])
    h = d(landmarks[indices[0]], landmarks[indices[3]])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.3


def get_perfil(id_usuario):
    return perfiles_col.find_one({"id_usuario": id_usuario}, {"_id": 0})


def umbral_ear(perfil):
    th_alta, th_mod = 0.21, 0.26
    if not perfil:
        return th_alta, th_mod
    conds = perfil.get("salud_facial", {}).get("condiciones", [])
    afectan = {"ptosis", "blefarospasmo", "estrabismo", "nistagmo"}
    if any(c in afectan for c in conds):
        return th_alta - 0.04, th_mod - 0.04
    return th_alta, th_mod


# ====================== HELPERS ML ======================
def calcular_metricas_regresion(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": (
            float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            if np.all(y_true != 0)
            else None
        ),
    }


# ============================================================
# AUTH
# ============================================================


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ProLife API v3.1 - ML + Roles"}), 200


@app.route("/login", methods=["POST"])
def login_usuario():
    try:
        data = request.json
        email, password = data.get("email"), data.get("password")
        if not email or not password:
            return jsonify({"error": "Faltan campos"}), 400
        u = usuarios_col.find_one({"email": email})
        if not u:
            return jsonify({"error": "Email no registrado"}), 404
        if bcrypt.checkpw(password.encode(), u["password_hash"].encode()):
            return (
                jsonify(
                    {
                        "mensaje": "Login exitoso",
                        "id_usuario": u["id_usuario"],
                        "nombre": u["nombre"],
                        "email": u["email"],
                        "rol": u.get("rol", "usuario"),  # ← NUEVO
                    }
                ),
                200,
            )
        return jsonify({"error": "Contraseña incorrecta"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/registro", methods=["POST"])
def registrar_usuario():
    try:
        data = request.json
        nombre = data.get("nombre")
        email = data.get("email")
        password = data.get("password")
        rol = data.get("rol", "usuario")  

        if not all([nombre, email, password]):
            return jsonify({"error": "Faltan campos"}), 400
        if usuarios_col.find_one({"email": email}):
            return jsonify({"error": "Email ya registrado"}), 400

        ultimo = usuarios_col.find_one(sort=[("id_usuario", -1)])
        nuevo_id = (ultimo["id_usuario"] + 1) if ultimo else 1

        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        usuarios_col.insert_one(
            {
                "id_usuario": nuevo_id,
                "nombre": nombre,
                "email": email,
                "password_hash": pw_hash,
                "rol": rol,  # ← NUEVO
                "fecha_registro": datetime.now().isoformat(),
            }
        )

        return (
            jsonify(
                {
                    "mensaje": "Registrado",
                    "id_usuario": nuevo_id,
                    "nombre": nombre,
                    "email": email,
                    "rol": rol,
                }
            ),
            201,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# TOKENS DE SESIÓN (React → Streamlit )
# ============================================================


@app.route("/session_token", methods=["POST"])
def crear_session_token():
    """
    React llama esto al hacer clic en el botón de Detección.
    Devuelve un token de un solo uso que expira en 5 minutos.
    Streamlit lo canjea por el id_usuario sin que el usuario lo vea.
    """
    try:
        data = request.json
        id_usuario = data.get("id_usuario")
        if not id_usuario:
            return jsonify({"error": "id_usuario requerido"}), 400

        token = str(uuid.uuid4())
        expira_en = datetime.utcnow() + timedelta(minutes=5)

        # Eliminar tokens anteriores del mismo usuario
        tokens_col.delete_many({"id_usuario": id_usuario})

        tokens_col.insert_one(
            {
                "token": token,
                "id_usuario": id_usuario,
                "expira_en": expira_en,
                "usado": False,
            }
        )
        return jsonify({"token": token}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/session_token/<token>", methods=["GET"])
def canjear_session_token(token):
    """
    Streamlit llama esto con el token de la URL.
    Retorna id_usuario si el token es válido y no ha expirado.
    El token se marca como usado (single-use).
    """
    try:
        doc = tokens_col.find_one({"token": token})
        if not doc:
            return jsonify({"error": "Token inválido"}), 404
        if doc.get("usado"):
            return jsonify({"error": "Token ya utilizado"}), 403
        if datetime.utcnow() > doc["expira_en"]:
            tokens_col.delete_one({"token": token})
            return jsonify({"error": "Token expirado"}), 403

        # Marcar como usado
        tokens_col.update_one({"token": token}, {"$set": {"usado": True}})

        # Devolver datos del usuario
        perfil = perfiles_col.find_one({"id_usuario": doc["id_usuario"]}, {"_id": 0})
        usuario = usuarios_col.find_one(
            {"id_usuario": doc["id_usuario"]}, {"_id": 0, "password_hash": 0}
        )
        return (
            jsonify(
                {
                    "id_usuario": doc["id_usuario"],
                    "nombre": usuario.get("nombre", "") if usuario else "",
                    "email": usuario.get("email", "") if usuario else "",
                    "perfil": perfil or {},
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# PERFIL DE SALUD
# ============================================================


@app.route("/perfil_salud", methods=["POST"])
def guardar_perfil_salud():
    try:
        data = request.json
        id_usuario = data.get("id_usuario")
        if not id_usuario:
            return jsonify({"error": "id_usuario requerido"}), 400
        perfil = {
            "id_usuario": id_usuario,
            "salud_facial": data.get("salud_facial", {}),
            "trabajo": data.get("trabajo", {}),
            "actividad_fisica": data.get("actividad_fisica", {}),
            "fecha_actualizacion": datetime.now().isoformat(),
        }
        perfiles_col.update_one(
            {"id_usuario": id_usuario}, {"$set": perfil}, upsert=True
        )
        return jsonify({"mensaje": "Perfil guardado"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/perfil_salud/<int:id_usuario>", methods=["GET"])
def obtener_perfil_salud(id_usuario):
    p = perfiles_col.find_one({"id_usuario": id_usuario}, {"_id": 0})
    if not p:
        return jsonify({"error": "Perfil no encontrado"}), 404
    return jsonify(p), 200


@app.route("/perfil_salud/<int:id_usuario>", methods=["PUT"])
def actualizar_perfil_salud(id_usuario):
    try:
        data = request.json
        protegidos = {"id_usuario", "email"}
        update = {k: v for k, v in data.items() if k not in protegidos}
        update["fecha_actualizacion"] = datetime.now().isoformat()
        res = perfiles_col.update_one({"id_usuario": id_usuario}, {"$set": update})
        if res.matched_count == 0:
            return jsonify({"error": "Perfil no encontrado"}), 404
        return jsonify({"mensaje": "Perfil actualizado"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# LOGS DE FATIGA
# ============================================================


@app.route("/log_nivel", methods=["POST"])
def log_nivel():
    """Recibe logs directos desde Streamlit (sin imagen)."""
    try:
        data = request.json
        logs_energia_col.insert_one(
            {
                "id_usuario": data.get("id_usuario"),
                "nivel_fatiga": data.get("nivel_fatiga", 1),
                "ear_value": data.get("ear_value", 0),
                "estado_key": data.get("estado_key", "normal"),
                "fecha_hora": data.get("fecha_hora", datetime.now().isoformat()),
            }
        )
        return jsonify({"mensaje": "Log guardado"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analizar_fatiga", methods=["POST"])
def analizar_fatiga():
    try:
        data = request.json
        image_data = data["image"].split(",")[1]
        id_usuario = data.get("id_usuario")

        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not res.multi_face_landmarks:
            return jsonify({"estado": "Rostro no detectado", "nivel": 0, "ear": 0})

        puntos = res.multi_face_landmarks[0].landmark
        perfil = get_perfil(id_usuario) if id_usuario else None
        th_alta, th_mod = umbral_ear(perfil)
        ear = (calcular_ear(puntos, OJO_IZQ) + calcular_ear(puntos, OJO_DER)) / 2

        if ear < th_alta:
            estado, nivel = "Fatiga Alta", 3
        elif ear < th_mod:
            estado, nivel = "Fatiga Moderada", 2
        else:
            estado, nivel = "Normal", 1

        if id_usuario:
            logs_energia_col.insert_one(
                {
                    "id_usuario": id_usuario,
                    "nivel_fatiga": nivel,
                    "ear_value": round(ear, 3),
                    "fecha_hora": datetime.now().isoformat(),
                }
            )
        return jsonify({"estado": estado, "ear": round(ear, 3), "nivel": nivel})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/historial_fatiga/<int:id_usuario>", methods=["GET"])
def obtener_historial_fatiga(id_usuario):
    try:
        limit = int(request.args.get("limit", 200))
        desde = request.args.get("desde")  # ISO string opcional
        query = {"id_usuario": id_usuario}
        if desde:
            query["fecha_hora"] = {"$gte": desde}
        logs = list(
            logs_energia_col.find(
                query,
                {
                    "_id": 0,
                    "nivel_fatiga": 1,
                    "ear_value": 1,
                    "fecha_hora": 1,
                    "estado_key": 1,
                },
            )
            .sort("fecha_hora", -1)
            .limit(limit)
        )
        return jsonify(logs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/resumen_fatiga/<int:id_usuario>", methods=["GET"])
def resumen_fatiga(id_usuario):
    """Stats agregados para las gráficas del dashboard."""
    try:
        desde = request.args.get("desde", "")
        query = {"id_usuario": id_usuario}
        if desde:
            query["fecha_hora"] = {"$gte": desde}
        logs = list(logs_energia_col.find(query, {"_id": 0}))
        total = len(logs)
        if total == 0:
            return (
                jsonify({"total": 0, "por_nivel": {}, "por_hora": {}, "por_dia": {}}),
                200,
            )

        por_nivel = {1: 0, 2: 0, 3: 0}
        por_hora = {str(h): 0 for h in range(24)}
        por_dia = {}

        for log in logs:
            n = log.get("nivel_fatiga", 1)
            por_nivel[n] = por_nivel.get(n, 0) + 1
            try:
                dt = datetime.fromisoformat(log["fecha_hora"])
                hora = str(dt.hour)
                dia = dt.strftime("%Y-%m-%d")
                por_hora[hora] = por_hora.get(hora, 0) + 1
                por_dia[dia] = por_dia.get(dia, {})
                por_dia[dia][str(n)] = por_dia[dia].get(str(n), 0) + 1
            except:
                pass

        ears = [l.get("ear_value", 0) for l in logs if l.get("ear_value")]
        n_tot = max(total, 1)
        return (
            jsonify(
                {
                    "total": total,
                    "por_nivel": por_nivel,
                    "por_hora": por_hora,
                    "por_dia": por_dia,
                    "ear_promedio": round(sum(ears) / len(ears), 3) if ears else 0,
                    "nivel_promedio": round(
                        sum(l.get("nivel_fatiga", 1) for l in logs) / n_tot, 2
                    ),
                    "nivel_1": por_nivel.get(1, 0),
                    "nivel_2": por_nivel.get(2, 0),
                    "nivel_3": por_nivel.get(3, 0),
                    "pct_normal": round(por_nivel.get(1, 0) / n_tot * 100),
                    "pct_moderada": round(por_nivel.get(2, 0) / n_tot * 100),
                    "pct_alta": round(por_nivel.get(3, 0) / n_tot * 100),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MODELOS ML
# ============================================================


@app.route("/modelo/regresion/simple", methods=["POST"])
def regresion_lineal_simple():
    try:
        data = request.json
        X = np.array(data["X"]).reshape(-1, 1)
        y = np.array(data["y"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return jsonify(
            {
                "modelo": "Regresión Lineal Simple",
                "coeficiente": float(model.coef_[0]),
                "intercepto": float(model.intercept_),
                **calcular_metricas_regresion(y_test, y_pred),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/modelo/regresion/multiple", methods=["POST"])
def regresion_multiple():
    try:
        data = request.json
        X = np.array(data["X"])  # shape (n_samples, n_features)
        y = np.array(data["y"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return jsonify(
            {
                "modelo": "Regresión Lineal Múltiple",
                "coeficientes": model.coef_.tolist(),
                "intercepto": float(model.intercept_),
                **calcular_metricas_regresion(y_test, y_pred),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/modelo/clasificacion/<int:id_usuario>", methods=["POST"])
def clasificacion(id_usuario):
    try:
        algoritmo = request.json.get("algoritmo", "random_forest")

        # Cargar datos del usuario
        logs = list(logs_energia_col.find({"id_usuario": id_usuario}, {"_id": 0}))
        if len(logs) < 10:
            return jsonify({"error": "No hay suficientes datos para entrenar"}), 400

        df = pd.DataFrame(logs)
        # Features ejemplo (puedes expandir)
        X = (
            df[["ear_value"]].values
            if "ear_value" in df.columns
            else np.random.rand(len(df), 1)
        )
        y = df["nivel_fatiga"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        if algoritmo == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algoritmo == "decision_tree":
            model = DecisionTreeClassifier(random_state=42)
        elif algoritmo == "svm":
            model = SVC(probability=True)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred).tolist()

        return jsonify(
            {
                "algoritmo": algoritmo,
                "metricas": {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(
                        precision_score(y_test, y_pred, average="weighted")
                    ),
                    "recall": float(recall_score(y_test, y_pred, average="weighted")),
                    "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
                },
                "matriz_confusion": cm,
                "n_muestras": len(y),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# USUARIOS + INDEX 
# ============================================================


@app.route("/usuarios", methods=["GET"])
def obtener_usuarios():
    try:
        u = list(usuarios_col.find({}, {"_id": 0, "password_hash": 0}))
        return jsonify(u), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return (
        jsonify(
            {
                "message": "ProLife API v3.1 - ML + Roles",
                "endpoints": [
                    "POST /registro",
                    "POST /login",
                    "POST /session_token",
                    "GET /session_token/<token>",
                    "POST /perfil_salud",
                    "GET /perfil_salud/<id>",
                    "PUT /perfil_salud/<id>",
                    "POST /analizar_fatiga",
                    "POST /log_nivel",
                    "GET /historial_fatiga/<id>",
                    "GET /resumen_fatiga/<id>",
                    "GET /health",
                    "GET /usuarios",
                    "POST /modelo/regresion/simple",
                    "POST /modelo/regresion/multiple",
                    "POST /modelo/clasificacion/<id_usuario>",
                ],
            }
        ),
        200,
    )
    
@app.route('/recomendaciones_ia/<int:id_usuario>', methods=['GET'])
def recomendaciones_ia(id_usuario):
    try:
        # Versión temporal sin Groq (para que funcione el registro)
        return jsonify({
            "recomendaciones": [
                {
                    "tipo": "playlist",
                    "titulo": "Lo-fi para concentrarte",
                    "descripcion": "Música suave que reduce el estrés y mejora el foco",
                    "duracion": "45 min"
                },
                {
                    "tipo": "playlist",
                    "titulo": "Energía mañanera",
                    "descripcion": "Canciones upbeat para empezar el día con energía",
                    "duracion": "30 min"
                },
                {
                    "tipo": "actividad",
                    "titulo": "Ejercicio de respiración 4-7-8",
                    "descripcion": "Respira 4 seg, retiene 7, exhala 8. Repite 4 veces.",
                    "duracion": "2 min"
                },
                {
                    "tipo": "actividad",
                    "titulo": "Micro-pausa de estiramiento",
                    "descripcion": "Estira cuello, hombros y muñecas por 60 segundos",
                    "duracion": "1 min"
                }
            ]
        })
    except Exception as e:
        return jsonify({"recomendaciones": []}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 ProLife API v3.1 → http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
