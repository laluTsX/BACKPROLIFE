from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta

class DataExtractor:
    """Clase para extraer datos de MongoDB"""
    
    def __init__(self, mongo_uri, db_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
    # ============================================
    # AQUÍ SE EXTRAE EL HISTORIAL DE FATIGA
    # ============================================
    def extract_fatigue_logs(self, days_back=30, user_id=None):
        """
        Extrae logs de fatiga de los últimos N días
        - Busca en la colección logs_energia
        - Filtra por fecha y usuario
        - Convierte a DataFrame para análisis
        """
        fecha_limite = datetime.now() - timedelta(days=days_back)
        
        query = {'fecha_hora': {'$gte': fecha_limite.isoformat()}}
        if user_id:
            query['id_usuario'] = user_id
            
        cursor = self.db.logs_energia.find(
            query,
            {'_id': 0, 'id_usuario': 1, 'nivel_fatiga': 1, 
            'ear_value': 1, 'fecha_hora': 1}
        )
        
        df = pd.DataFrame(list(cursor))
        
        if not df.empty:
            df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
            
        return df
    
    # ============================================
    # AQUÍ SE EXTRAEN LOS DATOS DE USUARIOS
    # ============================================
    def extract_user_data(self, user_id=None):
        """Extrae datos de usuarios registrados (sin contraseña)"""
        query = {}
        if user_id:
            query['id_usuario'] = user_id
            
        cursor = self.db.usuarios.find(
            query,
            {'_id': 0, 'password_hash': 0}
        )
        
        df = pd.DataFrame(list(cursor))
        return df
    
    # ============================================
    # AQUÍ SE EXTRAE TODO JUNTO
    # ============================================
    def extract_all_data(self, days_back=30):
        """Extrae todos los datos para el proceso ELT"""
        print("Extrayendo datos de MongoDB...")
        
        fatigue_df = self.extract_fatigue_logs(days_back)
        users_df = self.extract_user_data()
        
        print(f"Datos extraídos: {len(fatigue_df)} registros de fatiga, "
            f"{len(users_df)} usuarios")
        
        return fatigue_df, users_df