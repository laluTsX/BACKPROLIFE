import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import json
import os

class DataLoader:
    """Clase para cargar datos procesados"""
    
    def __init__(self, mongo_uri, db_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        os.makedirs('exports', exist_ok=True)
        
    def create_collections(self):
        """Crea las colecciones necesarias si no existen"""
        collections = [
            'analytics_diario',
            'resumen_usuario',
            'alertas_fatiga',
            'reportes_etl'
        ]
        
        existing = self.db.list_collection_names()
        created = []
        
        for col in collections:
            if col not in existing:
                self.db.create_collection(col)
                created.append(col)
                
        if created:
            print(f"Colecciones creadas: {', '.join(created)}")
            
    def load_daily_analytics(self, df):
        """Agrupa por día y guarda: EAR promedio, registros, alertas por día"""
        if df.empty:
            print("No hay datos diarios para cargar")
            return
            
        print("Cargando analytics diarios...")
        
        daily = df.groupby(df['fecha_hora'].dt.date).agg(
            total_registros=('ear_value', 'count'),
            ear_promedio=('ear_value', 'mean'),
            usuarios_unicos=('id_usuario', 'nunique'),
            fatiga_alta_count=('nivel_fatiga', lambda x: (x==3).sum()),
            fatiga_moderada_count=('nivel_fatiga', lambda x: (x==2).sum()),
            anomalias_count=('es_anomalia', 'sum')
        ).round(2)
        
        records = []
        for fecha, row in daily.iterrows():
            records.append({
                'fecha': str(fecha),
                'total_registros': int(row['total_registros']),
                'ear_promedio': float(row['ear_promedio']),
                'usuarios_unicos': int(row['usuarios_unicos']),
                'fatiga_alta_count': int(row['fatiga_alta_count']),
                'fatiga_moderada_count': int(row['fatiga_moderada_count']),
                'anomalias_count': int(row['anomalias_count']),
                'procesado_en': datetime.now().isoformat()
            })
            
        if records:
            self.db.analytics_diario.insert_many(records)
            print(f"Cargados {len(records)} registros diarios")
            
    def load_user_summary(self, agg_df):
        """Guarda: % fatiga alta, moderada, normal, EAR promedio, días monitoreados del usuario"""
        if agg_df.empty:
            print("No hay resumen de usuarios para cargar")
            return
            
        print("Cargando resumen por usuario...")
        
        self.db.resumen_usuario.delete_many({})
        
        records = []
        for user_id, row in agg_df.iterrows():
            record = {
                'id_usuario': int(user_id) if not isinstance(user_id, int) else user_id,
                'total_registros': int(row['total_registros']),
                'ear_promedio': float(row['ear_promedio']),
                'ear_minimo': float(row['ear_minimo']),
                'ear_maximo': float(row['ear_maximo']),
                'fatiga_alta_pct': float(row['fatiga_alta_pct']),
                'fatiga_moderada_pct': float(row['fatiga_moderada_pct']),
                'fatiga_normal_pct': float(row['fatiga_normal_pct']),
                'dias_monitoreo': int(row['dias_monitoreo']),
                'registros_por_dia': float(row['registros_por_dia']),
                'ultimo_registro': row['ultimo_registro'].isoformat(),
                'actualizado_en': datetime.now().isoformat()
            }
            records.append(record)
            
        if records:
            self.db.resumen_usuario.insert_many(records)
            print(f"Cargados {len(records)} resúmenes de usuarios")
            
    def load_alerts(self, df):
        """Filtra solo nivel 3 (Fatiga Alta) y guarda como alerta"""
        alertas = df[df['nivel_fatiga'] == 3].copy()
        
        if alertas.empty:
            print("No hay alertas para cargar")
            return
            
        print(f"Cargando {len(alertas)} alertas de fatiga...")
        
        records = []
        for _, row in alertas.iterrows():
            records.append({
                'id_usuario': int(row['id_usuario']),
                'fecha_hora': row['fecha_hora'].isoformat(),
                'ear_value': float(row['ear_value']),
                'periodo_dia': str(row['periodo_dia']),
                'es_anomalia': bool(row['es_anomalia']),
                'procesado_en': datetime.now().isoformat()
            })
            
        if records:
            for record in records:
                self.db.alertas_fatiga.update_one(
                    {
                        'id_usuario': record['id_usuario'],
                        'fecha_hora': record['fecha_hora']
                    },
                    {'$set': record},
                    upsert=True
                )
            print(f"Alertas procesadas: {len(records)}")
            
    def load_etl_report(self, report_data):
        """Carga reporte de la ejecución ETL"""
        self.db.reportes_etl.insert_one({
            'fecha_ejecucion': datetime.now().isoformat(),
            'estadisticas': report_data['stats'],
            'registros_procesados': report_data['total_processed'],
            'colecciones_actualizadas': report_data['collections_updated']
        })
        
    def export_to_csv(self, df, filename):
        """Exporta solo la informacion importante en un CSV limpio"""
        """Genera archivo CSV con columnas: ID, Nombre, Nvl. Fatiga, Valor EAR, Fecha"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Obtener nombres de usuarios
            usuarios = {}
            try:
                cursor = self.db.usuarios.find({}, {'id_usuario': 1, 'nombre': 1, '_id': 0})
                for user in cursor:
                    usuarios[user['id_usuario']] = user.get('nombre', 'Usuario ' + str(user['id_usuario']))
            except:
                pass
            
            # Crear DataFrame simplificado
            df_export = pd.DataFrame()
            df_export['ID'] = df['id_usuario'].astype(int)
            df_export['Nombre'] = df['id_usuario'].map(usuarios).fillna('Usuario ' + df['id_usuario'].astype(str))
            df_export['Nvl. Fatiga'] = df['nivel_fatiga'].map({1: 'Normal', 2: 'Moderada', 3: 'Alta'})
            df_export['Valor EAR'] = df['ear_value'].round(3)
            
            # Fecha en formato español
            meses_es = {
                1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril', 
                5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto', 
                9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
            }
            
            df_export['Fecha'] = df['fecha_hora'].apply(
                lambda x: f"{x.day} de {meses_es.get(x.month, '')} de {x.year} - {x.strftime('%I:%M %p')}"
            )
            
            # Guardar CSV - CAMBIA sep=';' POR sep=','
            filepath = f"exports/{filename}.csv"
            df_export.to_csv(filepath, index=False, encoding='utf-8-sig', sep=',')  # ← CAMBIADO A COMA
            
            print(f" CSV exportado: {filepath}")
            print(f"   - Columnas: ID | Nombre | Nvl. Fatiga | Valor EAR | Fecha")
            print(f"   - Total registros: {len(df_export)}")
            
            # Resumen en consola
            total = len(df)
            normal = (df['nivel_fatiga'] == 1).sum()
            moderada = (df['nivel_fatiga'] == 2).sum()
            alta = (df['nivel_fatiga'] == 3).sum()
            
            print(f"\n RESUMEN:")
            print(f"   Total: {total} | Normal: {normal} ({normal/total*100:.1f}%) | Moderada: {moderada} ({moderada/total*100:.1f}%) | Alta: {alta} ({alta/total*100:.1f}%)")
            
            return filepath
            
        except Exception as e:
            print(f" Error al exportar CSV: {e}")
            import traceback
            traceback.print_exc()
            return None