import pandas as pd
import numpy as np
from datetime import datetime

class DataTransformer:
    """Clase para transformar y limpiar datos"""
    
    def __init__(self):
        self.stats = {}
        
    def clean_fatigue_data(self, df):
        """Limpia y valida los datos de fatiga"""
        print("Limpiando datos de fatiga...")
        
        if df.empty:
            print("No hay datos para limpiar")
            return df
            
        df_clean = df.copy()
        
        # Estadísticas iniciales
        initial_count = len(df_clean)
        self.stats['initial_records'] = initial_count
        
        # 1. Eliminar registros con EAR fuera de rango (0-1)
        df_clean = df_clean[
            (df_clean['ear_value'] >= 0) & 
            (df_clean['ear_value'] <= 1)
        ]
        
        # 2. Validar niveles de fatiga (1-3)
        df_clean = df_clean[
            df_clean['nivel_fatiga'].isin([1, 2, 3])
        ]
        
        # 3. Eliminar duplicados
        df_clean = df_clean.drop_duplicates(
            subset=['id_usuario', 'fecha_hora']
        )
        
        # 4. Manejar valores nulos
        df_clean = df_clean.fillna({
            'ear_value': df_clean['ear_value'].median(),
            'nivel_fatiga': df_clean['nivel_fatiga'].mode()[0]
        })
        
        self.stats['cleaned_records'] = len(df_clean)
        self.stats['removed_records'] = initial_count - len(df_clean)
        
        print(f"Registros removidos: {self.stats['removed_records']}")
        
        return df_clean
    
    def engineer_features(self, df):
        """Crea nuevas características para análisis"""
        print("Creando características...")
        
        if df.empty:
            return df
            
        df_enriched = df.copy()
        
        # 1. Extraer componentes temporales
        df_enriched['hora'] = df_enriched['fecha_hora'].dt.hour
        df_enriched['dia_semana'] = df_enriched['fecha_hora'].dt.day_name()
        df_enriched['dia_numero'] = df_enriched['fecha_hora'].dt.day
        df_enriched['mes'] = df_enriched['fecha_hora'].dt.month
        df_enriched['es_fin_semana'] = df_enriched['dia_semana'].isin(
            ['Saturday', 'Sunday']
        ).astype(int)
        
        # 2. Clasificar período del día
        def clasificar_periodo(hora):
            if 5 <= hora < 12:
                return 'Mañana'
            elif 12 <= hora < 17:
                return 'Tarde'
            elif 17 <= hora < 22:
                return 'Noche'
            else:
                return 'Madrugada'
                
        df_enriched['periodo_dia'] = df_enriched['hora'].apply(clasificar_periodo)
        
        
        df_enriched = df_enriched.sort_values('fecha_hora')
        df_enriched['ear_ma_5'] = df_enriched.groupby('id_usuario')['ear_value'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # 4. Calcular cambios respecto al registro anterior
        df_enriched['ear_change'] = df_enriched.groupby('id_usuario')['ear_value'].diff()
        
        # 5. Categorizar fatiga
        def categorizar_fatiga(nivel):
            if nivel == 1:
                return 'Normal'
            elif nivel == 2:
                return 'Moderada'
            else:
                return 'Alta'
                
        df_enriched['categoria_fatiga'] = df_enriched['nivel_fatiga'].apply(
            categorizar_fatiga
        )
        
        return df_enriched
    
    def calculate_aggregations(self, df):
        """Calcula agregaciones por usuario"""
        print("Calculando estadísticas agregadas...")
        
        if df.empty:
            return pd.DataFrame()
            
        # Agregaciones por usuario
        agg_df = df.groupby('id_usuario').agg(
            total_registros=('ear_value', 'count'),
            ear_promedio=('ear_value', 'mean'),
            ear_minimo=('ear_value', 'min'),
            ear_maximo=('ear_value', 'max'),
            ear_std=('ear_value', 'std'),
            # PORCENTAJES DE FATIGA (SEGMENTACIÓN)
            fatiga_alta_pct=('nivel_fatiga', lambda x: (x == 3).mean() * 100),
            fatiga_moderada_pct=('nivel_fatiga', lambda x: (x == 2).mean() * 100),
            fatiga_normal_pct=('nivel_fatiga', lambda x: (x == 1).mean() * 100),
            primer_registro=('fecha_hora', 'min'),
            ultimo_registro=('fecha_hora', 'max')
        ).round(2)
        
        # Calcular días con registros
        agg_df['dias_monitoreo'] = (
            agg_df['ultimo_registro'] - agg_df['primer_registro']
        ).dt.days + 1
        
        # Calcular frecuencia de registros por día
        agg_df['registros_por_dia'] = (
            agg_df['total_registros'] / agg_df['dias_monitoreo']
        ).round(1)
        
        return agg_df
    
    def detect_anomalies(self, df, threshold=0.15):
        """Detecta anomalías en los niveles de fatiga"""
        print("Detectando anomalías...")
        
        if df.empty:
            return df
            
        # Calcular límites para EAR
        ear_mean = df['ear_value'].mean()
        ear_std = df['ear_value'].std()
        
        lower_bound = ear_mean - (2 * ear_std)
        upper_bound = ear_mean + (2 * ear_std)
        
        # Marcar anomalías
        df['es_anomalia'] = (
            (df['ear_value'] < lower_bound) | 
            (df['ear_value'] > upper_bound)
        ).astype(int)
        
        # Detectar cambios bruscos
        df['cambio_brusco'] = (
            df['ear_change'].abs() > threshold
        ).astype(int)
        
        return df