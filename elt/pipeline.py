from elt.extract import DataExtractor
from elt.transform import DataTransformer
from elt.load import DataLoader
from datetime import datetime
import pandas as pd

class ETLPipeline:
    """Proceso ELT"""
    
    def __init__(self, mongo_uri, db_name):
        self.extractor = DataExtractor(mongo_uri, db_name)
        self.transformer = DataTransformer()
        self.loader = DataLoader(mongo_uri, db_name)

    # ============================================
    # AQUÍ SE EJECUTA TODO EL PROCESO ELT
    # ============================================
        
    def run_full_pipeline(self, days_back=30):
        """Ejecuta el pipeline ELT completo"""
        """
        Orden del ELT:
        1. EXTRACT - Extrae datos de MongoDB
        2. LOAD - Guarda copia cruda (respaldo)
        3. TRANSFORM - Limpia, enriquece, detecta anomalías
        4. Guarda resultados finales
        5. EXPORTA - Generando un CSV
        """
        print("\n" + "="*50)
        print(f"INICIANDO ELT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        
        try:
            # 1. EXTRACT (E) - Extraer datos históricos
            print("FASE 1: EXTRACT (Extraer)")
            fatigue_df, users_df = self.extractor.extract_all_data(days_back)
            
            if fatigue_df.empty:
                print("No hay datos para procesar")
                return None
            
            # 2. LOAD (L) - Cargar datos crudos primero
            print("\nFASE 2: LOAD (Cargar datos crudos)")
            self.loader.create_collections()
            
            # Guardar copia de datos SIN procesar
            records_crudos = []
            for _, row in fatigue_df.iterrows():
                records_crudos.append({
                    'id_usuario': int(row['id_usuario']),
                    'nivel_fatiga': int(row['nivel_fatiga']),
                    'ear_value': float(row['ear_value']),
                    'fecha_hora': row['fecha_hora'].isoformat() if hasattr(row['fecha_hora'], 'isoformat') else str(row['fecha_hora']),
                    'cargado_en': datetime.now().isoformat()
                })
            
            if records_crudos:
                if 'datos_crudos_elt' not in self.loader.db.list_collection_names():
                    self.loader.db.create_collection('datos_crudos_elt')
                self.loader.db.datos_crudos_elt.insert_many(records_crudos)
                print(f"Cargados {len(records_crudos)} registros crudos en datos_crudos_elt")
                
            # 3. TRANSFORM (T) - Ahora transformamos  - Limpiar, enriquecer, detectar anomalías
            print("\nFASE 3: TRANSFORM (Transformar)")
            df_clean = self.transformer.clean_fatigue_data(fatigue_df)    
            df_enriched = self.transformer.engineer_features(df_clean)   # ← Aquí clasifica período
            df_final = self.transformer.detect_anomalies(df_enriched)    # ← Aquí detecta anomalías
            agg_df = self.transformer.calculate_aggregations(df_final)   # ← Aquí calcula % fatiga
            
            # 4. LOAD - Guardar resultados transformados
            print("\nFASE 4: LOAD (Cargar resultados)")
            self.loader.load_daily_analytics(df_final)    # ← Datos por día
            self.loader.load_user_summary(agg_df)         # ← Resumen usuario (segmentación)
            self.loader.load_alerts(df_final)             # ← Alertas fatiga alta
            
            collections_updated = [
                'datos_crudos_elt',
                'analytics_diario',
                'resumen_usuario',
                'alertas_fatiga'
            ]
            
            report = {
                'stats': self.transformer.stats,
                'total_processed': len(df_final),
                'collections_updated': collections_updated
            }
            
            self.loader.load_etl_report(report)
            
            # 5. EXPORTAR CSV
            if len(df_final) > 0:
                print("\nFASE 5: EXPORTANDO CSV")
                archivo = self.loader.export_to_csv(df_final, 'fatiga')
                if archivo:
                    print(f" Archivo: {archivo}")
            
            print("\n" + "="*50)
            print("ELT COMPLETADO EXITOSAMENTE")
            print("="*50)
            
            return {
                'processed_df': df_final,
                'aggregations': agg_df,
                'report': report
            }
            
        except Exception as e:
            print(f"\nERROR EN ELT: {str(e)}")
            raise e
            
    # ============================================
    # ANÁLISIS RÁPIDO PARA UN USUARIO
    # ============================================

    def run_quick_analysis(self, user_id):
        """Analisis rapido para un usuario especifico"""
        print(f"\nAnalisis rapido para usuario {user_id}")
        
        fatigue_df = self.extractor.extract_fatigue_logs(
            days_back=7,
            user_id=user_id
        )
        
        if fatigue_df.empty:
            print(f"No hay datos para el usuario {user_id}")
            return None
            
        df_clean = self.transformer.clean_fatigue_data(fatigue_df)
        df_enriched = self.transformer.engineer_features(df_clean)
        
        analysis = {
            'total_registros': len(df_enriched),
            'ear_promedio': df_enriched['ear_value'].mean(),
            'nivel_fatiga_promedio': df_enriched['nivel_fatiga'].mean(),
            'porcentaje_fatiga_alta': (df_enriched['nivel_fatiga'] == 3).mean() * 100,
            'mejor_horario': df_enriched.groupby('periodo_dia')['ear_value'].mean().idxmax(),
            'peor_horario': df_enriched.groupby('periodo_dia')['ear_value'].mean().idxmin()
        }
        
        return analysis