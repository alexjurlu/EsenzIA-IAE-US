import streamlit as st
import pandas as pd
import os
import plotly.express as px
import pickle
import numpy as np



st.set_page_config(page_title="EsenzIA - IA de Atención al Cliente", layout="wide")

def cargar_tabla_incidencias():
    """
    Función para la carga persistente de los resultados del pipeline. 
    He optado por CSV para facilitar la portabilidad del proyecto sin dependencias de base de datos externas.
    En proyectos anteriores y en la empresa, el uso de XLSX daba problemas en algunas tareas de importaciones.
    """
    ruta = "data/tabla_incidencias.csv"
    if os.path.exists(ruta):
        df_tmp = pd.read_csv(ruta)
        # Eliminamos datos sensibles para la versión pública
        cols_a_borrar = ['ID_Pedido', 'Texto_Completo_Mensaje']
        return df_tmp.drop(columns=[c for c in cols_a_borrar if c in df_tmp.columns])
    return None

def cargar_comparativa():
    ruta = "modelos/comparativa_modelos.pkl"
    if os.path.exists(ruta):
        with open(ruta, 'rb') as f:
            return pickle.load(f)
    return None

df = cargar_tabla_incidencias()
comparativa = cargar_comparativa()

st.title("Sistema de Análisis de Atención al Cliente")

# Usamos pestañas para organizar mejor la información
tab_context, tab1, tab2 = st.tabs(["Contexto y Datos", "Panel de Control", "Métricas y Comparativa"])

with tab_context:
    st.header("Contextualización del Problema")
    st.info("Esenzia es una marca de moda joven sevillana. Año tras año la empresa crece, y con ello se enfrenta a nuevos retos. El incremento del volumen de ventas conduce inevitablemente a un incremento del uso de los servicios, entre ellos Atención al Cliente. Como parte de un proyecto real de la empresa y a modo de una primera toma de contacto y aprendizaje, nace EsenzIA, una IA capaz de clasificar los casos que requieren de esta atención. A continuación se muestra un conjunto de instancias catalogadas usando 4 métodos de clasificación y 2 opciones, usando Word2Vec y sin usarlo. Además, mediante un modelo pre-entrenado de la librería pysentimiento que usa RoBeRTa, se ha añadido el sentimiento interpretado del mensaje, para así poder decidir qué casos según qué impaciencia experimente el cliente requieren una atención más urgente. Se han expuesto por último la comparativa de los 8 posibles escenarios. Por último, para no comprometer los datos de cliente, se han ocultado los números reales de los pedidos y los contenidos de los mensajes, en cumplimiento del RGPD.")
    
    st.divider()
    
    st.header("Metodología de Clasificación y Urgencia")
    st.write("""
    Para que el triaje sea efectivo, no basta con saber de qué trata el correo; necesitamos saber qué tan pronto debemos responder. Mi criterio para determinar la **Prioridad Urgente** se basa en la combinación de dos factores:
    
    *   **Análisis de Sentimiento (RoBERTa)**: Usamos el modelo para detectar el grado de malestar o impaciencia del cliente.
    *   **Criticidad de la Categoría**: Algunas categorías son intrínsecamente más graves (ej. *Pedido Perdido* o *Tara*) que otras (*Consulta General*).
    
    **Nota importante**: No todos los mensajes negativos son urgentes. Un mensaje se marca como **Prioridad ALTA** solo si el cliente muestra un sentimiento negativo **Y** la incidencia es de una categoría que afecta directamente a la entrega o al estado del producto. Esto evita "falsos positivos" en las urgencias y permite centrar el foco donde realmente importa.
    """)

    st.divider()
    
    st.header("Estadísticas de Datos Primitivos")
    if comparativa:
        stats = comparativa['stats_primitivas']
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de Correos Analizados", stats['total_filas'])
        c2.metric("Volumen de Texto (caracteres)", f"{stats['volumen_texto_total']:,}")
        c3.metric("Media de Palabras/Correo", f"{stats['media_palabras_por_correo']:.1f}")
        
        st.divider()
        st.subheader("Metodología de Entrenamiento")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write("**División de Datos:**")
            st.write("- **Entrenamiento (80%):** 260 correos")
            st.write("- **Test (20%):** 65 correos")
        with col_t2:
            st.write("**Validación:**")
            st.write("No he utilizado un tercer conjunto de validación debido a que el tamaño de la muestra (325 correos históricos) es reducido. En su lugar, he optado por un reparto robusto de Train/Test para maximizar el aprendizaje sin comprometer la evaluación final.")
        
        st.divider()
        st.subheader("Distribución Original de Categorías")
        df_dist = pd.DataFrame(list(stats['distribucion_categorias'].items()), columns=['Categoría', 'Cantidad'])
        fig_dist = px.bar(df_dist, x='Categoría', y='Cantidad', color='Categoría', title="Volumen por Categoría")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.divider()
    st.header("Transformación y Procesamiento de Datos")
    st.write("""
    Para pasar de los datos primitivos al modelo de IA, he realizado las siguientes transformaciones:
    1. **Limpieza**: Eliminación de caracteres especiales, números y conversión a minúsculas.
    2. **Tokenización y Stemming**: Reducción de las palabras a su raíz para unificar significados (ej. 'comprando' -> 'compr').
    3. **Filtrado de Stopwords**: Eliminación de palabras sin valor semántico (preposiciones, artículos).
    4. **Vectorización**: Conversión de texto a valores numéricos mediante dos enfoques comparados:
        - **Modelo Base (Sin Word2Vec)**: Utiliza la técnica de Bolsa de Palabras (Bag of Words). Es el método más sencillo y directo en el que "no se hace nada" a nivel semántico, simplemente se cuenta la frecuencia de aparición de los términos.
        - **Modelo Avanzado (Con Word2Vec)**: Utiliza embeddings densos para capturar relaciones de significado entre palabras, representando un salto tecnológico frente al modelo base.
    """)

with tab1:
    if df is not None and not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Casos", len(df))
        col2.metric("Interacciones", df['Num_Emails_Conversacion'].sum())
        col3.metric("Prioridad ALTA", len(df[df['Prioridad'] == 'ALTA (Urgente)']))
        col4.metric("Sentimiento Negativo", len(df[df['Sentimiento_Predominante'] == 'NEG']))

        st.divider()
        
        st.header("Vista General de Incidencias")
        c1, c2, c3 = st.columns(3)
        with c1:
            filtro_cat = st.multiselect("Filtrar por Categoría:", df['Clasificacion_Incidencia'].unique())
        with c2:
            filtro_sent = st.multiselect("Filtrar por Sentimiento:", df['Sentimiento_Predominante'].unique())
        with c3:
            filtro_prio = st.multiselect("Filtrar por Prioridad:", df['Prioridad'].unique())
        
        df_f = df.copy()
        if filtro_cat: df_f = df_f[df_f['Clasificacion_Incidencia'].isin(filtro_cat)]
        if filtro_sent: df_f = df_f[df_f['Sentimiento_Predominante'].isin(filtro_sent)]
        if filtro_prio: df_f = df_f[df_f['Prioridad'].isin(filtro_prio)]
        
        df_f['Caso_ID'] = range(1, len(df_f) + 1)
        st.dataframe(df_f[['Caso_ID', 'Clasificacion_Incidencia', 'Sentimiento_Predominante', 'Prioridad']], use_container_width=True)

        st.divider()

        st.header("Análisis Estadístico y Visualización")
        g1, g2, g3 = st.columns(3)
        
        with g1:
            fig_inc = px.pie(df, names='Clasificacion_Incidencia', title="Distribución de Incidencias", hole=0.4)
            st.plotly_chart(fig_inc, use_container_width=True)
        with g2:
            fig_sent = px.pie(df, names='Sentimiento_Predominante', title="Sentimiento Predominante", 
                              color='Sentimiento_Predominante',
                              color_discrete_map={'NEG': '#EF553B', 'NEU': '#636EFA', 'POS': '#00CC96'},
                              hole=0.4)
            st.plotly_chart(fig_sent, use_container_width=True)
        with g3:
            fig_prio = px.pie(df, names='Prioridad', title="Niveles de Prioridad", hole=0.4)
            st.plotly_chart(fig_prio, use_container_width=True)

        st.divider()

        st.header("Análisis Comparativo e Insights Automáticos")
        def generar_insights(df):
            stats = df.groupby('Clasificacion_Incidencia')['Sentimiento_Predominante'].value_counts(normalize=True).unstack(fill_value=0)
            if 'NEG' in stats.columns:
                peor_cat = stats['NEG'].idxmax()
                peor_val = stats['NEG'].max()
                mejor_cat = stats['NEG'].idxmin()
                mejor_val = stats['NEG'].min()
                diff = ((peor_val / mejor_val) - 1) * 100 if mejor_val > 0 else 100
                st.write(f"Análisis de lenguaje: Los clientes con incidencia de tipo **{peor_cat}** presentan el mayor índice de negatividad ({peor_val:.1%}).")
                st.write(f"Comparado con **{mejor_cat}**, son un **{diff:.1f}%** más propensos a utilizar un lenguaje negativo.")
            
            stats_prio = df.groupby('Clasificacion_Incidencia')['Prioridad'].value_counts(normalize=True).unstack(fill_value=0)
            if 'ALTA (Urgente)' in stats_prio.columns:
                max_urg = stats_prio['ALTA (Urgente)'].idxmax()
                st.write(f"Urgencia Crítica: La categoría **{max_urg}** requiere atención prioritaria, con un **{stats_prio['ALTA (Urgente)'].max():.1%}** de casos urgentes.")

        generar_insights(df)

        st.divider()

        st.header("Explorador por Tipo de Incidencia")
        cat_sel = st.selectbox("Seleccione categoría para detalle profundo:", df['Clasificacion_Incidencia'].unique())
        if cat_sel:
            df_cat = df[df['Clasificacion_Incidencia'] == cat_sel]
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Casos Totales", len(df_cat))
            
            urg_media = len(df_cat[df_cat['Prioridad'] == 'ALTA (Urgente)'])/len(df_cat)
            sc2.metric("Urgencia Media", f"{urg_media:.1%}")
            
            neg_val = len(df_cat[df_cat['Sentimiento_Predominante'] == 'NEG'])/len(df_cat)
            sc3.metric("Indice de Negatividad", f"{neg_val:.1%}")

        st.divider()
        st.info("El visor de conversaciones individuales ha sido deshabilitado para proteger la privacidad de los clientes en la versión pública.")
    else:
        st.warning("No se han encontrado datos procesados.")

with tab2:
    st.header("Métricas de Rendimiento y Comparativa de Modelos")
    
    if comparativa:
        resultados = comparativa['resultados']
        labels = comparativa['labels']
        
        # Resumen de métricas en tabla
        res_list = []
        for nombre, met in resultados.items():
            res_list.append({
                "Modelo": nombre,
                "Accuracy": met['accuracy'],
                "Recall (Macro)": met['recall'],
                "Word2Vec": "Sí" if met['usando_w2v'] else "No"
            })
        
        df_res = pd.DataFrame(res_list).sort_values(by="Accuracy", ascending=False)
        
        st.subheader("Comparativa de Escenarios (Con vs Sin Word2Vec)")
        st.table(df_res.style.format({"Accuracy": "{:.2%}", "Recall (Macro)": "{:.2%}"}))
        
        best_model = df_res.iloc[0]['Modelo']
        st.success(f"**Análisis de Resultados:** El modelo con mejor desempeño es **{best_model}**. "
                   "La elección se basa en un equilibrio entre la precisión global (Accuracy) y la capacidad de identificar "
                   "correctamente todas las categorías (Recall), especialmente las minoritarias como 'Pedidos Duplicados'.")

        st.divider()
        st.header("Optimización por Categoría Crítica")
        st.write("¿Hay alguna incidencia que te preocupe especialmente? Selecciona cuál quieres maximizar y te recomendaremos el mejor modelo para ella.")
        
        cat_priorizar = st.selectbox("Selecciona de qué incidencia quieres maximizar la tasa de acierto:", labels)
        
        # Encontrar el mejor modelo para esa categoría
        y_test = np.array(comparativa['y_test'])
        mejor_acc_cat = -1
        modelo_recomendado = list(resultados.keys())[0]
        
        for nombre, met in resultados.items():
            y_pred = np.array(met['y_pred'])
            mask = (y_test == cat_priorizar)
            acc_cat = (y_pred[mask] == cat_priorizar).sum() / mask.sum() if mask.sum() > 0 else 0
            if acc_cat > mejor_acc_cat:
                mejor_acc_cat = acc_cat
                modelo_recomendado = nombre
        
        st.success(f"Para maximizar el acierto en **{cat_priorizar}**, el sistema recomienda usar: **{modelo_recomendado}** (Acierto específico: {mejor_acc_cat:.1%})")

        st.divider()
        
        st.header("Explorador Detallado por Modelo")
        
        # Usamos el modelo recomendado como valor por defecto
        lista_modelos = list(resultados.keys())
        idx_defecto = lista_modelos.index(modelo_recomendado)
        
        modelo_sel = st.selectbox("Analiza el rendimiento profundo de cada modelo:", lista_modelos, index=idx_defecto)
        
        if modelo_sel:
            met = resultados[modelo_sel]
            m1, m2 = st.columns([1, 2])
            
            with m1:
                st.metric("Accuracy", f"{met['accuracy']:.2%}")
                st.metric("Recall", f"{met['recall']:.2%}")
                
                st.write("""
                A veces la exactitud del modelo no lo es todo a la hora de valorarlo. Aquí interviene el término **Recall**, que es básicamente la capacidad que tiene nuestra IA para no "olvidarse" de ningún caso real. 
                
                En nuestro modelo, si por ejemplo tenemos 10 correos sobre un **Pedido Duplicado** y solo detectamos 2, el Recall sería muy bajo. Lo que buscamos con esta métrica es asegurar que, cuando existe una incidencia crítica, el modelo sea capaz de cazarla siempre.
                """)
                
            with m2:
                st.subheader("Matriz de Confusión")
                fig_cm = px.imshow(
                    met['cm'],
                    text_auto=True,
                    x=labels,
                    y=labels,
                    labels=dict(x="Predicción", y="Realidad", color="Casos"),
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            st.divider()
            st.subheader(f"Análisis Exploratorio de Categorías - {modelo_sel}")
            
            # Análisis de errores por categoría
            y_test = np.array(comparativa['y_test'])
            y_pred = np.array(met['y_pred'])
            
            cat_stats = []
            for cat in labels:
                mask = (y_test == cat)
                total = mask.sum()
                correctos = (y_pred[mask] == cat).sum()
                acc_cat = correctos / total if total > 0 else 0
                cat_stats.append({"Categoría": cat, "Ejemplos": total, "Precisión en Cat": acc_cat})
            
            df_cat_stats = pd.DataFrame(cat_stats)
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                fig_cat = px.bar(df_cat_stats, x='Categoría', y='Precisión en Cat', color='Ejemplos',
                                title="Efectividad del Modelo por Tipo de Categoría")
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with c_col2:
                st.write("**Resumen de Estratificación:**")
                st.write(f"Gracias a la estratificación, hemos asegurado que incluso categorías con pocos datos "
                         f"(como los {df_cat_stats[df_cat_stats['Categoría']=='PAQUETE DUPLICADO']['Ejemplos'].values[0]} casos de prueba de Duplicados) "
                         f"estén representadas en el entrenamiento y evaluación.")
                st.dataframe(df_cat_stats, use_container_width=True)

    else:
        st.error("No se han encontrado resultados de la comparativa. Por favor, ejecuta 'src/entrenamiento_avanzado.py'.")