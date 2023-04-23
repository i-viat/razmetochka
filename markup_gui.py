import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
import seaborn as sns
import matplotlib.ticker as ticker
import os
import shutil

from lib.advisor import Advisor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from dotenv import dotenv_values


               
def annotate_test(label):
    if uploaded_file is not None:
        st.session_state.annotations_test[st.session_state.current_test_file] = label
    st.session_state.annotated_test = pd.DataFrame(list(st.session_state.annotations_test.items()), columns = [str(select_col), 'class'])
    if st.session_state.test_hist.empty == False:
        st.session_state.annotated_test = st.session_state.test_hist.append(st.session_state.annotated_test).reset_index(drop=True) #
    st.session_state.test_sample = st.session_state.test_sample[~st.session_state.test_sample.isin([st.session_state.current_test_file])]
    if st.session_state.test_sample.empty == False:
        st.session_state.current_test_file = random.choice(st.session_state.test_sample.tolist())  

              
def annotate(label):
    st.session_state.annotations[st.session_state.current_file] = label  
    st.session_state.files = st.session_state.files[~st.session_state.files.isin([st.session_state.current_file])]
    if st.session_state.files.empty == False:
        st.session_state.current_file = random.choice(st.session_state.files.tolist())

        
def change_marked_value(df):
    with st.expander('Изменить значение рамзетки'):
        raw = st.text_input('Номер строки')
        new_class = st.selectbox('Выберите класс', class_list)
        if st.button('Изменить'):
            df.iloc[[int(raw)],[1]] = new_class     
    
    
def adjust_chunk_size():
    with st.sidebar.expander('Изменить размер чанка'):
        incr_decr_tr = st.selectbox('', ('Увеличить', 'Уменьшить'))
        num_fill_tr = st.text_input('На сколько заполнений?')
        
        if st.button('Изменить', key = 'add_chunk'): 
            if incr_decr_tr == 'Уменьшить':
                st.session_state.chunk_size = st.session_state.chunk_size - int(num_fill_tr) 
                
                excess_chunk_sample = st.session_state.files.sample(n = int(num_fill_tr),  replace = False)
                st.session_state.files = st.session_state.files.drop(excess_chunk_sample.index).reset_index(drop = True)
                excess_chunk_sample = pd.merge(st.session_state.raw_data, excess_chunk_sample, on=str(select_col), how='right').drop_duplicates()
                st.session_state.input_df = st.session_state.input_df.append(excess_chunk_sample).reset_index(drop = True)
                st.session_state.current_file = random.choice(st.session_state.files.tolist())
                              
            else:
                st.session_state.chunk_size = st.session_state.chunk_size + int(num_fill_tr)               
                add_chunk = st.session_state.input_df[select_col].sample(n = int(num_fill_tr), replace = False)
                st.session_state.input_df = st.session_state.input_df.drop(add_chunk.index).reset_index(drop = True)
                st.session_state.files = st.session_state.files.append(add_chunk).reset_index(drop = True)
                st.session_state.current_file = random.choice(st.session_state.files.tolist())
                    
                    
def adjust_test_size():
    with st.sidebar.expander('Изменить размер тестовой выборки'):
        incr_decr = st.selectbox('', ('Увеличить', 'Уменьшить'), key = 'add_test')
        num_fill = st.text_input('На сколько заполнений?', key = 'ad_test1')
        
        if st.button('Изменить', key = 'ad_test2'):           
            if incr_decr == 'Уменьшить':
                excess_test_sample = st.session_state.test_sample.sample(n = int(num_fill), replace = False)
                st.session_state.test_sample = st.session_state.test_sample.drop(excess_test_sample.index).reset_index(drop = True)
                excess_test_sample = pd.merge(st.session_state.raw_data, excess_test_sample, on=str(select_col), how='right').drop_duplicates()
                st.session_state.input_df = st.session_state.input_df.append(excess_test_sample).reset_index(drop = True)
                st.session_state.current_test_file = random.choice(st.session_state.test_sample.tolist())
            
            else:
                add_test = st.session_state.input_df[select_col].sample(n = int(num_fill), replace = False)
                st.session_state.input_df = st.session_state.input_df.drop(add_test.index).reset_index(drop = True)
                st.session_state.test_sample = st.session_state.test_sample.append(add_test).reset_index(drop = True)
                st.session_state.current_test_file = random.choice(st.session_state.test_sample.tolist())

         
def concat_to_export(label1, inner_df, label2 = "", inner_df2 = "", label3 = "", inner_df3 = ""):
    inner_df['type_sample'] = label1
    st.session_state.upload_df['index_orig'] = st.session_state.upload_df.index
    inner_df = pd.merge(st.session_state.upload_df, inner_df, on=str(select_col), how='right').drop_duplicates()
    out_df = st.session_state.upload_df.drop(inner_df['index_orig'])
    
    if label2 != "":
        inner_df2['type_sample'] = label2
        inner_df2 = pd.merge(st.session_state.upload_df, inner_df2, on=str(select_col), how='right').drop_duplicates()
        out_df = out_df.drop(inner_df2['index_orig'])
        
        if label3 != "":
            inner_df3['type_sample'] = label3
            inner_df3 = pd.merge(st.session_state.upload_df, inner_df3, on=str(select_col), how='right').drop_duplicates()
            out_df = out_df.drop(inner_df3['index_orig'])
            out_df = inner_df.append([inner_df2, inner_df3, out_df]).reset_index(drop = True)  
            
        else:
            out_df = inner_df.append([inner_df2, out_df]).reset_index(drop = True)
    
    else:
        out_df = inner_df.append(out_df).reset_index(drop = True)
    
    out_df = out_df.drop(['index_orig'], axis=1)
    return out_df


def draw_metrics():
    with st.expander('Динамика метрик качества на валидационной выборке'): 
        if len(st.session_state.chunk_len) > 1:
            fig, ax = plt.subplots()
            sns.set_theme()
            ax = sns.lineplot(data = st.session_state.metrics_sum, ax = ax)
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(st.session_state.chunk_size))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.set_ylim(0, 1)
            ax.set(xlabel='Размечено заполнений')
            st.pyplot(fig)             
          

def model_learn(marked_chunk, val_sam, sample_type):  
    if __name__ == '__main__':
        
        if page == 'Новый проект (по умолчанию)' or st.session_state.annotated_df.empty: # or = сценарии 2 и 3 в режиме продолжения
            advisor = Advisor(
                vectorizer=TfidfVectorizer(max_features=10000),
                model=SVC(random_state=1234) 
            )    
            
        else:
            advisor = model_load

        marked_chunk.columns = [str(select_col), 'class']
        val_sam.columns = [str(select_col), 'class']       
        
        advisor.fit_vectorizer(marked_chunk[str(select_col)])
            
        try:
            advisor.fit_model(marked_chunk[str(select_col)], marked_chunk['class'])
     
        except ValueError:
            print('Разметьте больше чанков!')

        score = advisor.classification_report(val_sam[str(select_col)], val_sam['class'])
        score_df = pd.DataFrame(score)
        
        report = pd.DataFrame(score).transpose()
        
        if sample_type == "valid":
            st.write("### Качество модели на валидационной выборке") 
        
        else:
            st.write("### Качество модели на тестовой выборке")
        
        st.table(report)
        
        score_df = score_df.iloc[[2], 2:5]
        score_df = score_df.add_prefix('f1_')
        st.session_state.metrics_sum = pd.concat([st.session_state.metrics_sum, score_df])
        st.session_state.chunk_len.append(len(marked_chunk))
        st.session_state.metrics_sum["chunk"] = st.session_state.chunk_len
        st.session_state.metrics_sum = st.session_state.metrics_sum.set_index('chunk')
        
        return advisor
                  
       
def markup_func(val_samp, test_samp):
    adjust_chunk_size()
    col1.write(st.session_state.current_file)

    with col2:
        if st.session_state.input_df.empty == False:
            st.write(
                "Размечено:",
                st.session_state.chunk_size - len(st.session_state.files),
                "– Осталось:",
                len(st.session_state.files),
            )
            for value in class_list:
                st.button(value, on_click = annotate, args=(value,))   
            
            st.write("### Аннотация")
            
            if st.session_state.annotated_df.empty: # для нового проекта
                annotated = pd.DataFrame(list(st.session_state.annotations.items()), columns = [str(select_col), 'class'])
                change_marked_value(annotated)
                st.dataframe(annotated)
                
            else: # для продолжения
                annotated = pd.DataFrame(list(st.session_state.annotations.items()), columns = [str(select_col), 'class'])
                annotated = st.session_state.annotated_df.append(annotated).reset_index(drop=True)
                change_marked_value(annotated)
                st.dataframe(annotated)                                   
            
            st.write(
            "Неразмеченных заполнений в корпусе:",
            len(st.session_state.input_df)
            ) 
            
            if st.session_state.files.empty:
                st.success(
                f"🎈 Готово! {len(st.session_state.annotations)} заполнений размечено"
                )
                
                st.session_state.model = model_learn(annotated, val_samp, "valid")                
                st.session_state.files = st.session_state.model.predict_proba(st.session_state.input_df[select_col])
                st.session_state.files = st.session_state.files.sort_values(by='proba').head(st.session_state.chunk_size).reset_index(drop = True) 
                st.session_state.input_df = st.session_state.input_df.drop(st.session_state.files.index).reset_index(drop = True)
                st.session_state.files = st.session_state.files[select_col]
                st.session_state.current_file = random.choice(st.session_state.files.tolist())
                
                st.button('Продолжить разметку (вывести следующий чанк)')
                           
            if st.checkbox("Завершить разметку:"):
                download = st.radio('', ('Продолжить позже', 'Разметить весь корпус'))

                if download == 'Продолжить позже': 
                    annotated.columns = [str(select_col), 'class']
                    output_df = concat_to_export('train', annotated, 'valid', val_samp, 'test', test_samp)
                    st.download_button("Выгрузить корпус",  output_df.to_csv(index=False).encode('utf-8'), file_name = "train_marked.csv")
                    
                    if st.session_state.model != "": 
                        output_model = pickle.dumps(st.session_state.model, protocol=None)
                        st.download_button('Выгрузить модель', output_model, file_name = "model.bin")                

                else: 
                    annotated.columns = [str(select_col), 'class']
                    st.session_state.model = model_learn(annotated, test_samp, "test")
                    output_df = concat_to_export('train', annotated, 'valid', val_samp, 'test', test_samp)
                    output_df1 = output_df[output_df['class'].isnull()]
                    output_df = output_df.dropna()
                    output_df1['class'] = st.session_state.model.predict(output_df1[select_col])
                    output_df = output_df.append(output_df1).drop(['type_sample'], axis=1)
                    st.download_button("Выгрузить корпус",  output_df.to_csv(index=False).encode('utf-8'), file_name = "corpus_marked.csv")

                    output_model = pickle.dumps(st.session_state.model, protocol=None)
                    st.download_button('Выгрузить модель', output_model, file_name = "model.bin")
                
        else:
            st.success(
                f"🎈 Рамзетка корпуса завершена! Всего размечено {len(st.session_state.annotations)} заполнений"
            )
            obj = model_learn(annotated, test_samp)
            annotated.columns = [str(select_col), 'class']
            output_df = concat_to_export('train', annotated, 'valid', val_samp, 'test', test_samp)
            
            st.download_button("Выгрузить размеченный корпус",  output_df.to_csv(index=False).encode('utf-8'), file_name = "corpus_marked.csv")
            
            output_model = pickle.dumps(st.session_state.model, protocol=None)
            st.download_button('Выгрузить модель', output_model, file_name = "model.bin")      
        
      
def markup_test():
    if st.session_state.test_sample.empty == False:
        adjust_test_size()
        col1.write(st.session_state.current_test_file)
        
        with col2:
            st.write(
                "Размечено:",
                len(st.session_state.annotations_test),
                "– Осталось:",
                len(st.session_state.test_sample),
            )
            for value in class_list:
                st.button(value, on_click = annotate_test, args=(value,))

            st.write("### Аннотация")
            
            
            if st.session_state.test_hist.empty: # для нового проекта
                st.session_state.annotated_test = pd.DataFrame(list(st.session_state.annotations_test.items()), columns = [str(select_col), 'class'])
                change_marked_value(st.session_state.annotated_test)
                st.dataframe(st.session_state.annotated_test)                

            else: # для продолжения 
                if st.session_state.annotated_test.empty:
                    st.session_state.annotated_test = st.session_state.test_hist.append(st.session_state.annotated_test).reset_index(drop=True) 
                change_marked_value(st.session_state.annotated_test)
                st.dataframe(st.session_state.annotated_test)
                
            st.write(
            "Неразмеченных заполнений в корпусе:",
            len(st.session_state.input_df)
            )
        
        output_df = concat_to_export('test', st.session_state.annotated_test)
        st.download_button("Завершить разметку (выгрузить размеченный тест)",  output_df.to_csv(index=False).encode('utf-8'), file_name = "test_marked.csv")                          
    
    else:
        if st.session_state.annotations == {}:
            st.success(
                    f"🎈 Рамзетка тестовой и валидационной выборки завершена! Всего размечено {len(st.session_state.annotations_test)} заполнений"
                )

            if st.session_state.val_sample.empty:
                st.session_state.val_sample = st.session_state.annotated_test.sample(frac = 0.5) 
                st.session_state.test_out = st.session_state.annotated_test.drop(st.session_state.val_sample.index).reset_index(drop = True)
                st.session_state.val_sample = st.session_state.val_sample.reset_index(drop = True)
                
            annotated_test = st.session_state.test_out
            valid_sample = st.session_state.val_sample
            output_df = concat_to_export('test', annotated_test, 'valid', valid_sample) 
            st.download_button("Завершить разметку (выгрузить размеченный тест)",  output_df.to_csv(index=False).encode('utf-8'), file_name = "testval_marked.csv")            

        if st.checkbox('Приступить к разметке чанков обучающей выборки'):
            with st.expander("Посмотреть тестовую выборку"):
                st.table(st.session_state.test_out[[str(select_col), 'class']])
            with st.expander("Посмотреть вал выборку"):
                st.table(st.session_state.val_sample[[str(select_col), 'class']])

            valid_sample = st.session_state.val_sample[[str(select_col), 'class']]
            annotated_test = st.session_state.test_out[[str(select_col), 'class']]
            markup_func(valid_sample, annotated_test)
    
 

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}d
</style> """, unsafe_allow_html=True)

if "entkey" not in st.session_state:
    st.session_state.entkey = ""

if st.session_state.entkey == "":
    st.session_state.placeholder1 = st.sidebar.empty()
    st.session_state.placeholder2 = st.sidebar.empty()
    st.session_state.placeholder3 = st.sidebar.empty()
    st.session_state.placeholder1.title('Авторизация')
    username = st.session_state.placeholder2.text_input('Логин')
    st.session_state.username = username
    password = st.session_state.placeholder3.text_input('Пароль', type='password')
    st.session_state.password = password
        
if st.session_state.username == 'cpur_user' and st.session_state.password == '6DTE3Bkeho2m':
    st.session_state.placeholder1.empty()
    st.session_state.placeholder2.empty()
    st.session_state.placeholder3.empty()
    st.session_state.entkey = "something"

    page = st.sidebar.radio('Хотите продолжить работу?', ('Новый проект (по умолчанию)', 'Продолжить работу'))

    if "test_sample" and "current_test_file" and "files" and "current_file" and "chunk_size" not in st.session_state:
        st.session_state.test_sample = pd.DataFrame()
        st.session_state.current_test_file = ''
        st.session_state.files = pd.DataFrame()
        st.session_state.current_file = ''
        st.session_state.chunk_size = ''

    if page == 'Новый проект (по умолчанию)':
        st.sidebar.header('Задайте параметры разметки')
        uploaded_file = st.sidebar.file_uploader("Загрузите корпус текстов (в формате csv):", type=["csv"])

        if uploaded_file is not None:
            upload_df = pd.read_csv(uploaded_file)
            select_col = st.sidebar.selectbox('Выберите колонку датафрейма для разметки', upload_df.columns)
            raw_data = upload_df.drop_duplicates(subset=[select_col]).reset_index(drop = True)
            test_val_size = st.sidebar.number_input('Какой процент корпуса вы готовы отдать под тестовую и валидационную выборки?', min_value=20.0, max_value=30.0, value=20.0, step=1.0)
            st.sidebar.markdown(f'<p style="background-color:pink;font-size:15px;border-radius:2%;"> Рекомендованный размер: <strong>>20% </strong> корпуса</p>', unsafe_allow_html=True)

            class_num = st.sidebar.number_input('Сколько классов вы хотите выделить?', min_value=2, max_value=10, value=2, step=1)
            i = 1
            class_list = []
            keys = []
            key = 'a'
            while i <= class_num:
                class_list.append(st.sidebar.text_input("Введите обозначение класса " + str(i), key = key))
                keys.append(i)
                i += 1
                key *= 2 

            chunk_size = st.sidebar.number_input('Сколько заполнений должен содержать один чанк обучающей выборки?', min_value=500, max_value=5000, value=500, step=50)
            chunk = raw_data[select_col].sample(n = chunk_size).reset_index(drop = True) 
            input_df = raw_data.drop(chunk.index).reset_index(drop = True)
            #mix_words = st.sidebar.selectbox('Изменится ли смысл элементов корпуса, если в них перемешать слова?', ('Да', 'Нет'))

            if "upload_df" and "annotations" and "annotated_df" and "model" not in st.session_state:
                st.session_state.upload_df = upload_df
                st.session_state.annotations = {}
                st.session_state.raw_data = raw_data
                st.session_state.annotated_df = pd.DataFrame()
                st.session_state.model = ""        

            if "val_sample" and "test_out" and "annotations_test" and "current_test_file" and "annotated_test" and "metrics_sum" and "chunk_len" and "test_hist" not in st.session_state:
                st.session_state.val_sample = pd.DataFrame()
                st.session_state.test_out = pd.DataFrame()
                st.session_state.annotations_test = {}
                st.session_state.annotated_test = pd.DataFrame()
                st.session_state.metrics_sum = pd.DataFrame() 
                st.session_state.chunk_len = []
                st.session_state.test_hist = pd.DataFrame() 


            if st.sidebar.checkbox('Приступить к разметке тестовой выборки') == False:
                st.session_state.test_sample = input_df[select_col].sample(frac = test_val_size/100)
                input_df = input_df.drop(st.session_state.test_sample.index).reset_index(drop = True)
                st.session_state.input_df = input_df
                st.session_state.raw_data = raw_data
                st.session_state.test_sample = st.session_state.test_sample.reset_index(drop = True)
                st.session_state.current_test_file = st.session_state.test_sample.iloc[0]
                st.session_state.files = chunk
                st.session_state.current_file = chunk.iloc[0]
                st.session_state.chunk_size = chunk_size

            else:
                st.write("")
                col1, col2 = st.columns(2)
                markup_test()     
                draw_metrics()       

    else:
        st.sidebar.header('Задайте параметры разметки')
        uploaded_file = st.sidebar.file_uploader("Загрузите размеченный корпус текстов (в формате csv):", type=["csv"])
        if uploaded_file is not None: 
            upload_df = pd.read_csv(uploaded_file)
            select_col = st.sidebar.selectbox('Выберите колонку датафрейма для разметки', upload_df.columns)
            raw_data = upload_df.drop_duplicates(subset=[select_col]).reset_index(drop = True) #
            annotated_df = raw_data.dropna()
            raw_data = raw_data.drop(['class', 'type_sample'], axis=1)
            input_df = raw_data.drop(annotated_df.index).reset_index(drop = True)
            annotated_train = annotated_df[annotated_df['type_sample'] == 'train'].reset_index(drop=True)
            samples_needed = {'train', 'test', 'valid'}
            samples_present = set(annotated_df['type_sample'])

            if samples_needed.issubset(samples_present) == False:
                missed_samp = samples_needed - samples_present

                if 'train' in missed_samp:
                    st.sidebar.warning(f'Предупреждение: в вашем корпусе не размечена обучающая выборка')

                if 'valid' in missed_samp:
                    st.sidebar.warning(f'Предупреждение: в вашем корпусе не размечена валидационная выборка')
                    valid_sample = None
                else:
                    valid_sample = annotated_df[annotated_df['type_sample'] == 'valid'].reset_index(drop = True)
                    valid_sample = valid_sample[[str(select_col), 'class']]

                if 'test' in missed_samp:
                    st.sidebar.warning(f'Предупреждение: в вашем корпусе не размечена тестовая выборка')
                else:
                    annotated_test = annotated_df[annotated_df['type_sample'] == 'test'].reset_index(drop = True)
                    annotated_test = annotated_test[[str(select_col), 'class']]

            else:
                valid_sample = annotated_df[annotated_df['type_sample'] == 'valid'].reset_index(drop = True)
                valid_sample = valid_sample[[str(select_col), 'class']]
                annotated_test = annotated_df[annotated_df['type_sample'] == 'test'].reset_index(drop = True)
                annotated_test = annotated_test[[str(select_col), 'class']]                        

            chunk_size = st.sidebar.number_input('Сколько заполнений должен содержать один чанк обучающей выборки?', min_value=500, max_value=5000, value=500, step=50)
            chunk = input_df[select_col].sample(n = chunk_size).reset_index(drop = True)
            input_df = input_df.drop(chunk.index).reset_index(drop = True)
            class_list = list(annotated_df['class'].unique())
            class_list = [str(x) for x in class_list]
            class_list = [x for x in class_list if x != 'nan'] 
            if len(class_list) > 1:
                st.sidebar.info(f'Загруженный корпус содержит следующие классы: {str(class_list)[1:-1]}')  
            else:
                st.sidebar.warning('Предупреждение: в вашем корпусе выделен только один класс заполнений')

            class_num = st.sidebar.number_input('Сколько классов вы хотите добавить?', min_value=0, max_value=10, value=0, step=1)
            i = 1
            keys = []
            key = 'a'
            while i <= class_num:
                class_list.append(st.sidebar.text_input("Введите обозначение класса " + str(i), key = key))
                keys.append(i)
                i += 1
                key *= 2 
            #mix_words = st.sidebar.selectbox('Если перемешать слова в элементах корпуса, их смысл изменится?', ('Да', 'Нет'))

            if "upload_df" and "annotations" and "annotated_df" not in st.session_state:
                st.session_state.upload_df = upload_df.drop(columns=['type_sample', 'class'])
                st.session_state.annotations = {} 
                st.session_state.raw_data = raw_data
                st.session_state.annotated_df = annotated_train[[str(select_col), 'class']]

            if "metrics_sum" and "val_sample" and "test_out" and "chunk_len" not in st.session_state:
                st.session_state.val_sample = pd.DataFrame()
                st.session_state.test_out = pd.DataFrame()
                st.session_state.metrics_sum = pd.DataFrame() 
                st.session_state.chunk_len = []

            if "annotations_test" and "test_hist" not in st.session_state:
                st.session_state.annotations_test = {}
                st.session_state.annotated_test = pd.DataFrame()
                st.session_state.test_hist = annotated_test

        if uploaded_file is not None: 

            if st.session_state.annotated_df.empty is False:   # сценарий 1 - есть train
                uploaded_model = st.sidebar.file_uploader("Загрузите дамп модели (в формате bin):", type=["bin"])   

                if uploaded_model is not None:
                    os.makedirs('tempDir', exist_ok=True)
                    with open(os.path.join("tempDir", uploaded_model.name), "wb") as f: 
                        f.write(uploaded_model.getbuffer()) 
                    model_load = pickle.load(open(os.path.join("tempDir", uploaded_model.name), 'rb'))
                    shutil.rmtree('tempDir/', ignore_errors=True)   

                    if "model" not in st.session_state:
                        st.session_state.model = model_load

                    if st.sidebar.checkbox('Продолжить разметку обучающей выборки') == False:
                        st.session_state.annotated_df = annotated_train[[str(select_col), 'class']]
                        st.session_state.files = chunk
                        st.session_state.current_file = chunk.iloc[0]
                        st.session_state.chunk_size = chunk_size
                        st.session_state.input_df = input_df

                    else:    
                        st.write("")
                        col1, col2 = st.columns(2)
                        markup_func(valid_sample, annotated_test)

                        with st.expander("Посмотреть тестовую выборку"):
                            st.table(annotated_test[[str(select_col), 'class']])
                        with st.expander("Посмотреть вал выборку"):
                            st.table(valid_sample[[str(select_col), 'class']])                                
                        draw_metrics()

            else:

                if valid_sample is not None: # сценарий 2 - есть test и val
                    if "model" not in st.session_state:
                        st.session_state.model = ""
                        st.session_state.annotated_df = pd.DataFrame()

                    if st.sidebar.checkbox('Приступить к разметке обучающей выборки') == False:
                        st.session_state.files = chunk
                        st.session_state.current_file = chunk.iloc[0]
                        st.session_state.chunk_size = chunk_size
                        st.session_state.input_df = input_df

                    else:    
                        st.write("")
                        col1, col2 = st.columns(2)
                        markup_func(valid_sample, annotated_test)

                        with st.expander("Посмотреть тестовую выборку"):
                            st.table(annotated_test[[str(select_col), 'class']])
                        with st.expander("Посмотреть вал выборку"):
                            st.table(valid_sample[[str(select_col), 'class']])
                        draw_metrics()  

                else:    # сценарий 3 - есть только тест (i.e. надо доразметить)

                    if "model" not in st.session_state:
                        st.session_state.model = ""
                        st.session_state.annotated_df = pd.DataFrame()

                    st.sidebar.markdown("Какой процент корпуса вы готовы отдать под тестовую и валидационную выборки?")
                    minimum_value = 20.0 - (len(st.session_state.test_hist)/len(raw_data)*100)
                    test_val_size = st.sidebar.number_input('(с учётом размеченных заполнений)', min_value=minimum_value, max_value=30.0, value=minimum_value, step=1.0)
                    st.sidebar.info(f'Тестовая и валид-ая выборки сейчас составляют ~ {round(len(st.session_state.test_hist)/len(raw_data)*100, 2)} % корпуса')
                    st.sidebar.markdown(f'<p style="background-color:pink;font-size:15px;border-radius:2%;"> Рекомендованный совокупный размер: <strong>>20% </strong> корпуса</p>', unsafe_allow_html=True)

                    if st.sidebar.checkbox('Продолжить разметку тестовой и валидационной выборки') == False:
                        st.session_state.raw_data = raw_data
                        st.session_state.files = chunk
                        st.session_state.current_file = chunk.iloc[0]
                        st.session_state.chunk_size = chunk_size
                        st.session_state.test_sample = input_df[select_col].sample(frac = test_val_size/100)
                        st.session_state.input_df = input_df.drop(st.session_state.test_sample.index).reset_index(drop = True)
                        st.session_state.test_sample = st.session_state.test_sample.reset_index(drop = True)
                        st.session_state.test_hist = annotated_df[annotated_df['type_sample'] == 'test'].reset_index(drop = True)
                        st.session_state.test_hist = st.session_state.test_hist[[str(select_col), 'class']]
                        st.session_state.current_test_file = st.session_state.test_sample.iloc[0]

                    else:
                        st.write("")
                        col1, col2 = st.columns(2)
                        markup_test()  
                        draw_metrics()

else:
    if username and password != "":
        st.sidebar.error("Неправильный логин или пароль")
