
import sys

import logging

logging.basicConfig(filename="logs_out.log",
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a',
                    encoding='utf-8')

logger = logging.getLogger('model_logger')
logger.setLevel(logging.INFO)


# Формат конфиг файла может быть таким.
# [Model]
# model (По умолчанию t-tech/T-lite-it-1.0)
# max_new_tokens (По умолчанию 512)
# top_k (По умолчанию 50)
# top_p (По умолчанию 0.85)
# temperature (По умолчанию 0.3)
# gpu_memory_utilization (По умолчанию 0.1)
# use_swap_memory (По умолчанию False)
# swap_space_size (По умолчанию 64GiB)

# [Path_to_PDFS]
# path (По умолчанию PDFS)

# [RAG]
# top_k (По умолчанию 8)
# chunk_size (По умолчанию 2000)
# chunk_overlap (По умолчанию 300)

# [BOT]
# token (Токен бота)

# Парсер конфигурационных файлов.
import configparser

config = configparser.ConfigParser()

with open('settings.ini', 'r', encoding='utf-8') as f:
    config.read_file(f)

#config.read("settings.ini")

try:
    token = config['BOT']['token']
except:
    logger.error('Нет токена бота в конфиге, не могу начать работу бота')
    sys.exit('Нет токена бота в конфиге')

# Чтение config файла. Модель
try:
  model = config['Model']['model']
except:
  model = "t-tech/T-lite-it-1.0"
  logger.warning('Не нашёл model в конфиге, использую модель по умолчанию (%s)', model)
try:
  max_new_tokens = int(config['Model']['max_new_tokens'])
except:
  max_new_tokens = 512
  logger.warning('Не нашёл max_new_tokens в конфиге, использую максимальное количество токентов на выход по умолчанию (%s)', str(max_new_tokens))
try:
  top_k = int(config['Model']['top_k'])
except:
  top_k = 50
  logger.warning('Не нашёл tok_k для модели в конфиге, использую top_k модели по умолчанию (%s)', str(top_k))
try:
  top_p = float(config['Model']['top_p'])
except:
  top_p = 0.85
  logger.warning('Не нашёл top_p в конфиге, использую top_p модели по умолчанию (%s)', str(top_p))
try:
  temperature = float(config['Model']['temperature'])
except:
  temperature = 0.35
  logger.warning('Не нашёл temperature в конфиге, использую температуру модели по умолчанию (%s)', str(temperature))
try:
  gpu_memory_utilization = float(config['Model']['gpu_memory_utilization'])
except:
  gpu_memory_utilization = 0.1
  logger.warning('Не нашёл gpu_memory_utilization в конфиге, использую значение использования видеопамяти по умолчанию (%s)', str(gpu_memory_utilization))
try:
  use_swap_memory = config['Model']['use_swap_memory'] == 'True'
except:
  use_swap_memory = False
  logger.warning('Не нашёл use_swap_memory в конфиге, использую %s на параметр подмены видеопамяти', str(use_swap_memory))
try:
  swap_space_size = config['Model']['swap_space_size']
except:
  swap_space_size = "64GiB"
  logger.warning('Не нашёл swap_space_size в конфиге, использую значение размера заменяющей памяти по умолчанию (%s)', swap_space_size)

# Путь до pdf котекста
try:
  pdf_path = config['Path_to_PDFS']['path']
except:
  pdf_path = 'PDFS'
  logger.warning('Не нашёл pdf_path в конфиге, использую расположение по умолчанию (%s)', pdf_path)

# Количество верхних результатов rag берущихся в контекст модели
try:
  rag_top_k = int(config['RAG']['top_k'])
except:
  rag_top_k = 8
  logger.warning('Не нашёл rag_top_k в конфиге, использую top_k для rag по умолчанию (%s)', str(rag_top_k))
try:
  rag_chunk_size = int(config['RAG']['chunk_size'])
except:
  rag_chunk_size = 2000
  logger.warning('Не нашёл rag_chunk_size в конфиге, использую размер чанка по умолчанию (%s)', str(rag_chunk_size))
try:
  rag_chunk_overlap = int(config['RAG']['chunk_overlap'])
except:
  rag_chunk_overlap = 300
  logger.warning('Не нашёл rag_chunk_overlap в конфиге, использую размер пересечения чанков по умолчанию (%s)', str(rag_chunk_overlap))

# Собираем все pdf файлы для RAG

import glob
pdfs = glob.glob(pdf_path + "/*.pdf")
# Конвертер таблиц в pdf файлах, чтобы rag легче по ним проходился.

import pdf_table2json.converter as converter

logger.info('Начинаю конвертировать таблицы из pdf файлов директории в json формат')

jsons = []
for pdf in pdfs:
  logger.info('Конвертирую файл %s', pdf)
  jsons.append(converter.main(pdf))

logger.info('Закончил конверсию pdf файлов')

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.docstore.document import Document

# Собираем все таблицы в один список формата langchain document

json_docs = []
for i, json in enumerate(jsons):
  json_docs.append(Document(page_content=json, metadata={"source" : pdfs[i]}))

# Подчищаю созданный мусор библиотекой pdf_table2json. Так как не хочет она не создавать картинки страниц.

logger.info('Начинаю чистку мусора после библиотеки трансформации pdf таблиц')

import os
dirs_to_delete = []
for pdf in pdfs:
  dirs_to_delete.append(os.path.splitext(os.path.basename(pdf))[0])
  pass

import shutil
for dir in dirs_to_delete:
  logger.info('Удаляю папку %s', pdf_path + '/' + dir)
  shutil.rmtree(pdf_path + '/' + dir)

logger.info('Чистка окончена')

path_to_pdfs = pdf_path + '/'

# Загрузщик pdf файлов подгружает всю директорию, все документы pdf формата.

logger.info('Подгружаю файлы из директории %s', path_to_pdfs)

try:
    loader = DirectoryLoader(path_to_pdfs, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
except Exception as e:
    logger.error("Проблема в загрузке файлов в директорию, сообщение ошибки: %s", repr(e))
    sys.exit("Проблема в загрузке файлов (возможно директория не доступна) " + repr(e))
logger.info('Закончил подгрузку файлов из директории %s', path_to_pdfs)

# Убираем символы перехода на новую строку, для упрощения восприятия текста моделью.

for doc in docs:
  doc.page_content = doc.page_content.replace('\n', '')

for json_doc in json_docs:
  docs.append(json_doc)

# Разбиваем файлы на чанки

text_splitter = RecursiveCharacterTextSplitter(chunk_size=rag_chunk_size, chunk_overlap=rag_chunk_overlap)

logger.info('Начинаю разбивать тексты на чанки (размер чанка: %s; размер пересечение чанков: %s)', rag_chunk_size, rag_chunk_overlap)

all_splits = text_splitter.split_documents(docs)

logger.info('Закончил разбивать тексты')

from langchain_chroma import Chroma


# Подгружаем эмбеддер

from sentence_transformers import SentenceTransformer

embedding_model_name = 'all-MiniLM-L6-v2'

logger.info('Начинаю подгружать модель embedding %s', embedding_model_name)

try:
  embedding_model = SentenceTransformer(embedding_model_name)
except Exception as e:
  logger.error('Не получилось подгрузить модель embedding %s. Сообщение ошибки: %s', embedding_model_name, repr(e) )
  print(repr(e))
  sys.exit('Ошибка в подгрузке модели embeddeding')

logger.info('Закончил подгружать модель embedding %s', embedding_model_name)

# Класс обёртка для совместимости с langchain Chroma

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True).tolist()

    def embed_query(self, query):
        return self.model.encode(query, convert_to_tensor=True).tolist()

embeddings = SentenceTransformerEmbeddings(embedding_model)

from langchain_chroma import Chroma

# Применяем эмбеддинг ко всем чанкам файлов и подгружаем эмбеддинги в память.

logging.info('Начинаю процесс эмбеддинга текстов')

try:
  vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
except Exception as e:
  logger.error('Не получилось применить эмбеддинг к текстам. Сообщение ошибки: %s', repr(e) )
  print(repr(e))
  sys.exit('Ошибка в применении эмбеддинга')

logging.info('Завершил процесс эмбеддинга текстов')

from langchain_core.runnables import RunnableLambda

# Создаём часть пайплайна отвечающую за поиск по файлам.

try:
  retriever = RunnableLambda(vectorstore.similarity_search).bind(k=rag_top_k)  # select top result
except Exception as e:
  logger.error('Не получилось создать retriever часть пайплайна %s', repr(e) )
  print(repr(e))
  sys.exit('Ошибка в создании части пайплайна (retriever)')

from langchain_community.llms import VLLM

# vllm подгружает модель первый раз, потом, если тензоры модели остались на месте, следующий запуск должен просто их загрузить в память и инициализировать модель.
# В зависимости от выбора в конфиг файле, используется либо подмена памяти (При использовании больших моделей можно использовать видео карту
# с меньшей видеопамятью), либо модель запускается без подмены.)

logger.info('Начинаю загружать модель %s. max_new_tokens = %s, top_k = %s, top_p = %s, temperature = %s, use_swap_memory = %s, gpu_memory_utilization = %s, swap_space_size = %s',
                model, str(max_new_tokens), str(top_k), str(top_p), str(temperature),
                  str(use_swap_memory), str(gpu_memory_utilization), str(swap_space_size))
try:
  if use_swap_memory:
    llm = VLLM(
      model=model,
      trust_remote_code=True,  # mandatory for hf models
      max_new_tokens=max_new_tokens, # 512 256 768
      top_k=top_k, # 10 - 200
      top_p=top_p, # 0 - 1
      temperature=temperature, # 0 - 1
      gpu_memory_utilization=gpu_memory_utilization,
      swap_space_size=swap_space_size          # Offload to CPU memory (swap space size) Возможно получится с этим использовать модель на 32 миллиарда параметров.
    )
  else:
    llm = VLLM(
      model=model,
      trust_remote_code=True,  # mandatory for hf models
      max_new_tokens=max_new_tokens, # 512 256 768
      top_k=top_k, # 10 - 200
      top_p=top_p, # 0 - 1
      temperature=temperature, # 0 - 1
      gpu_memory_utilization=gpu_memory_utilization
    )
except Exception as e:
  logger.error('Не получилось подгрузить модель %s. Сообщение ошибки: %s', model, repr(e) )
  print(repr(e))
  sys.exit('Ошибка в подгрузке модели')

logger.info('Закончил загружать модель')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Основной промпт модели

message = """
Ты — информационный чат-бот ЛЭТИ, предназначенный для помощи абитуриентам. Твоя задача — отвечать на вопросы, связанные с поступлением в университет, программами обучения, требованиями для поступления, сроками подачи документов и другими важными вопросами для абитуриентов.

Будь точным и предоставляй актуальную информацию.
Никогда не упоминай конекст в своем ответе.
Если вопрос касается тем, которые не относятся к твоей сфере знаний, объясни это абитуриенту, чтобы он знал, что ты не можешь ответить на этот вопрос.
Отвечай ясно и лаконично, избегай излишних подробностей.
Помни, что ты не используешь приватные или закрытые данные, а только доступный контент.
Если информация не известна или неясна, честно сообщи, что ты не имеешь ответа, и предложи обратиться к другим официальным источникам или лицам.
Пример вопросов, на которые ты должен отвечать:

Как подать заявление в ЛЭТИ?
Какие экзамены нужно сдать для поступления на факультет информационных технологий?
Какой минимальный балл ЕГЭ нужен для поступления на бакалавриат?
Какие документы требуются для поступления в магистратуру?
Когда начинаются вступительные испытания в ЛЭТИ в этом году?

Далее следует вопрос и контекст к вопросу, который ты должна использовать:

Вопрос:
"{question}"

Контекст:
"{context}"
"""

# llm.temperature = 0.35 # 0 - 1
# llm.max_new_tokens = 512 #
# llm.top_k=50 # 10 - 200
# llm.top_p = 0.85 # 0 - 1

prompt = ChatPromptTemplate.from_messages([("human", message)])

# Pipeline langchain, использующий RAG, затем модель.

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

logger.info('Собрал pipeline')

#retriever.invoke('Сколько времени у меня будет на подачу оригиналов документов после этапа приоритетного зачисления?')

#response = rag_chain.invoke("Где можно узнать результаты пройденных вступительных испытаний?")
#print('\n')
#print(response)

#import pandas as pd

#df = pd.read_csv('/content/submission.csv')

#ans = []
#questions = df['question']
#for question in questions:
#  response = rag_chain.invoke(question)
#  ans.append(response)

#res_df = pd.DataFrame({"question": df['question']})
#res_df['answer'] = ans
#res_df.to_csv('result_submission.csv', index=False)


import nest_asyncio
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from datetime import datetime

# Ввод токена бота
TOKEN = token

# Позволить асинхронному коду работать в коллабе
nest_asyncio.apply()

# Обработчик команды /start бота
# Когда пользователь прописывает команду /start бот приветствует пользователя.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Получил команду /start")
    await update.message.reply_text('Здравствуйте, я информационный чат-бот ЛЭТИ, предназначенный для помощи абитуриентам.\nМожете спросить меня базовые вопросы, связанные с поступлением в университет ЛЭТИ.')

# Функция ответа на сообщения пользователя
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    response = rag_chain.invoke(user_message)

    logger.info('Бот на вопрос: %s, ответил: %s', user_message, response)

    await update.message.reply_text(response)

# Главная функция инициализирующая бота
async def main() -> None:
    logger.info("Начало инициализации бота")
    app = ApplicationBuilder().token(TOKEN).build()

    # Добавляем обработчики на команду старта и сообщения от пользователя.
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    logger.info("Начинаю пуллинг сообщений")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    logger.info("Бот активен, ожидает сообщений")
    await asyncio.Future()  # Продолжаем работу бота

# Завершаем работу всех остальных незавершённых асинхронных задач
try:
    loop = asyncio.get_running_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()
    loop.stop()
    loop.run_forever()
    loop.close()
except:
    pass

logger.info("Активируем бота")
# Активируем бота через функцию main
try:
    asyncio.run(main())
except Exception as e:
    logger.error("Ошибка в боте, сообщение ошибки: %s", repr(e) )
    print(repr(e))
    sys.exit("Ошибка в боте")

