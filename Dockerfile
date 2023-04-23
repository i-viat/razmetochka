FROM python:3.7-buster 
RUN mkdir /opt/app
RUN mkdir /opt/app/razmetochka
WORKDIR /opt/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /opt/app/razmetochka
WORKDIR /opt/app/razmetochka
RUN apt update && apt install dumb-init
EXPOSE 8601
ENTRYPOINT dumb-init streamlit run markup_gui.py --server.port 8601
