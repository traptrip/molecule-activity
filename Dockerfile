FROM tensorflow/tensorflow:latest-gpu


ARG APP_DIR=/app
WORKDIR "$APP_DIR"

RUN apt update && \
    apt install -y libsm6 libxext6 libxrender-dev
RUN apt install -y python3-pip && pip3 install -U pip

COPY requirements.txt $APP_DIR/
RUN pip install --user -r requirements.txt --no-cache
RUN pip install tensorflow_addons
COPY . $APP_DIR/
ENTRYPOINT ["sh", "entrypoint.sh"]
