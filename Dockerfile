FROM ubuntu:18.04

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get -qq -y install curl python3.8 ffmpeg libsm6 libxext6 tesseract-ocr git
RUN set -xe \
    && apt-get -y install python3-pip
RUN pip3 install --upgrade pip
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "./text-detection.py" ]
