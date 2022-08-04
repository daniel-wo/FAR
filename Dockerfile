FROM python:3.8

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get -qq -y install curl ffmpeg libsm6 libxext6 git
#RUN set -xe \
#    && apt-get -y install python3-pip
RUN pip3 install --upgrade pip
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN useradd -m user
RUN mkdir -p /home/user/.ssh
RUN chown -R user:user /home/user/.ssh

COPY . .

CMD [ "python3", "./text-detection.py" ]
