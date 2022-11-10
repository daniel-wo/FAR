FROM python:3.8


RUN apt-get update
RUN apt-get -qq -y install curl ffmpeg libsm6 libxext6 git git-lfs
#RUN set -xe \
#    && apt-get -y install python3-pip
RUN pip3 install --upgrade pip
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install -U albumentations --no-binary qudida,albumentations
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN useradd -m user
RUN mkdir -p /home/user/.ssh
RUN chown -R user:user /home/user/.ssh

COPY . .


WORKDIR /recognition
ENTRYPOINT ["python3", "automaton_recognizer.py"]
CMD []
