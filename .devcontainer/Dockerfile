FROM ubuntu:jammy

RUN apt-get update && apt-get -y install --no-install-recommends build-essential \
    gcc git gnupg2 openssh-client ca-certificates python3 python3-pip && \
    apt-get clean autoclean && \
    apt-get autoremove

RUN pip3 install --upgrade pip && \
    pip3 install black cryptography joblib matplotlib numpy pandas scikit-learn scipy seaborn && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 

RUN addgroup --gid 1000 vscode
RUN adduser --disabled-password --gecos "" --uid 1000 --gid 1000 vscode
USER vscode
ENV HOME /home/vscode
