FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Qt, OpenGL, X11 virtual display, and VNC streaming
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-shape0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    xvfb \
    x11vnc \
    openbox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone NoVNC to map the virtual desktop directly to a web browser port
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc && \
    git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify && \
    ln -s /opt/novnc/vnc.html /opt/novnc/index.html

WORKDIR /app

# Upgrade pip and install requirements
RUN pip3 install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all project code
COPY . .

# Expose port 7860 (Hugging Face default web port)
EXPOSE 7860

ENV DISPLAY=:1
ENV RESOLUTION=1280x800x24

# Set up the startup script to load the virtual display, proxy, and launch main.py
CMD Xvfb :1 -screen 0 $RESOLUTION & \
    sleep 2 && \
    x11vnc -display :1 -nopw -listen localhost -xkb & \
    sleep 2 && \
    /opt/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 0.0.0.0:7860 & \
    sleep 2 && \
    python3 main.py
