ARG TARGETPLATFORM
FROM ghcr.io/astral-sh/uv:latest AS uv_source

FROM ghcr.io/linuxserver/baseimage-selkies:ubuntunoble

# --- Webtop / Selenium environment setup ---
ENV TITLE="Selenium LLM Engine"
ENV PIXELFLUX_USE_XSHM=0 \
    PIXELFLUX_DISABLE_XSHM=1 \
    PIXELFLUX_NO_XSHM=1 \
    QT_X11_NO_MITSHM=1 \
    DISABLE_XSHM=1 \
    BROWSER=/usr/local/bin/chromium-browser
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Inject uv binary from astral image
COPY --from=uv_source /uv /usr/local/bin/uv

COPY webtop/s6-services/uvicorn /etc/services.d/uvicorn
# Install system packages and browser deps
RUN echo 'Package: snapd' > /etc/apt/preferences.d/no-snap && \
    echo 'Pin: release a=*' >> /etc/apt/preferences.d/no-snap && \
    echo 'Pin-Priority: -10' >> /etc/apt/preferences.d/no-snap && \
    apt-get update && apt-get purge -y snapd && \
    apt-get autoremove -y && rm -rf /snap /var/snap /var/lib/snapd && \
        apt-get install -y --no-install-recommends \
            xz-utils \
      python3 python3-venv python3-pip \
      git curl wget unzip nano vim \
      lsb-release ca-certificates openssl \
      htop net-tools iputils-ping \
      ffmpeg mariadb-client libmariadb3 libmariadb-dev \
      espeak-ng libespeak-ng1 \
      xorg dbus-x11 x11-xserver-utils \
      xfce4 xfce4-goodies xfce4-terminal thunar mousepad ristretto \
      adwaita-icon-theme util-linux dbus-x11 at-spi2-core \
      pulseaudio pulseaudio-utils pavucontrol \
      fonts-liberation libnss3 libxss1 libappindicator3-1 libatk-bridge2.0-0 \
      libgtk-3-0 libgbm-dev libasound2t64 xvfb x11vnc fluxbox novnc python3-websockify \
      ca-certificates && \
    update-ca-certificates --fresh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install gemini-cli (optional helper, same as Synthetic Heart)
RUN pip3 install --no-cache-dir gemini-cli || true

# Install Chromium from Debian repos if not preinstalled
RUN ARCH="${TARGETARCH}" && \
    if [ -z "$ARCH" ]; then ARCH=amd64; fi && \
    apt-get update && \
    apt-get purge -y google-chrome google-chrome-stable || true && \
    apt-get install -y --no-install-recommends debian-archive-keyring && \
    echo "deb [arch=$ARCH signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list.d/debian-chromium.list && \
    echo "deb [arch=$ARCH signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://security.debian.org/debian-security bookworm-security main" >> /etc/apt/sources.list.d/debian-chromium.list && \
    apt-get update && \
    # Hard-pinned to a version compatible with undetected-chromedriver 3.5.5.
    # Bump this when both Chromium AND undetected-chromedriver have been
    # verified together.  Do NOT revert to dynamic candidate resolution.
    CHROMIUM_VERSION="149.0.7827.196-1~deb12u1" && \
    apt-get install -y --no-install-recommends chromium=$CHROMIUM_VERSION chromium-driver=$CHROMIUM_VERSION && \
    apt-mark hold chromium chromium-driver && \
    rm -f /etc/apt/sources.list.d/debian-chromium.list && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    chromium --version || true


# Chromium profile setup (in /config/.config/chromium-synth — matches SyntH)
RUN mkdir -p '/config/.config/chromium-synth' && \
    chown -R abc:abc /config && \
    chmod -R 775 /config && \
    mkdir -p /usr/local/share/applications

RUN cat > /usr/local/share/applications/chromium-synth.desktop <<'EOF'
[Desktop Entry]
Version=1.0
Name=Chromium SyntH
Exec=/usr/bin/chromium --no-sandbox --user-data-dir=/config/.config/chromium-synth %U
Terminal=false
Type=Application
Categories=Network;WebBrowser;
EOF

RUN mkdir -p /config/.local/share/applications && \
    cp /usr/local/share/applications/chromium-synth.desktop /config/.local/share/applications/ && \
    chown -R abc:abc /config/.local

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

ENV SELENIUM_LLM_DB=/app/data/selenium_engine.db
ENV CHROMIUM_HEADLESS=0
ENV PYTHONUNBUFFERED=1

RUN echo xfce4-session > /config/desktop-session
# S6 Websockify
COPY webtop/s6-services/websockify /etc/s6-overlay/s6-rc.d/websockify
RUN chmod +x /etc/s6-overlay/s6-rc.d/websockify/run && \
    echo 'longrun' > /etc/s6-overlay/s6-rc.d/websockify/type && \
    mkdir -p /etc/s6-overlay/s6-rc.d/user/contents.d && \
    echo websockify > /etc/s6-overlay/s6-rc.d/user/contents.d/websockify && \
    chown -R abc:abc /etc/s6-overlay/s6-rc.d/websockify

# Final cleanup
RUN mv /usr/bin/thunar /usr/bin/thunar-real && \
  rm -f /etc/xdg/autostart/xfce4-power-manager.desktop /etc/xdg/autostart/xscreensaver.desktop && \
  rm -rf /tmp/*

