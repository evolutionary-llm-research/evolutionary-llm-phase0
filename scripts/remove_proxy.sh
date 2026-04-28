#!/bin/bash
# Skrypt do usuwania zmiennych proxy z plików konfiguracyjnych i środowiska WSL

# Usuń zmienne proxy z bieżącej sesji
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# Usuń proxy z ~/.bashrc ~/.profile ~/.bash_profile
for file in ~/.bashrc ~/.profile ~/.bash_profile; do
    if [ -f "$file" ]; then
        sed -i '/http_proxy/d' "$file"
        sed -i '/https_proxy/d' "$file"
        sed -i '/HTTP_PROXY/d' "$file"
        sed -i '/HTTPS_PROXY/d' "$file"
    fi
done

# Usuń proxy z /etc/environment (wymaga sudo)
if [ -f /etc/environment ]; then
    sudo sed -i '/http_proxy/d' /etc/environment
    sudo sed -i '/https_proxy/d' /etc/environment
    sudo sed -i '/HTTP_PROXY/d' /etc/environment
    sudo sed -i '/HTTPS_PROXY/d' /etc/environment
fi

# Usuń proxy z /etc/apt/apt.conf (jeśli istnieje)
if [ -f /etc/apt/apt.conf ]; then
    sudo sed -i '/Acquire::http::Proxy/d' /etc/apt/apt.conf
    sudo sed -i '/Acquire::https::Proxy/d' /etc/apt/apt.conf
fi

echo "Proxy zostało usunięte z plików konfiguracyjnych. Zrestartuj terminal lub wykonaj: source ~/.bashrc"
