# Dynamic WSL2 proxy startup script
# Detects current WSL virtual adapter IP and starts proxy on correct address

$wslIP = (Get-NetIPAddress -InterfaceAlias "vEthernet (WSL)" -AddressFamily IPv4).IPAddress
C:\Python\python.exe -m proxy --hostname $wslIP --port 8080 --log-level ERROR
