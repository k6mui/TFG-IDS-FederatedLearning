# Windows version, this script will generate all certificates if ca.crt does not exist

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $MyInvocation.MyCommand.Path)

$CA_PASSWORD = "notsafe"
$CERT_DIR = ".cache\certificates"

# Generate directories if not exists
New-Item -ItemType Directory -Force -Path $CERT_DIR

# Removing all files from CERT_DIR
Remove-Item -Force -Path "$CERT_DIR\*"

# Generate the root certificate authority key and certificate based on key
& openssl genrsa -out "$CERT_DIR\ca.key" 4096
& openssl req -new -x509 -key "$CERT_DIR\ca.key" -sha256 -subj "/C=DE/ST=HH/O=CA, Inc." -days 365 -out "$CERT_DIR\ca.crt"

# Generate a new private key for the server
& openssl genrsa -out "$CERT_DIR\server.key" 4096

# Create a signing CSR
& openssl req -new -key "$CERT_DIR\server.key" -out "$CERT_DIR\server.csr" -config ".\certificates\certificate.conf"

# Generate a certificate for the server
& openssl x509 -req -in "$CERT_DIR\server.csr" -CA "$CERT_DIR\ca.crt" -CAkey "$CERT_DIR\ca.key" -CAcreateserial -out "$CERT_DIR\server.pem" -days 365 -sha256 -extfile ".\certificates\certificate.conf" -extensions req_ext
