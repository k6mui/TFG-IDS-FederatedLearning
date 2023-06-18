# Define the certificate directory
$CERT_DIR = ".\cache\certificates"

# Create the directories if they do not exist
if (!(Test-Path -path $CERT_DIR)) {
    New-Item -ItemType Directory -Force -Path $CERT_DIR
}

# Generate the root certificate authority key and certificate based on key
openssl genrsa -out $CERT_DIR\ca.key 4096

# Create a new self-signed X.509 certificate using the newly created private key
openssl req -new -x509 -key $CERT_DIR\ca.key -sha256 -subj "/C=DE/ST=HH/O=CA, Inc." -days 365 -out $CERT_DIR\ca.crt

# Generate a new private key for the server
openssl genrsa -out $CERT_DIR\server.key 4096

# Create a signing CSR
openssl req -new -key $CERT_DIR\server.key -out $CERT_DIR\server.csr -config .\certificate.conf

# Generate a certificate for the server
openssl x509 -req -in $CERT_DIR\server.csr -CA $CERT_DIR\ca.crt -CAkey $CERT_DIR\ca.key -CAcreateserial -out $CERT_DIR\server.pem -days 365 -sha256 -extfile .\certificate.conf -extensions req_ext
