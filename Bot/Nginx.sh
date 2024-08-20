#!/bin/bash

# Update and install packages
apt update && apt install -y nginx gunicorn python3-pip

# Install Flask using pip3
pip3 install flask

# Create Nginx configuration for the app
NGINX_CONF="/etc/nginx/sites-enabled/flask_app"
echo "server {
    listen 8020;

    location / {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host \$host;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}" > $NGINX_CONF

# Remove the default Nginx site configuration
unlink /etc/nginx/sites-enabled/default

# Restart Nginx to apply the changes
systemctl restart nginx

echo "Nginx and Gunicorn have been configured successfully."
