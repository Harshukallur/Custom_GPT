
import os
import truststore

# Use the OS certificate store so HTTPS works behind corporate SSL inspection
truststore.inject_into_ssl()

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    print("Flask app is running on http://127.0.0.1:5001/")
    app.run(host="127.0.0.1", port=5001, debug=True)
