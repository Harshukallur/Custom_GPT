
import os
from app.server import app

print("Current working directory:", os.getcwd())

if __name__ == "__main__":
    print("Flask app is running on http://127.0.0.1:5001/")
    app.run(host="127.0.0.1", port=5001, debug=True)
