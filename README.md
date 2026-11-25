
# **Custom-GPT Chat Application**

### **Prerequisites:**

Before you begin, ensure you have the following installed:

- Python 3.12 or newer (required by `pyproject.toml`)
- A Groq API key

### **Setup Instructions:**

1. Create a Virtual Environment:
   - From the project folder, run:
     ```
     python -m venv venv
     ```

2. Activate the Environment:
   - Windows (PowerShell):
     ```
     .\venv\Scripts\Activate.ps1
     ```
     If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.
   - macOS / Linux:
     ```
     source venv/bin/activate
     ```

3. Install Requirements:
   - Install the necessary dependencies using pip:
     ```
     pip install -r requirements.txt
     ```

4. Add Your API Key:
   - Create a `.env` file in the project folder containing:
     ```
     API_KEY=your_groq_api_key
     ```

5. Run the Application:
   - Start the application by running:
     ```
     python run_local.py
     ```

### **Usage:**

Once the application is running, open your web browser and navigate to `http://127.0.0.1:5001/` to access the chat interface. You can start asking questions, and the AI will respond based on your input.

### **Troubleshooting:**

- **"Sorry, I couldn't get a response at this time."** The app could not reach the Groq API. Check the latest file in `logs/` for the exact error.
- **`CERTIFICATE_VERIFY_FAILED` / self-signed certificate errors:** Common on corporate networks that inspect HTTPS traffic. The app uses `truststore` to trust the operating system's certificate store, so make sure dependencies are installed with `pip install -r requirements.txt`.
