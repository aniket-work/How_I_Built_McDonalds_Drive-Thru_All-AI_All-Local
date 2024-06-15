# How_I_Built_McDonalds_Drive-Thru_All-AI_All-Local


🛡 How I Built McDonald's Drive-Thru: All-AI, All-Local 🛡

**Prerequisites:**

- Python installed on your system.
- A basic understanding of virtual environments and command-line tools.

**Steps:**

1. **Virtual Environment Setup:**

   - Create a dedicated virtual environment for our project:
   
     ```bash
     python -m venv Built_McDonalds_Drive_Thru_ALL_AI_Local 
     ```

   - Activate the environment:
   
     - Windows:
        ```bash
        Built_McDonalds_Drive_Thru_ALL_AI_Local\Scripts\activate
        ```
     - Unix/macOS:
        ```bash
        source Built_McDonalds_Drive_Thru_ALL_AI_Local/bin/activate
        ```

2. **Install Project Dependencies:**

   - Install required packages using `pip`:
   
     ```bash
     pip install -r requirements.txt
     ```

3. **Setup Groq Key:**

   - Obtain your Groq API key from [Groq Console](https://console.groq.com/keys).
   - Set your key in the `.env` file as follows:
     ```plaintext
     GROQ_API_KEY=<YOUR_KEY>
     ```
     
4. **Install and Run Qdrant DB**

    ```plaintext
    C:\Users\worka>docker pull qdrant/qdrant
    docker run -p 6333:6333 qdrant/qdrant

    curl http://localhost:6333/collections
    http://localhost:6333/dashboard
    ```

5. **Install CUDA**

   ```plaintext
   https://developer.nvidia.com/cuda-toolkit-archive
   https://developer.nvidia.com/rdp/cudnn-archive
   ```
6. 

 
