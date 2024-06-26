# How I Built McDonald's Drive-Thru: All-AI, All-Local

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

   - Navigate to your project directory and install required packages using `pip`:
   
     ```bash
     cd path/to/your/project
     pip install -r requirements.txt
     ```

3. **Setup Groq Key:**

   - Obtain your Groq API key from [Groq Console](https://console.groq.com/keys).
   - Set your key in the `.env` file as follows:
   
     ```plaintext
     GROQ_API_KEY=<YOUR_KEY>
     ```

4. **Install and Run Qdrant DB**

   Qdrant DB is a vector search engine used for efficient similarity search. We'll run it in a Docker container.

   - **Install Docker:**

     Docker is required to run Qdrant DB in a containerized environment. Install Docker Desktop from [Docker's official website](https://www.docker.com/products/docker-desktop).

   - **Run Qdrant DB Container:**

     Pull the Qdrant Docker image from Docker Hub and start the container:

     ```bash
     docker pull qdrant/qdrant
     docker run -p 6333:6333 qdrant/qdrant
     ```

     This will pull the latest Qdrant image and start it on port 6333.

   - **Verify Qdrant Installation:**

     After starting the container, you can verify Qdrant is running by checking its API endpoints:

     ```plaintext
     curl http://localhost:6333/collections
     ```

     This command should return information about the collections managed by Qdrant.

   - **Access Qdrant Dashboard:**

     Qdrant provides a web-based dashboard for monitoring and managing collections. Open the following URL in your web browser:

     ```
     http://localhost:6333/dashboard
     ```

5. **Install NVIDIA CUDA (Optional for GPU acceleration):**

   If you plan to use GPU acceleration, install NVIDIA CUDA Toolkit and cuDNN:

   - Download CUDA Toolkit from [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
   - Download cuDNN from [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).

6. **Run the Drive-Through Application**

   Finally, execute the following command to start the Drive-Through application:

   ```bash
   streamlit run DriveThrough.py
