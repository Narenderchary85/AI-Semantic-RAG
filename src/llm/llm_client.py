import subprocess
import shlex
import json

class OllamaClient:
   def __init__(self, model="mistral:latest"):
        self.model = model

   def generate(self, prompt, max_tokens=300):
        cmd = ["ollama", "run", self.model]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        stdout, stderr = process.communicate(prompt)

        if process.returncode != 0:
            raise RuntimeError(f"Ollama error: {stderr}")

        return stdout.strip()