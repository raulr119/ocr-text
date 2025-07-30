Here’s your **production-ready README.md** for the full workflow:

---

# 🚀 ID Card OCR API – Docker Deployment Guide

This guide covers everything you need to:

✅ Build a Docker image for your project
✅ Push it to Docker Hub
✅ Deploy on any server (CPU or GPU)
✅ Make the API accessible publicly
✅ Update the app easily after changes

---

## ✅ 1. **Prerequisites**

* [Docker installed](https://docs.docker.com/get-docker/) on your local machine and the server.
* A [Docker Hub account](https://hub.docker.com/).
* Your project has:

  * `Dockerfile`
  * `requirements.txt`
  * `app/main.py` (FastAPI entry point)
  * `.env` file for config (optional)

---

## ✅ 2. **Build Docker Image Locally**

From your project root directory (where `Dockerfile` is):

```bash
# Login to Docker Hub
docker login

# Build the image
docker build -t your-dockerhub-username/ekyc:latest .

# (Optional) Add version tag for stability
docker build -t your-dockerhub-username/ekyc:1.0.0 -t your-dockerhub-username/ekyc:latest .
```

---

## ✅ 3. **Push Image to Docker Hub**

```bash
docker push your-dockerhub-username/ekyc:latest
# (If versioned)
docker push your-dockerhub-username/ekyc:1.0.0
```

---

## ✅ 4. **Deploy on Any Server**

1. **Install Docker on the server: Run this command line by line**

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
exit
```
**Optional command for installation**

```bash
curl -fsSL https://get.docker.com | sh

```

2. **Pull your image:**

```bash
docker pull gladiator11/ekyc:1.0.0
```

3. **Run the container:**

```bash
docker run -d \
  --name ekyc \
  -p 8080:8080 \
  -e PORT=8080 \
  --restart unless-stopped \
  gladiator11/ekyc:1.0.0
```

✔ **Auto-restart** after reboot (`--restart unless-stopped`).
✔ Works on **GPU if available** (CUDA), otherwise CPU fallback.

---

## ✅ 5. **Open Port for Public Access**

* Allow port in the server firewall:

```bash
sudo ufw allow 8080
```

* Access API from your browser:

```
http://<your-server-ip>:8080/docs
```

---

## ✅ 6. **Update After Code Changes**

When you make changes to the project:

```bash
# Locally
docker build -t your-dockerhub-username/ekyc:latest .
docker push your-dockerhub-username/ekyc:latest

# On server
docker pull your-dockerhub-username/ekyc:latest
docker stop id-ocr-service && docker rm id-ocr-service
docker run -d -p 8080:8080 -e PORT=8080 --restart unless-stopped your-dockerhub-username/ekyc:latest
```

---

## ✅ 7. **Verify Deployment**

* Check running containers:

```bash
docker ps
```

* Test health:

```bash
curl http://<your-server-ip>:8080/health
```

---

### ✅ Key Features of This Setup

✔ Single universal image → works on **CPU or GPU** automatically.
✔ Minimal commands on the server → **no need for docker-compose**.
✔ Auto-restart after server reboot.
✔ Easy updates using `docker pull` + `docker run`.


