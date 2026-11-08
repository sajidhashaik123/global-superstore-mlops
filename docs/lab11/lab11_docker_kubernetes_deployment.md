
# Experiment 11: Docker Containerization and Kubernetes Deployment

## 1. Aim

Containerize and deploy machine learning applications, trained models, APIs, dependencies, and configuration files using Docker in isolated and reproducible environments. Evaluate portability, dependency isolation, and prediction consistency, and explore Kubernetes-based deployment, service exposure, scaling, failure handling, debugging, and recovery workflows.

## 2. Objectives

1. Containerize the machine learning API using Docker.
2. Package the trained model, preprocessing artifacts, application code, and dependencies.
3. Deploy the containerized ML application in an isolated environment.
4. Verify dependency isolation and application portability.
5. Check prediction consistency between Docker and Kubernetes deployments.
6. Deploy the application using Kubernetes Deployment.
7. Expose the application using a Kubernetes Service.
8. Perform horizontal scaling by increasing the number of replicas.
9. Test failure handling and automatic Pod recovery.
10. Inspect Kubernetes events and Pod information for debugging.

## 3. Dataset and Model

**Dataset:** Global Superstore

**Model source:** Experiment 7

**Model type:** Logistic Regression

**Prediction target:** Profitability

**Model artifacts:**
- `models/lab7/v1/classifier.joblib`
- `models/lab7/v1/preprocessor.joblib`
- `models/lab7/v1/training_metadata.json`

**Model version:** `lab7-v1`

The existing trained model and preprocessing artifacts were reused. No new dataset or target variable was introduced.

## 4. Software and Tools Required

- Python 3.13
- Visual Studio Code
- Docker Desktop
- Docker Engine
- Kubernetes
- kubectl
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- NumPy
- Joblib
- PowerShell

## 5. Implementation

### Step 1: Docker Environment Verification

Docker Desktop was installed and configured with the WSL2 backend.

Commands:

```powershell
docker --version
docker info
```

Docker version verified: `29.9.2`.

### Step 2: Dependency Configuration

The existing `requirements.txt` was updated with the API dependencies.

Additional packages:

```text
fastapi==0.162.0
uvicorn==0.54.0
pydantic==2.14.3
```

Other project dependencies include:

```text
pandas
numpy
scikit-learn
mlflow==3.18.0
scipy
joblib
```

### Step 3: Dockerfile Creation

A Dockerfile was created in the project root.

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "lab10.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
```

The Dockerfile:
- Uses Python 3.13 slim as the base image.
- Creates `/app` as the working directory.
- Installs application dependencies.
- Copies source code and trained model artifacts.
- Exposes port 8000.
- Starts the FastAPI application using Uvicorn.

### Step 4: Docker Ignore Configuration

A `.dockerignore` file was created.

```text
.venv/
__pycache__/
*.pyc
.git/
.github/
mlruns/
mlflow.db
.ipynb_checkpoints/
.pytest_cache/
.env
```

This excludes unnecessary local files from the Docker build context.

### Step 5: Build Docker Image

Command:

```powershell
docker build -t global-superstore-ml-api:v1 .
```

Result: Docker image built successfully.

Image name: `global-superstore-ml-api:v1`

### Step 6: Run Docker Container

Command:

```powershell
docker run -d --name global-superstore-api -p 8000:8000 global-superstore-ml-api:v1
```

The container started successfully.

API documentation:

`http://localhost:8000/docs`

Health endpoint:

`http://localhost:8000/health`

### Step 7: Docker API Testing

The following endpoints were tested:

| Endpoint | Method | Result |
|---|---|---|
| `/health` | GET | HTTP 200 |
| `/metadata` | GET | HTTP 200 |
| `/predict` | POST | HTTP 200 |
| `/predict/batch` | POST | HTTP 200 |

An invalid input was also tested and FastAPI returned HTTP 422, confirming input validation.

### Step 8: Kubernetes Environment Setup

Kubernetes was enabled through Docker Desktop.

Commands:

```powershell
kubectl version --client
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods -A
```

The Kubernetes node was:

`desktop-control-plane`

Node status: Ready.

Kubernetes version: `v1.36.1`.

### Step 9: Import Docker Image into Kubernetes

The locally built Docker image was imported into the Kubernetes container runtime.

Command:

```powershell
docker image save global-superstore-ml-api:v1 -o global-superstore-api.tar
```

Image import:

```powershell
cmd.exe /c "docker image save global-superstore-ml-api:v1 | docker exec -i desktop-control-plane ctr -n k8s.io images import -"
```

Verification:

```powershell
docker exec desktop-control-plane ctr -n k8s.io images ls
```

The image was successfully imported into Kubernetes containerd.

### Step 10: Kubernetes Deployment Configuration

A `k8s/deployment.yaml` file was created.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: global-superstore-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: global-superstore-api
  template:
    metadata:
      labels:
        app: global-superstore-api
    spec:
      containers:
        - name: ml-api
          image: global-superstore-ml-api:v1
          imagePullPolicy: Never
          ports:
            - containerPort: 8000
```

Apply the configuration:

```powershell
kubectl apply -f k8s/deployment.yaml
```

Verification:

```powershell
kubectl get deployments
kubectl get pods -l app=global-superstore-api -o wide
```

The Deployment and Pod were created successfully.

### Step 11: Kubernetes Application Logs

Command:

```powershell
kubectl logs deployment/global-superstore-api
```

The logs confirmed:
- Model and preprocessor loaded successfully.
- Application startup completed.
- Uvicorn started on port 8000.

### Step 12: Kubernetes Service Configuration

A `k8s/service.yaml` file was created.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: global-superstore-api-service
spec:
  type: NodePort
  selector:
    app: global-superstore-api
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
      nodePort: 30080
```

Apply:

```powershell
kubectl apply -f k8s/service.yaml
```

Verify:

```powershell
kubectl get services
```

The Service was created successfully with NodePort `30080`.

### Step 13: Access Kubernetes API

The API was accessed using Kubernetes port forwarding.

Command:

```powershell
kubectl port-forward service/global-superstore-api-service 8081:8000
```

The API was successfully accessed at:

`http://localhost:8081/health`

Swagger UI:

`http://localhost:8081/docs`

The health endpoint returned:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Step 14: Prediction Consistency Testing

The same Global Superstore record was submitted to the Docker API and Kubernetes API.

Input record:

- Sales: 2309.65
- Quantity: 7
- Discount: 0.0
- Shipping Cost: 933.57
- Order Year: 2012
- Order Month: 7
- Ship Mode: Same Day
- Segment: Consumer
- City: New York City
- State: New York
- Country: United States
- Market: US
- Region: East
- Category: Technology
- Sub-Category: Accessories
- Order Priority: Critical

Observed results:

| Metric | Docker | Kubernetes |
|---|---|---|
| Prediction | Profitable | Profitable |
| Profitability | 1 | 1 |
| Probability | 0.9923 | 0.9923 |
| Model version | lab7-v1 | lab7-v1 |
| HTTP status | 200 | 200 |

Result: Both environments produced identical predictions for the tested record.

### Step 15: Horizontal Scaling

The Deployment was initially configured with one replica.

Scaling command:

```powershell
kubectl scale deployment global-superstore-api --replicas=3
```

Verification:

```powershell
kubectl get pods -l app=global-superstore-api -o wide
```

Three Pods were successfully created and reached Running status.

### Step 16: Failure Handling and Automatic Recovery

One Pod was deliberately deleted to test Kubernetes self-healing.

Command:

```powershell
kubectl delete pod global-superstore-api-7467f5949f-hf5fj
```

Kubernetes automatically created a replacement Pod.

Verification:

```powershell
kubectl get pods -l app=global-superstore-api -o wide
kubectl get deployments
```

Observed result:

- Three Pods were available again.
- The replacement Pod reached Running status.
- All three replicas became ready.
- The Deployment returned to 3/3 available.

This demonstrated automatic Pod recovery.

### Step 17: Kubernetes Debugging

Kubernetes events were inspected using:

```powershell
kubectl get events --sort-by=.metadata.creationTimestamp
```

Detailed Pod information was inspected using:

```powershell
kubectl describe pod global-superstore-api-7467f5949f-xz4gd
```

The output confirmed:
- Pod status: Running.
- Container state: Running.
- Ready: True.
- Restart count: 0.
- Container image: `global-superstore-ml-api:v1`.
- Container port: 8000.

Events such as Scheduled, Pulled, Created, Started, Killing, and SuccessfulCreate were examined.

### Step 18: Dependency Isolation Verification

Installed packages inside the Docker container were inspected.

Command:

```powershell
docker exec global-superstore-api pip list
```

Important installed dependencies included:

| Package | Version |
|---|---|
| FastAPI | 0.162.0 |
| Uvicorn | 0.54.0 |
| Pydantic | 2.14.3 |
| NumPy | 2.5.4 |
| Pandas | 3.1.1 |
| Scikit-learn | 1.9.1 |
| MLflow | 3.18.0 |
| SciPy | 1.18.1 |
| Joblib | 1.6.0 |

This verified that application dependencies were installed inside the Docker container rather than relying on the Windows virtual environment.

### Step 19: Docker Image Portability Inspection

Command:

```powershell
docker image inspect global-superstore-ml-api:v1 --format "Image: {{.RepoTags}}`nArchitecture: {{.Architecture}}`nOS: {{.Os}}`nSize: {{.Size}} bytes"
```

Observed output:

```text
Image: [global-superstore-ml-api:v1]
Architecture: amd64
OS: linux
Size: 1312555643 bytes
```

The image was successfully executed in both the standalone Docker container and the local Kubernetes environment.

### Step 20: Final Deployment Verification

Command:

```powershell
kubectl get deployments,pods,services
```

Final observed results:

- Deployment READY: 3/3.
- UP-TO-DATE: 3.
- AVAILABLE: 3.
- Three Pods: Running.
- Pod restart counts: 0.
- Service type: NodePort.
- Service port: 8000:30080/TCP.

## 6. Results and Observations

1. The machine learning application was successfully containerized using Docker.
2. The trained model and preprocessing artifacts were included in the image.
3. The FastAPI application ran successfully inside the Docker container.
4. The containerized application was deployed successfully in Kubernetes.
5. Kubernetes Service and port forwarding enabled API access.
6. Docker and Kubernetes produced identical predictions for the tested record.
7. Horizontal scaling from one to three replicas was successful.
8. Kubernetes automatically replaced a deliberately deleted Pod.
9. Kubernetes events and Pod details were used to inspect deployment behavior.
10. Dependency isolation was verified by inspecting packages inside the container.

## 7. Evaluation and Limitations

- Prediction consistency was verified using the same test record in both environments; broader multi-record consistency testing can provide additional evidence.
- The Kubernetes cluster used one node, so multi-node portability and node-failure recovery were not tested.
- The Docker image was verified on Linux AMD64 architecture.
- Several dependencies in `requirements.txt` are not version-pinned, so complete dependency-lock reproducibility remains a possible improvement.
- The NodePort Service was accessed through `kubectl port-forward`; direct external NodePort access was not separately tested.
- Kubernetes recovery was tested by deleting an application Pod, not by simulating a complete node failure.

## 8. Conclusion

The experiment successfully demonstrated containerization and deployment of the Global Superstore machine learning API using Docker and Kubernetes.

The trained Logistic Regression model and preprocessing artifacts were packaged into a Docker image and deployed through a Kubernetes Deployment. A Kubernetes Service was configured to expose the application, and the API was accessed successfully using port forwarding.

Prediction consistency was verified between Docker and Kubernetes for the tested Global Superstore record. Horizontal scaling from one to three replicas was performed successfully. Automatic Pod recovery was also demonstrated by deliberately deleting one Pod and observing Kubernetes create a replacement.

Kubernetes events, container logs, and detailed Pod descriptions were inspected to understand deployment behavior and debugging workflows.

Thus, the experiment demonstrated Docker-based containerization, dependency isolation, deployment portability between the tested environments, prediction consistency, Kubernetes service exposure, scaling, failure handling, debugging, and automatic recovery.
