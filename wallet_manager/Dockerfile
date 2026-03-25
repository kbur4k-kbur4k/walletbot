FROM python:3.11-slim

WORKDIR /app

# Create a non-root user with a home directory
RUN addgroup --system appgroup \
 && adduser --system --ingroup appgroup --home /home/appuser appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . \
 && chown -R appuser:appgroup /app

USER appuser

ENTRYPOINT ["python", "-m", "wallet_manager.orchestrator"]
