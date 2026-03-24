FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app

COPY environment.yml ./environment.yml
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "mlops-student-env", "/bin/bash", "-c"]

COPY . .

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "mlops-student-env", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
