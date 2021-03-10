# -----------------------------------------
# Workshop - List of TODOs for bedrock.hcl
# -----------------------------------------
# 1. Name the stanzas x2
# 2. Name the step in the stanza
# 3. Increase resources for training


# IMPORTANT: Bedrock HCL version
version = "1.0"

# TODO: name the stanza
train {
    # TODO: name the step
    step train {
        image = "continuumio/miniconda3:latest"
        install = [
            "conda env update -f environment-train.yaml",
            "eval \"$(conda shell.bash hook)\"",
            "conda activate veritas"
        ]
        script = [{sh = ["python train.py"]}]
        # TODO: Increase the resources... reasonably!
        resources {
            cpu = "1"
            memory = "4G"
        }
        parameters {
            TRAINING_DATA_AWS_BUCKET = "s3://veritas-credit-scoring/data/training/latest"
        }
    }
}

# TODO: name the stanza
serve {
    image = "python:3.7"
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    script = [
        {sh = [
            "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve:app"
        ]}
    ]
    parameters {
        WORKERS = "1"
    }
}