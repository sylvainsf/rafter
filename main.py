# main.py (Full Python Workflow)
import argparse
import numpy as np
import openai
import requests
import yaml
import json
import os
import time
import subprocess
import tiktoken  # For token counting

# --- Configuration ---
# Default values
DEFAULT_OPENAPI_SPEC_PATH = "api.swagger.yaml"
DEFAULT_GOLANG_TEST_DIRECTORY = "pkg/"
DEFAULT_GO_AGENT_EXECUTABLE = "./main"
DEFAULT_AZURE_OPENAI_VERSION = "2023-05-15"
DEFAULT_LARGER_MODEL_ENGINE = "gpt-4o"
DEFAULT_FAST_MODEL_ENGINE = "gpt-35-turbo"
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_INGESTION_MAX_TOKENS = 256
DEFAULT_VALIDATION_MAX_TOKENS = 128
DEFAULT_VARIANT_GENERATION_MAX_TOKENS = 512
DEFAULT_VARIANT_GENERATION_BATCH_SIZE = 10
DEFAULT_VARIANT_GENERATION_EPOCHS = 3
DEFAULT_VARIANT_GENERATION_CHECKPOINT_FILE = "generation_checkpoint.json"
DEFAULT_DESTRUCTIVE_GENERATION_MAX_TOKENS = 256
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="RAFT implementation for Radius API fine-tuning.")
parser.add_argument("--openapi_spec_path", default=DEFAULT_OPENAPI_SPEC_PATH, help="Path to the OpenAPI specification file.")
parser.add_argument("--golang_test_directory", default=DEFAULT_GOLANG_TEST_DIRECTORY, help="Directory containing Golang test files.")
parser.add_argument("--go_agent_executable", default=DEFAULT_GO_AGENT_EXECUTABLE, help="Path to the compiled Go agent executable.")
parser.add_argument("--azure_openai_key", default=os.environ.get("AZURE_OPENAI_KEY"), help="Azure OpenAI API key.")
parser.add_argument("--azure_openai_endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"), help="Azure OpenAI endpoint URL.")
parser.add_argument("--azure_openai_version", default=os.environ.get("AZURE_API_VERSION") if os.environ.get("AZURE_API_VERSION") else DEFAULT_AZURE_OPENAI_VERSION, help="Azure OpenAI API version.")
parser.add_argument("--larger_model_engine", default=DEFAULT_LARGER_MODEL_ENGINE, help="Name of the larger Azure OpenAI model.")
parser.add_argument("--fast_model_engine", default=DEFAULT_FAST_MODEL_ENGINE, help="Name of the faster Azure OpenAI model.")
parser.add_argument("--similarity_threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Similarity threshold for validation.")
parser.add_argument("--ingestion_max_tokens", type=int, default=DEFAULT_INGESTION_MAX_TOKENS, help="Max tokens for ingestion.")
parser.add_argument("--validation_max_tokens", type=int, default=DEFAULT_VALIDATION_MAX_TOKENS, help="Max tokens for validation.")
parser.add_argument("--variant_generation_max_tokens", type=int, default=DEFAULT_VARIANT_GENERATION_MAX_TOKENS, help="Max tokens for variant generation.")
parser.add_argument("--variant_generation_batch_size", type=int, default=DEFAULT_VARIANT_GENERATION_BATCH_SIZE, help="Batch size for variant generation.")
parser.add_argument("--variant_generation_epochs", type=int, default=DEFAULT_VARIANT_GENERATION_EPOCHS, help="Number of epochs for variant generation.")
parser.add_argument("--variant_generation_checkpoint_file", default=DEFAULT_VARIANT_GENERATION_CHECKPOINT_FILE, help="Checkpoint file for variant generation.")
parser.add_argument("--destructive_generation_max_tokens", type=int, default=DEFAULT_DESTRUCTIVE_GENERATION_MAX_TOKENS, help="Max tokens for destructive generation.")
parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL, help="The embedding model to use")


args = parser.parse_args()

print(args)

# Assign command-line arguments to variables
OPENAPI_SPEC_PATH = args.openapi_spec_path
GOLANG_TEST_DIRECTORY = args.golang_test_directory
GO_AGENT_EXECUTABLE = args.go_agent_executable
AZURE_OPENAI_API_KEY = args.azure_openai_key
AZURE_OPENAI_ENDPOINT = args.azure_openai_endpoint
AZURE_OPENAI_VERSION = args.azure_openai_version
LARGER_MODEL_ENGINE = args.larger_model_engine
FAST_MODEL_ENGINE = args.fast_model_engine
SIMILARITY_THRESHOLD = args.similarity_threshold
INGESTION_MAX_TOKENS = args.ingestion_max_tokens
VALIDATION_MAX_TOKENS = args.validation_max_tokens
VARIANT_GENERATION_MAX_TOKENS = args.variant_generation_max_tokens
VARIANT_GENERATION_BATCH_SIZE = args.variant_generation_batch_size
VARIANT_GENERATION_EPOCHS = args.variant_generation_epochs
VARIANT_GENERATION_CHECKPOINT_FILE = args.variant_generation_checkpoint_file
DESTRUCTIVE_GENERATION_MAX_TOKENS = args.destructive_generation_max_tokens
EMBEDDING_MODEL = args.embedding_model


# --- Helper Functions ---

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity."""
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0
    return dot_product / (magnitude_vec1 * magnitude_vec2)

def get_openai_embedding(text, model=EMBEDDING_MODEL):
    # """Gets OpenAI embedding."""
    # openai.api_type = "azure"
    # openai.api_key = AZURE_OPENAI_API_KEY
    # openai.api_base = AZURE_OPENAI_ENDPOINT
    # openai.api_version = AZURE_OPENAI_VERSION
    # response = openai.Embedding.create(input=text, engine=model)
    # return response['data'][0]['embedding']
    try:
        client = openai.AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
    
        response = client.embeddings.create(input=[text], model=model)
        
        return response.data[0].embedding
    except Exception as e:
        print(f"An error occurred while trying to call the api: {e}")
        print(f"api key: {AZURE_OPENAI_API_KEY}")
        print(f"endpoint: {AZURE_OPENAI_ENDPOINT}")
        raise e

def is_destructive(text, additional_keywords=None):
    """Checks for destructive actions."""
    destructive_keywords = ["delete", "remove", "drop", "truncate", "disable", "shutdown", "destroy"]
    if additional_keywords:
        destructive_keywords.extend(additional_keywords)
    text = text.lower()
    for keyword in destructive_keywords:
        if keyword in text:
            return True
    return False

def count_tokens(text, model_name="gpt-3.5-turbo"):
    """Counts the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base") #Default
    num_tokens = len(encoding.encode(text))
    return num_tokens

# --- Agents (Python) ---

class OpenAPI_IngestionAgent:
    def __init__(self, openapi_spec_path, max_tokens=INGESTION_MAX_TOKENS):
        self.openapi_spec_path = openapi_spec_path
        self.max_tokens = max_tokens
        with open(self.openapi_spec_path, 'r') as f:
            try:
                self.spec = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error loading OpenAPI spec: {exc}")
                self.spec = None

    def ingest(self, batch_size=10):
        if self.spec is None:
            return []

        extracted_data = []
        destructive_data = []
        operations = []  # Collect operations first
        for path, methods in self.spec.get('paths', {}).items():
            for method, details in methods.items():
                operations.append((path, method, details))

        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            for path, method, details in batch:
                operation_id = details.get('operationId', '')
                summary = details.get('summary', '')
                description = details.get('description', '')
                params = []
                for param in details.get('parameters', []):
                    params.append(f"{param.get('name', '')} ({param.get('in', '')}, {param.get('schema', {}).get('type', '')})")

                prompt = f"""
                You are an expert in extracting information from OpenAPI specifications.
                Given the following OpenAPI details for the '{method.upper()}' operation on '{path}':

                Operation ID: {details.get('operationId', 'N/A')}
                Summary: {details.get('summary', 'N/A')}
                Description: {details.get('description', 'N/A')}
                Parameters: {details.get('parameters', [])}
                Responses: {details.get('responses', {})}

                Generate a question-answer pair about this API operation. The question should be
                a realistic user query. The answer should be concise and accurate, based *solely*
                on the provided information.

                Consider:
                * How-to questions
                * What-is questions
                * Parameter usage questions
                * Error handling questions
                * Example request questions

                Output in JSON format:
                {{
                    "question": "<question>",
                    "answer": "<answer>"
                }}
                """
                try:
                    client = openai.AzureOpenAI(
                        api_key=AZURE_OPENAI_API_KEY,
                        api_version=AZURE_OPENAI_VERSION,
                        azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    )
                    messages=[
                           {"role": "system", "content": "You are a helpful assistant specializing in OpenAPI specifications."},
  
                    # response = openai.chat.completions.create(
                    #     engine="gpt-35-turbo", # Or your deployment name
                    #     messages=[
                    #         {"role": "system", "content": "You are a helpful assistant specializing in OpenAPI specifications."},
                            {"role": "user", "content": prompt}
                        ],
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                    qa_pair = json.loads(client['choices'][0]['message']['content'])
                    question = qa_pair["question"]
                    answer = qa_pair["answer"]

                    data_entry = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that provides information about the Radius API."},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ],
                        "source": "OpenAPI",
                        "operation_id": operation_id,
                        "path": path,
                        "method": method
                    }

                    if is_destructive(question) or is_destructive(answer):
                        data_entry["flagged"] = "true"
                        data_entry["reason"] = "Potential destructive action keywords detected."
                        destructive_data.append(data_entry)  # Add to destructive_data
                    else:
                      extracted_data.append(data_entry)
                except (json.JSONDecodeError, KeyError, openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
                   print(f"Error processing OpenAPI response: {e}")
                   if isinstance(e, openai.error.RateLimitError):
                       time.sleep(60)  # Simple retry
                #                             # Dump the entire response object to stdout for debugging
                #     print("--- Raw Response Object (for debugging) ---")
                #     print(json.dumps(client, indent=2))
                #     print("--- End of Raw Response ---")
                # except NameError:
                #          print("--- Raw Response Object (for debugging) ---")
                #          print("Response object was not created")
                #          print("--- End of Raw Response ---")
                # except Exception as e:
                #         print(f"An error occurred while trying to dump the response: {e}")

            return extracted_data, destructive_data

class OpenAPI_ValidationAgent:
    def __init__(self, fast_model_engine=FAST_MODEL_ENGINE, max_tokens=VALIDATION_MAX_TOKENS):
        self.fast_model_engine = fast_model_engine
        self.max_tokens = max_tokens

    def validate(self, question, context, generated_answer):
        """Validates a single generated answer."""
        context_embedding = get_openai_embedding(context)
        generated_answer_embedding = get_openai_embedding(generated_answer)
        fast_model_answer = self.get_fast_model_response(question, context)
        fast_model_answer_embedding = get_openai_embedding(fast_model_answer)

        similarity_to_context = cosine_similarity(generated_answer_embedding, context_embedding)
        similarity_to_fast_model = cosine_similarity(generated_answer_embedding, fast_model_answer_embedding)

        return similarity_to_context, similarity_to_fast_model

    def get_fast_model_response(self, question, context):
        """Gets a response from the fast model."""
        prompt = f"""
        You are a validation agent for an LLM. Your task is to provide a concise answer
        to the question, *based solely on the provided context (from an OpenAPI spec)*.
        Do not hallucinate.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        try:
            openai.api_type = "azure"
            openai.api_key = AZURE_OPENAI_API_KEY
            openai.api_base = AZURE_OPENAI_ENDPOINT
            openai.api_version = AZURE_OPENAI_VERSION
            # response = openai.chat.completions.create(
            #     engine=self.fast_model_engine,
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=self.max_tokens,
            client = openai.AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=self.fast_model_engine,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            return client['choices'][0]['message']['content']
        except (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
            print(f"Error in fast model response: {e}")
            if isinstance(e, openai.error.RateLimitError):
                time.sleep(60)
            return ""


class Golang_ValidationAgent:
    def __init__(self, fast_model_engine=FAST_MODEL_ENGINE, max_tokens=VALIDATION_MAX_TOKENS):
        self.fast_model_engine = fast_model_engine
        self.max_tokens = max_tokens

    def validate(self, question, context, generated_answer):
        """Validates a single generated answer (Go context)."""
        context_embedding = get_openai_embedding(context)
        generated_answer_embedding = get_openai_embedding(generated_answer)
        fast_model_answer = self.get_fast_model_response(question, context)
        fast_model_answer_embedding = get_openai_embedding(fast_model_answer)

        similarity_to_context = cosine_similarity(generated_answer_embedding, context_embedding)
        similarity_to_fast_model = cosine_similarity(generated_answer_embedding, fast_model_answer_embedding)
        return similarity_to_context, similarity_to_fast_model

    def get_fast_model_response(self, question, context):
        """Gets response from fast model (Go context)."""
        prompt = f"""
        You are a validation agent for an LLM.  Provide a concise answer
        to the question, *based solely on the context (from a Golang test)*.
        Do not hallucinate

        Context (from Go test):
        {context}

        Question:
        {question}

        Answer:
        """
        try:
            openai.api_type = "azure"
            openai.api_key = AZURE_OPENAI_API_KEY
            openai.api_base = AZURE_OPENAI_ENDPOINT
            openai.api_version = AZURE_OPENAI_VERSION
            response = openai.chat.completions.create(
                engine=self.fast_model_engine,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
            )
            return response['choices'][0]['message']['content']
        except (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
            print(f"Error in fast model response (Golang): {e}")
            if isinstance(e, openai.error.RateLimitError):
                time.sleep(60)
            return ""


# --- Data Generation Functions ---

def generate_validation_data(openapi_agent, go_agent_executable, go_test_directory):
    """Generates the validation dataset (OpenAPI and Go)."""
    validation_data = []
    destructive_data = []

    print("Ingesting data from OpenAPI...")
    print(openapi_agent.ingest())
    openapi_data, openapi_destructive = openapi_agent.ingest()
    for item in openapi_data:
        item["source_type"] = "openapi"
    validation_data.extend(openapi_data)

    print("Ingesting data from Golang tests...")
    try:
        result = subprocess.run(
            [go_agent_executable, go_test_directory],
            capture_output=True,
            text=True,
            check=True
        )
        go_data = json.loads(result.stdout)
        for item in go_data:
            formatted_item = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": item["Question"]},
                    {"role": "assistant", "content": item["Answer"]}
                ],
                "source": f"Golang Test: {item['SourceFile']}",
                "source_type": "golang",
                "test_name": item["TestName"],
                "file_path": item["SourceFile"]
            }
            validation_data.append(formatted_item)

    except subprocess.CalledProcessError as e:
        print(f"Error running Go agent: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding Go agent output: {e}\nOutput: {result.stdout}")
    return validation_data, destructive_data



def generate_training_data(validation_data, openapi_validation_agent, golang_validation_agent,
                           larger_model_engine, batch_size, epochs, checkpoint_file):
    """Generates training data with synthetic variants and validation.

    Args:
        validation_data: Initial dataset.
        openapi_validation_agent: Agent for OpenAPI validation.
        golang_validation_agent: Agent for Golang validation.
        larger_model_engine: Larger Azure OpenAI model.
        batch_size: Batch size for *variant generation*.
        epochs: Number of passes over the validation data.
        checkpoint_file: File to save/resume progress.
    """
    training_data = []
    start_epoch = 0
    start_index = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_epoch = checkpoint['epoch']
            start_index = checkpoint.get('index', 0)  # Handle potential missing index
            training_data = checkpoint['data']
            print(f"Resuming from epoch {start_epoch}, index {start_index}")

    for epoch in range(start_epoch, epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        # Reset index for each epoch to start from the beginning of validation_data
        current_index = 0 if epoch != start_epoch else start_index

        while current_index < len(validation_data):
            batch = validation_data[current_index:current_index + batch_size]
            for item in batch:
              # --- Variant Generation ---
              question = item["messages"][1]["content"]
              answer = item["messages"][2]["content"]
              context = item["source"]  # Use the source as context
              source_type = item.get("source_type", "openapi") # Default

              variant_generation_prompt = f"""
                You are a data augmentation agent for an LLM. Generate variations of
                the provided question-answer pair. The variations should:

                * Maintain the core meaning.
                * Use different phrasing.
                * Explore parameter names (if applicable).
                * Include *slightly* incorrect/ambiguous questions (but not nonsensical).
                * *Strictly avoid* generating destructive actions.

                Original Question: {question}
                Original Answer: {answer}

                Generate a JSON object containing a 'question' and 'answer' field.
                """
            try:
                    response = openai.chat.completions.create(
                        engine=larger_model_engine,
                        messages=[
                            {"role": "system", "content": "You are a data augmentation agent."},
                            {"role": "user", "content": variant_generation_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=VARIANT_GENERATION_MAX_TOKENS,
                    )
                    variant = json.loads(response['choices'][0]['message']['content'])
                    generated_question = variant["question"]
                    generated_answer = variant["answer"]

                    # Choose the correct validation agent
                    if source_type == "openapi":
                        validation_agent = openapi_validation_agent
                    elif source_type == "golang":
                        validation_agent = golang_validation_agent
                    else:
                        print(f"Warning: Unknown source type {source_type}. Skipping validation.")
                        continue

                    similarity_to_context, similarity_to_fast_model = validation_agent.validate(
                        generated_question, context, generated_answer
                    )

                    if similarity_to_context >= SIMILARITY_THRESHOLD and similarity_to_fast_model >= SIMILARITY_THRESHOLD:
                        training_data.append({
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant that provides information about the Radius API."},
                                {"role": "user", "content": generated_question},
                                {"role": "assistant", "content": generated_answer}
                            ],
                            "source": item["source"],  # Keep original source info
                            "source_type": source_type
                        })

            except (json.JSONDecodeError, KeyError, openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
                    print(f"Error generating/validating variant: {e}")
                    if isinstance(e, openai.error.RateLimitError):
                        time.sleep(60)

            current_index += batch_size
            print(f"Epoch {epoch+1}, Processed {current_index}/{len(validation_data)} validation items")

            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'index': current_index,
                    'data': training_data
                }, f)

    return training_data

def save_data(data, filename):
    """Saves data to a JSONL file."""
    with open(filename, "w") as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')


# --- Main Workflow ---

def main():
    """Main function to orchestrate the RAFT process."""

    # --- 1. Initialize Agents ---
    openapi_agent = OpenAPI_IngestionAgent(OPENAPI_SPEC_PATH)
    openapi_validation_agent = OpenAPI_ValidationAgent()
    golang_validation_agent = Golang_ValidationAgent()

    # --- 2. Data Generation ---
    validation_data, initial_destructive_data = generate_validation_data(
        openapi_agent, GO_AGENT_EXECUTABLE, GOLANG_TEST_DIRECTORY
    )
    save_data(validation_data, "validation_data.jsonl")
    print(f"Validation data generated: {len(validation_data)} items")

    training_data = generate_training_data(
        validation_data, openapi_validation_agent, golang_validation_agent,
        LARGER_MODEL_ENGINE, VARIANT_GENERATION_BATCH_SIZE, VARIANT_GENERATION_EPOCHS,
        VARIANT_GENERATION_CHECKPOINT_FILE
    )
    print(f"Generated {len(training_data)} training examples.")

    # 3. Generate Training Data (Synthetic Variants)
    openapi_validation_agent = OpenAPI_ValidationAgent()
    golang_validation_agent = Golang_ValidationAgent()
    training_data = generate_training_data(
        validation_data,
        openapi_validation_agent,
        golang_validation_agent,
        LARGER_MODEL_ENGINE,
        VARIANT_GENERATION_BATCH_SIZE,
        VARIANT_GENERATION_EPOCHS,
        VARIANT_GENERATION_CHECKPOINT_FILE
    )
    save_data(training_data, "training_data_raw.jsonl") # Save *before* merging
    print(f"Training data (raw) generated: {len(training_data)} items")

    # --- 4. Fine-Tuning (using Azure OpenAI API) ---
    #   This part would involve using the Azure OpenAI API to upload the data
    #   and create/manage a fine-tuning job.  This is *not* fully implemented
    #   in this example, as it requires interaction with the Azure API. 
    #   In this implementation we want human review of destructive data before 
    #   upload so we are not automating.
    print("----- Fine-Tuning (Conceptual) -----")
    print("Upload training and validation data to Azure OpenAI.")
    print("Create and manage a fine-tuning job using the Azure OpenAI API.")
    print("-------------------------------------")


if __name__ == "__main__":
    main()