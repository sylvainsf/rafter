# Retrieval Augmented Fine-Tuning (RAFT) for Radius: An Agentic, Azure-Powered Implementation

This project is an implementation of the Retrieval Augmented Fine-Tuning (RAFT) framework, inspired by Cedric Vidal's blog post, ["RAFT: A new way to teach LLMs to be better at RAG."](https://techcommunity.microsoft.com/blog/aiplatformblog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/4084674) We aim to build a highly accurate, reliable, and configurable LLM specifically for the open-source Radius project (github.com/radius-project/radius). Our approach is agentic, leveraging Azure AI services. It currently handles both the Radius OpenAPI specification and Golang tests, with a strong emphasis on safety through a system for handling potentially destructive actions. The primary workflow is implemented in Python, with a supporting (simplified) data ingestion agent in Go. Configuration parameters (max tokens, batch sizes, epochs, checkpoints) are included to handle complex APIs.

Once the concept is proven out the goal is to modularize the ingesters and validators for various types of sources, even beyond languages and APIs to include things like textbooks with test/answer keys or any other form of structured assertions on a domain that could be used in this workflow.

## The Problem: LLMs, Hallucinations, and Destructive Actions

Large Language Models (LLMs), while powerful, are prone to hallucination – generating plausible but incorrect information. In the context of Radius, which manages infrastructure and application deployments, a hallucinated response suggesting a *destructive* action (like deleting a resource group or misconfiguring a deployment) is a serious and unacceptable risk. This implementation directly addresses both the accuracy of the LLM and the safety of its generated responses.

## Our Goal: A Safe and Accurate Expert LLM, Inspired by RAFT

Our goal is to build an LLM that acts as an expert on the Radius API. Its knowledge is firmly grounded in two authoritative sources: the project's OpenAPI specification and its Golang tests. We aim to achieve the accuracy benefits outlined in the RAFT approach while also incorporating a robust safety mechanism to prevent destructive actions.

Eventually the Expert LLM of the API could be part of a higher abstraction workflow as a validator for RAFT on Radius Bicep.

## Introducing Our Agentic RAFT Implementation

This document outlines an agentic implementation of the RAFT framework. We build upon the core principles of RAFT (using retrieval augmentation and fine-tuning) and extend it with:

*   **Agentic Architecture:** Specialized, independently tuned agents handle different data sources and tasks.
*   **Multi-Stage Validation:** A "fast-model" validation step checks generated answers for correctness.
*   **Destructive Action Handling:** A system for identifying and mitigating potentially destructive actions.
*   **Configurability:** Parameters to control the data generation and make it useful for complex APIs.

## 1. The Foundation: Authoritative Sources of Truth for Radius

We rely on:

*   **Radius OpenAPI Specification:** The OpenAPI (Swagger) definition, which provides a machine-readable description of the Radius API's endpoints, request/response schemas, and parameters.
*   **Radius Golang Tests:** The Go tests within the Radius project's codebase. These tests provide concrete examples of how the API is used in practice and what constitutes correct behavior.

## 2. Ingestion and Validation Agents: The Agentic Core

We use specialized, independently tuned *agents* for each source type and for validation:

*   **OpenAPI Ingestion Agent (Python):** This agent parses the Radius OpenAPI specification file. It extracts relevant information (endpoints, schemas, parameters, descriptions) and transforms this information into question-answer pairs suitable for fine-tuning the LLM. It uses a carefully crafted prompt to guide an LLM in this transformation process.

*   **Golang Test Ingestion Agent (Go):** This is a Go program that reads Go test files and extracts basic information (test function names and a placeholder answer). A *full* implementation would use the Go AST (Abstract Syntax Tree) and an LLM for more sophisticated analysis, but this simplified version demonstrates the principle and the interaction with the Python workflow

    **To Compile the Go Agent:**

    1.  Save the Go code as `main.go`.
    2.  Open a terminal in the directory where you saved `main.go`.
    3.  Run the command: `go build main.go`
    4.  This will create an executable file named `main` (or `main.exe` on Windows). This is the `GO_AGENT_EXECUTABLE` that the Python code will use.

*   **OpenAPI Validation Agent (Python):** This agent implements the *fast-model validation* step, a key component inspired by the RAFT paper. It uses a smaller, faster LLM (e.g., `gpt-35-turbo`) to check the correctness of generated answers against the original context.
*   **Golang Validation Agent (Python):** This agent is similar to the OpenAPI validation agent, but its prompt is tailored to understand the context derived from Go tests.
*   **Agent Tuning:** Each agent is independently tuned for its specific task (e.g., adjusting `max_tokens`).
*   **User-Specified Sources:** The user provides the paths to the Radius OpenAPI specification file and the directory containing the Go tests.

## 3. Retrieval System (Azure AI Search):

*   **Vector Database (Azure AI Search):** We use Azure AI Search's vector search capabilities.
*   **Indexing:** Extracted knowledge snippets (from both OpenAPI and Go tests) are converted to embeddings (using Azure OpenAI's embedding models) and used to fine tune an Expert LLM in Azure AI.
*   **Retrieval:** The Expert LLM can then be used for accurate user queries about the API. 

## 4. AI-Powered Dataset Generation: Training, Validation, and Destructive Action Handling (Continued)

Inspired by RAFT, we generate multiple datasets to prepare for fine-tuning:

*   **Prompt Engineering:** All LLM interactions use carefully designed prompts, specific to each agent and task.  These prompts are crucial for controlling the LLM's behavior and ensuring the quality of the generated data.

*   **Dataset 1: Validation Dataset (Directly from Sources):**
    *   The Ingestion Agents (OpenAPI and Golang) use their respective prompts to transform the extracted information (from the OpenAPI spec and Go tests) into the `{"messages": [...]}` format required by the Azure OpenAI fine-tuning API. This dataset serves as a baseline for validation and should be highly accurate as the LLM is only given the structure of the ingested data and asked to turn it into a human readable tuple for fine tuning.

*   **Dataset 2: Training Dataset (Synthetically Generated Variants with Agentic Validation):**
    *   A larger Azure OpenAI model (e.g., `gpt-4`) is used to generate variations of the question-answer pairs in the validation dataset. This expands the training data and improves the LLM's robustness.
    *   **Variant Generation Prompt:**

        ```python
        variant_generation_prompt = f"""
        You are a data augmentation agent for an LLM. Generate variations of
        the provided question-answer pair. The variations should:

        * Maintain the core meaning of the original question and answer.
        * Use different phrasing and sentence structures.
        * Explore different parameter names (if applicable, based on the context).
        * Include *slightly* incorrect or ambiguous questions (but not completely nonsensical ones) to improve robustness.
        * *Strictly avoid* generating any questions or answers that suggest destructive actions.

        Original Question: {{question}}
        Original Answer: {{answer}}

        Generate a JSON object containing a 'question' and 'answer' field.
        """
        ```

    *   **Configurable Parameters:**
        *   `max_tokens`: Controls the maximum length of generated variants.
        *   `batch_size`: Processes data in batches to tune around token limits.
        *   `epochs`: The number of passes over the validation data.
        *   `checkpoint_file`: Saves intermediate progress and allows resuming.

    *   **Agentic Fast Model Validation (TODO move this to validation section):** This is a crucial step to filter out low-quality or hallucinated variants:
        1.  **Generate Variant:** The larger model generates a variation based on the `variant_generation_prompt`.
        2.  **Get Embeddings:** Obtain embeddings for the generated question, generated answer, and the original context (from the validation dataset).
        3.  **Fast Model Answer:** The appropriate validation agent (either `OpenAPI_ValidationAgent` or `Golang_ValidationAgent`, depending on the source of the original data) is used with its `get_fast_model_response` method, along with the *original context*.
        4.  **Cosine Similarity Calculation:** Calculate the cosine similarity between:
            *   The generated answer's embedding and the original context embedding.
            *   The generated answer's embedding and the fast model's answer embedding.
        5.  **Thresholding:** If *either* similarity score falls below a predefined threshold (`SIMILARITY_THRESHOLD`), the variant is discarded.  This ensures that the generated variants remain grounded in the original information and are consistent with the fast model's understanding.

*   **Dataset 3: Destructive Action Dataset (Human Review):** This dataset focuses on *negative* examples – questions and answers that involve potentially dangerous actions.  This is crucial for training the LLM to *avoid* generating such responses to questions unrelated such as: "How do I replace X with Y" doesn't inherently mean to delete X.

    *   **Destructive Action Identification:** We use a multi-pronged approach:
        *   **Keyword Matching:** A basic (but essential) check for keywords associated with destructive actions.

            ```python
            def is_destructive(text, additional_keywords=None):
                destructive_keywords = ["delete", "remove", "drop", "truncate", "disable", "shutdown", "destroy"]
                if additional_keywords:
                    destructive_keywords.extend(additional_keywords)
                text = text.lower()
                for keyword in destructive_keywords:
                    if keyword in text:
                        return True
                return False
            ```

        *   **Prompt Injection (During Synthetic Generation):** We *intentionally* prompt the larger model to generate destructive examples.  This is done with a *separate* prompt, and the results are handled with extreme care. *Not currently implemented, just an idea around tuning a data set for security analysis*

            ```python
            destructive_prompt = """
            You are tasked with generating examples of potentially DANGEROUS or DESTRUCTIVE
            commands/API calls related to the Radius API.  These examples will be used to train a safety
            system to *prevent* such actions. Be creative and comprehensive, focusing on actions that could:

            * Delete data or resources.
            * Modify configurations in a way that causes outages or instability.
            * Disable security features.
            * Expose sensitive information.

            For each example, provide a short, clear question and a corresponding
            command/API call (even though it is dangerous).

            Output in JSON format:
            {
                "question": "<question>",
                "answer": "<potentially destructive command/API call>"
            }
            """
            ```

        *   **Dedicated Destructive Action Classifier (Optional):** A more sophisticated approach would involve training a separate classifier (potentially a fine-tuned LLM or a traditional machine learning model) specifically to identify destructive actions.  This could be used in addition to keyword matching and prompt injection. *Not currently implemented*
        *   **Human Review Queue:** All intentionally destructive examples (identified by the standard api usage) are added to a file for *human review*.  This is a critical safety step. These are stored in a separate file for use with review tools mentioned below. 
        *   **Feedback Loop (Simulated):** The results of human review are used to improve the system:
            *   **Improve `is_destructive`:** Add new keywords or refine the logic based on human feedback.
            *   **Train/Fine-tune Classifier:** If a dedicated classifier is used, the human-labeled data is used for training or fine-tuning.
            *   **Adjust Prompts:** Modify the prompts used for synthetic generation to reduce the likelihood of generating destructive examples in the future.

## 5. Fine-Tuning Process using Azure OpenAI Fine-tuning API:

*TODO: the validation dataset needs to be uploaded and used to fine tune the fast validation LLM for use in the workflow for synthetic dataset generation. 
*   **Review Destructive Datasets:** Call the `destructive_review_simulator.py` script on the generated destructive dataset, this will prompt an user to review each case and generate an approved set for merging. 
*   **Merge Datasets:** Call `merge_datasets.py training_file reviewed_destructive_file output_file merge=True` to merge the reviewed destructive dataset with the training dataset.
*   **Data Upload:** Upload the merged training data and validatation data.
*   **Fine-tuning Job Creation:** Use the Azure OpenAI API to create a fine-tuning job, specifying the training and validation data, the base model (e.g., `gpt-35-turbo`), and other hyperparameters.
*   **Model Training:** Azure OpenAI trains the model.
*   **Model Deployment:** Once training is complete, Azure OpenAI deploys the fine-tuned model to a dedicated endpoint.

## 6. Separate Validation LLM:

*TODO: highlight the separate validation LLM and move before the fine tuning step for the expert LLM

## 7. (Optional) Evaluation After Fine-Tuning:

A held-out test set (not used during training or validation) can be used to evaluate the performance of the fine-tuned model.

## The RAFT Workflow (Python and Go):

Before you can run `main.py`, you need to set up your environment. This involves installing the necessary Python libraries and ensuring you have a compiled Go executable for the Go agent. First, install the required Python packages using pip. Open your terminal (or command prompt on Windows) and run:

```bash
pip install numpy openai requests pyyaml tiktoken
```

Next, you need the compiled Go agent. Navigate to the directory where you've saved the Go code (`main.go`) and compile it using the Go build command:

```bash
go build main.go
```

This will create an executable file named `main` (or `main.exe` on Windows) in that directory. Make sure this executable is in the *same directory* as your `main.py` file.

Now, let's talk about configuration and file paths. The `main.py` script relies on several configuration parameters, including file paths, which you can adjust directly within the script:

*   **`OPENAPI_SPEC_PATH = "api.swagger.yaml"`:**  This variable specifies the path to your OpenAPI specification file.  By default, it's set to `"api.swagger.yaml"`, meaning the script expects a file named `api.swagger.yaml` in the same directory.  You *must* create this file and populate it with a valid OpenAPI specification for the Radius API. If your OpenAPI spec is located elsewhere, change this variable accordingly (e.g., `OPENAPI_SPEC_PATH = "/path/to/your/api_spec.yaml"`).
*   **`GOLANG_TEST_DIRECTORY = "pkg/"`:** This variable points to the directory containing your Go tests.  The default is `"pkg/"`, assuming a standard Go project structure where tests are in a subdirectory named `pkg`.  If your Go tests are located in a different directory (for example, `tests/golang/`), you *must* update this variable: `GOLANG_TEST_DIRECTORY = "tests/golang/"`. Make sure the path is relative to where `main.py` will be run.
*   **`GO_AGENT_EXECUTABLE = "./main"`:** This specifies the path to the compiled Go agent executable. The default, `./main`, assumes the executable is in the same directory as `main.py`.  If you compiled it in a different location, adjust this path.

Crucially, you need to set two environment variables before running the script. These provide your Azure OpenAI credentials:

*   **`AZURE_OPENAI_KEY`:** Your Azure OpenAI API key.
*   **`AZURE_OPENAI_ENDPOINT`:** Your Azure OpenAI endpoint URL.

How you set these depends on your operating system.

**Linux/macOS:**

```bash
export AZURE_OPENAI_KEY="your_api_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint_url"
```

**Windows (Command Prompt):**

```batch
set AZURE_OPENAI_KEY=your_api_key
set AZURE_OPENAI_ENDPOINT=your_endpoint_url
```

**Windows (PowerShell):**

```powershell
$env:AZURE_OPENAI_KEY = "your_api_key"
$env:AZURE_OPENAI_ENDPOINT = "your_endpoint_url"
```

Replace `"your_api_key"` and `"your_endpoint_url"` with your actual credentials. It's *highly* recommended to set these permanently in your environment configuration to avoid having to set them every time.

Once everything is set up, navigate to the directory containing `main.py` in your terminal and execute the script:

```bash
python main.py
```

This will start the entire data generation workflow.  The script will print progress updates to the console and save the generated datasets (validation, training, and destructive) as `.jsonl` files in the same directory. The `destructive_reviewer.py` and `merge_datasets.py` scripts will not be called automatically. Run `python destructive_reviewer.py destructive_data_raw.jsonl reviewed_data.jsonl` to generate approved destructive training data, then use `merge_datasets.py training_data_raw.jsonl, reviewed_data.jsonl, output_file.jsonl`. The decision of whether to merge the destructive data with the training data is *critical* and should be made with caution, ideally avoiding merging for fine-tuning purposes.
