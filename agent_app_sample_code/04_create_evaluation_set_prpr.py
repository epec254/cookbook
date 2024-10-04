# Databricks notebook source
# MAGIC %md
# MAGIC # Managed Evaluation Datasets - Private Preview
# MAGIC
# MAGIC This notebook will get you started with our managed evaluation datasets private preview. By running this notebook, the following things will happen:
# MAGIC
# MAGIC 1. **Optional**: We will analyze your docs table and tag each document with an auto-detected topic. This will guide our synthetic generation process.
# MAGIC 1. We will create tables to store and manage your evaluation dataset.
# MAGIC 1. We will output a UI link for the developer to share with the SME to collect an evaluation dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Python libraries
# MAGIC
# MAGIC 🚫✏️ Only modify if you need additional packages in your code changes to the document parsing or chunking logic.

# COMMAND ----------

# DBTITLE 1,Installing deps
# Managed evals Private Preview
%pip install -U -qqq 'https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/managed-evals/staging/databricks_managed_evals-latest-py3-none-any.whl' -r requirements.txt --force-reinstall
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚫✏️ Setup: Load the Agent's configuration that is shared with the other notebooks
# MAGIC
# MAGIC If you used the other cookbook notebooks, the configuration for Managed Evaluation is populated automatically from those configs.  *These values can be set here if you want to use this notebook independently.*
# MAGIC
# MAGIC Requirements:
# MAGIC 1. `DOCUMENT_TABLE_NAME`: A delta table with at least 2 columns:
# MAGIC     -  `DOCUMENT_TABLE_PRIMARY_KEY`: A primary key column (usually doc URI) 
# MAGIC     -  `DOCUMENT_TABLE_TEXT_COLUMN`: A string column with the text of the whole (pre-chunked) document. If you have pdfs, this is after you process the pdfs.
# MAGIC 2. `MODEL_SERVING_ENDPOINT_NAME`: A model serving endpoint name of a deployed agent on Databricks
# MAGIC
# MAGIC Optional:
# MAGIC 1. `AGENT_NAME`: The display name of your agent that will show up in the UI
# MAGIC
# MAGIC Output:
# MAGIC 1. `EVALS_TABLE_NAME`: The UC location of the evaluation table

# COMMAND ----------

from cookbook_utils.cookbook_config import AgentCookbookConfig
from datapipeline_utils.data_pipeline_config import UnstructuredDataPipelineStorageConfig
from databricks import agents

# Load the shared configuration
cookbook_shared_config = AgentCookbookConfig.from_yaml_file('./configs/cookbook_config.yaml')

# Load the data pipeline outputs
datapipeline_output_config = UnstructuredDataPipelineStorageConfig.from_yaml_file('./configs/data_pipeline_storage_config.yaml')

# The name of the primary key in your document table (usually doc URI)
DOCUMENT_TABLE_PRIMARY_KEY = 'doc_uri'
# The name of the column in the text of your whole (pre-chunked) document.
DOCUMENT_TABLE_TEXT_COLUMN = 'doc_content'

# The name of your document table. We'll use this table to discover tags and synthesize evaluations.
DOCUMENT_TABLE_NAME = datapipeline_output_config.parsed_docs_table

# The name of your deployed agent. We will run this agent against any evaluations.
# You can find your agent endpoint name in the Machine Learning > Serving page in the Databricks UI.
MODEL_SERVING_ENDPOINT_NAME = agents.get_deployments(cookbook_shared_config.uc_model)[0].endpoint_name

# The collected evals will be stored in this UC table name.
EVALS_TABLE_NAME = cookbook_shared_config.evaluation_set_table

# Optional
AGENT_NAME = cookbook_shared_config.uc_asset_prefix

# COMMAND ----------

agents.get_deployments(cookbook_shared_config.uc_model)[0].endpoint_name


# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Setup: Import APIs

# COMMAND ----------

# DBTITLE 1,Import APIs
from managed_evals import (
    create_evals_table, # Create the evals table.
    delete_evals_table, # Delete the evals table.

    get_evals_link, # An http link to the Evals UI
    get_evals, # Retrieving the evaluation data as a dataframe.
    add_evals, # Batch adding evals to the dataset.

    detect_topics, # Detection of topics given a table of source documents.
    generate_evals, # Synthetic eval generation
    
    # Adding access to the Evals Dataset
    PermissionLevel,
    set_evals_permissions,
    
    # Setting different evaluation modes (feedback, ground_truth or grading_notes)
    EvalMode,
    set_eval_mode,
    
    # Render mode for the documents (markdown or plain)
    DocsRenderMode
)
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚫✏️ 1️⃣ Run topic discovery
# MAGIC ### This step will be automatically skipped if you have less than 10 docs
# MAGIC
# MAGIC We can analyze your docs table and automatically tag each document with a topic. These topics power the synthetic question generation, and are used as tags to organize your evals.

# COMMAND ----------

# MAGIC %md
# MAGIC Detects tags in a delta table. Outputs are written to a private table linked to the evals dataset.
# MAGIC
# MAGIC   Args:
# MAGIC    - **evals_table_name**: The name of the evals table, in the format {catalog}.{schema}.{table}.
# MAGIC    - **delta_table_name**: The name of the delta table to detect tags in.
# MAGIC    - **pkey_col_name**: The name of the primary key column in the delta table.
# MAGIC    - **content_col_name**: The name of the column containing the text data in the delta table.
# MAGIC    - **embedding_endpoint_name**: The name of the embedding endpoint to use for clustering.
# MAGIC    - **sample_size**: The number of rows to sample for topic detection.

# COMMAND ----------

# DBTITLE 1,Discovery of topics
num_docs = spark.sql(f"select count(*) as num_docs from {DOCUMENT_TABLE_NAME}").collect()[0]['num_docs']

if num_docs > 10:
    detect_topics(
        evals_table_name=EVALS_TABLE_NAME,
        delta_table_name=DOCUMENT_TABLE_NAME,
        pkey_col_name=DOCUMENT_TABLE_PRIMARY_KEY,
        content_col_name=DOCUMENT_TABLE_TEXT_COLUMN
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅✏️ 2️⃣ Optional: Import any existing evals

# COMMAND ----------

# DBTITLE 1,Importing existing evals
# To add evals, you can convert an existing MLFlow evaluation dataframe
# Or you can add them manually. The full schema is documented at # https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-set.html#evaluation-set-schema
existing_evals = [
    {
        'request': 'Here is an evaluation question',
        'response': 'This is a previously generated response',
        'retrieved_context': [
            {'doc_uri': 'test-uri', 'content': 'retrieved chunk'},
            {'doc_uri': 'test-uri2', 'content': 'asdf'}]
    },
    {
        'request': 'Another evaluation question',
        'response': 'Another previosly generated response',
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Create an evaluation dataset in `grading_notes` mode

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ✅✏️ Configure the number of evaluations to generate.
# MAGIC
# MAGIC 🚫✏️ For all other parameters, you can accept the defaults.
# MAGIC
# MAGIC Creates an evals dataset, backed by the `evals_table_name` delta table, seeded with `NUM_SYNTHETIC_EVALS` evals.
# MAGIC
# MAGIC Args:
# MAGIC - **[NEW]** **generate_num_evals**: The number of evals to synthetically generate
# MAGIC - **agent_name**: A human-friendly name for the agent that these managed evals are evaluating.
# MAGIC - **evals_table_name**: The name of the evals table in the format {catalog}.{schema}.{prefix}.
# MAGIC - **model_serving_endpoint_name**: The name of the model serving endpoint to evaluate.
# MAGIC - **docs_table_name**: The name of the table containing documents used for synthetic data generation.
# MAGIC - **primary_key_col_name**: The name of the primary key column in the docs table.
# MAGIC - **content_col_name**: The name of the content column in the docs table.
# MAGIC - **eval_mode**: The mode of evaluation. One of [feedback|grading_notes|ground_truth].
# MAGIC - **agent_name**: A human-friendly name for the agent that these evals are evaluating.
# MAGIC - **docs_render_mode**: The rendering mode for the documents: "markdown"|"plain".
# MAGIC - **existing_evals**: A list of evals to import into the managed evals table.
# MAGIC - **existing_tags**: A list of tags to import into the managed evals table.
# MAGIC - **autoassign_evals**: Set to True to automatically tag evals.

# COMMAND ----------

# DBTITLE 1,Creating an eval dataset
NUM_SYNTHETIC_EVALS = 250

create_evals_table(
    evals_table_name=EVALS_TABLE_NAME,
    model_serving_endpoint_name=MODEL_SERVING_ENDPOINT_NAME,
    
    # Documents table, used for synthetic data generation.
    docs_table_name=DOCUMENT_TABLE_NAME,
    primary_key_col_name=DOCUMENT_TABLE_PRIMARY_KEY,
    content_col_name=DOCUMENT_TABLE_TEXT_COLUMN,
    
    agent_name = AGENT_NAME,
    
    # One of "grading_notes", "ground_truth" or "feedback".
    eval_mode = "grading_notes",
    
    # Existing data.
    existing_evals=[], #existing_evals,  # Optional, use existing evals if you have them.
    existing_tags=[], # ['My Custom Tag'],  # Optional, use existing tags if you have them.

    generate_num_evals=NUM_SYNTHETIC_EVALS # Synthetic eval generation.
)

sme_ui_link = get_evals_link(evals_table_name=EVALS_TABLE_NAME)
displayHTML(
    f'<a href="{sme_ui_link}" target="_blank"><button style="color: white; background-color: #0073e6; padding: 10px 24px; cursor: pointer; border: none; border-radius: 4px;">Visit Evals UI</button></a>'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4️⃣ Run evaluation on the collected eval dataset
# MAGIC You can now use `02_agent` to run evaluation on the synthetic dataset using the `🅲 Evaluate the Agent using your evaluation set` cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5️⃣ Share the above link with the SME
# MAGIC
# MAGIC Share the UI link as well as the SME UI Guide with an SME that will help you collect an evaluation set.
# MAGIC
# MAGIC Also ask them to read the specific mode section of the guide under "Assessing AI responses":
# MAGIC - `feedback` is Feedback mode
# MAGIC - `grading_notes` is Grading notes mode
# MAGIC - `ground_truth` is Reference answer mode
# MAGIC
# MAGIC The SME must have access to your Databricks workspace in order for the UI link to work.

# COMMAND ----------

# DBTITLE 1,Enable SME access
# Enable your SME to read/write to your evaluation dataset. They must have access to your Databricks Workspace.
# 'account users' here means all users in the account.
SME_EMAILS = ['account users'] # ['somebody@company.com', 'somebody2@company.com']

set_evals_permissions(users=SME_EMAILS, evals_table_name=EVALS_TABLE_NAME, permission_level=PermissionLevel.CAN_QUERY)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 🛑 Appendix: additional utilities

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: Generate even more evals after creation

# COMMAND ----------

# DBTITLE 1,Synthesize more evals
NUM_ADDITIONAL_EVALS = 10

evals = generate_evals(evals_table_name=EVALS_TABLE_NAME, num_evals=NUM_ADDITIONAL_EVALS)
add_evals(evals=evals, evals_table_name=EVALS_TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: Switching the eval mode to `ground_truth` and collect other data
# MAGIC
# MAGIC If you decide to switch the mode, you can do this at anytime. After the new mode is set, the SME will need to refresh the UI to see the new eval mode. It's best not to switch the mode during an active labeling session.

# COMMAND ----------

set_eval_mode(evals_table_name=EVALS_TABLE_NAME, eval_mode='ground_truth')

displayHTML(
    f'<a href="{get_evals_link(evals_table_name=EVALS_TABLE_NAME)}" target="_blank"><button style="color: white; background-color: #0073e6; padding: 10px 24px; cursor: pointer; border: none; border-radius: 4px;">Visit Evals UI</button></a>'
)
