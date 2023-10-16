import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.chat_models import ChatOpenAI
from langsmith import Client

load_dotenv()

from main import get_retriever, create_test_chain

MODEL_NAME = "gpt-3.5-turbo"
client = Client()
upload_dataset = False
default_dataset = 'cmhc-full-2023-10-16'


def create_dataset():
    """
    Create a dataset using csv input
    """
    df_testing_set = pd.read_csv('CMHC Publications - tests.csv')
    df_testing_set = df_testing_set[df_testing_set.iloc[:, 0].notna()]

    today = datetime.now().strftime("%Y-%m-%d")
    dataset_name = "cmhc-full-" + today

    try:
        dataset = client.create_dataset(dataset_name=dataset_name)
    except ValueError:
        return dataset_name

    for index, row in df_testing_set.iterrows():
        client.create_example(
            inputs={
                "question": row['Question'],
                "chat_history": [{}]
            },
            outputs={
                "output": row["Correct Value (if number)"],
                "source": row["Correct reference"],
            },
            dataset_id=dataset.id,
        )

    return dataset_name


classifier_llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME)

if upload_dataset:
    dataset_name = create_dataset()
else:
    dataset_name = default_dataset

eval_config = RunEvalConfig(
    evaluators=["qa"],
    input_key="question",
    prediction_key="output",
    reference_key="output",
)

llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        streaming=True,
        temperature=0,
)

retriever = get_retriever()
answer_chain = create_test_chain(
    llm,
    retriever,
)


results = run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=answer_chain,
    evaluation=eval_config,
    verbose=True,
    concurrency_level=1,
    project_name=os.getenv('LANGCHAIN_PROJECT')+"-eval-"+datetime.now().strftime("%Y-%m-%d-%H%M%S")
)

print("-" * 50)
print("Test project name: ", results["project_name"])
print("Model output: ", results["results"])
