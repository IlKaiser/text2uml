# Text2UML
Code for the experiments of the paper Assessing the Suitability of Large Language Models in Generating UML Class Diagrams as Conceptual Models (Under Review)

![Architecture of the experiments](./images/text2uml_arch.pdf)

<object data="./images/text2uml_arch.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="./images/text2uml_arch.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="./images/text2uml_arch.pdf">Download PDF</a>.</p>
    </embed>
</object>

# Repo Structure
The repo structure is at if follows:
- notebook/
- dataset/
- environment.yml

In notebook folder you will found a notebook for each prompting technique used in the experiments, the notebook used to generate the dataset for huggingface and the dataset for the evaluation.
In dataser folder you will found the raw text data and corresponding plant uml that composes the dataset.
The environment.yml is used to setup the python virtual enviroment for the experiments.

For more info on the dataset and the evaluation you can look the online appendix at [OSF](https://osf.io/rbe7d/files/osfstorage).


# Setup
To recreate the experiments you will need:

## 1. Setup python Virtual Env
Make sure you have [Conda](https://anaconda.org/anaconda/conda) installed on your machine, and run:

```sh
conda env create -f environment.yml -p text2uml
```

To create a text2uml environment.

## 2. Setup .env
To access local variables for the api keys it is needed a .env file. You can create one with this template:

```sh
OPENAI_API_KEY=YOUR_OPEN_AI_KEY
HF_TOKEN=YOUR_HF_TOKEN
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_TREE
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
DEEPSEEK_API_KEY=YOUR_DEEPSEK_API_KEY
```

Note that none of these variable are mandatory, but remeber to add your key to text the corresponding LLM. Langchain APIs enable the tracing of the LLM generation to [LangSmith](https://www.langchain.com/langsmith). It is enabled by default.
# Run the experiments
To run the experiments type:

```sh
conda activate text2uml
jupyer notebook
```

And choose the notebook you want to execute. 

NOTE: to run some models locally you will nee hardware with adeguate performances. In the script there is native support for Apple Silicon architecture, but to enable the Cuda backed it is as easy as adding the models in the huggingface part.
