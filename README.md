# PhyloLM

PhyloLM is a cost-effective tool to maps model by similarity in a black box setting. Similarity matrices and genetic materials can then be used to plot dendrograms, MDS or to predict benchmark scores for example.

paper : https://arxiv.org/abs/2404.04671

This repository contains the code for replicating figures in the paper and is given for transparency and replication purposes. The code of PhyloLM being very short and simple, we encourage people to try to implement it in their own code in order to fasten and optimise their pipelines. This version of the code may not be fit for production environments in terms of optimization requirements.

We made a colab demo that implements PhyloLM in a simple and versatile manner :
colab demo : https://colab.research.google.com/drive/1agNE52eUevgdJ3KL3ytv5Y9JBbfJRYqd?usp=copy

## Installation instructions
This repository uses several libraries but some are optional depending on what you want to do with PhyloLM.

To install the base of PhyloLM run
```
pip install -r requirements.txt --use-pep517
```

The unrooted trees are no longer supported by the latest biopython version and require a previous version. This can be tricky to install depending on your OS. These a NOT necessary to run the code as you can plot rooted trees instead. Here are the lines to install them

```
pip install biopython==1.69
pip install pygraphviz
```

Then download the HF benchmark scores

git clone https://huggingface.co/datasets/open-llm-leaderboard/results

If you want to use Mistral, OPENAI or Claude API you can fill the .api_[NAME] with the API tokens. To access VertexAI you'll need to follow instructions on how to setup vertexAI authentification and you should update the .api_vertexai with the project name.

## Documentation

The 5 main files of the study are given in this repository. These files reuse the dataset acquired through the study but in a compressed manner : the original code computes initially the similarity matrix from all the answers from the LLMs. Here to fit in the 100MB requirements we directly provide the similarity matrices. You can delete the data folder to recompute everything from scratch but it will require very expensive hardware and this version of the code isn't optimized for this. We didn't include the efficient LLM loading scripts as they may depend on the hardware users are using. We encourage people to adapt this code to fit their own optimized hardware. 

Here are the 5 notebooks :
- phylolm\_figures\_dendrograms.ipynb that is a notebook used to make all the figures related to dendrograms and similarity matrices
- phylolm\_figures\_benchmarks.ipynb that contains the code to make benchmark prediction
- phylolm\_figures\_hyperparameters.ipynb contains the code for the hyperparameter estimate. The data for this code are not provided here.
- gene_maker.ipynb that contains the code for making genes out of benchmarks
- colab_notebook.ipynb that is a simple notebook to upload on colab and provides a very simple interface with PhyloLM to run on the LLMs you want without needing much compute yourself. Free GPU is enough to run it quite fast.
