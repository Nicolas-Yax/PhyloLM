# PhyloLM

🧬🤖 PhyloLM is a cost-effective tool to maps model by similarity in a black box setting. Similarity matrices and genetic materials can then be used to plot dendrograms, MDS or to predict many things about LLMs like benchmark scores for example. 

📖 paper : https://arxiv.org/abs/2404.04671

🔬 This repository contains the code for replicating figures in the paper and is given for transparency and replication purposes and may not be fit for production environments in terms of optimization requirements.

🌐 We encourage people that are interested in PhyloLM to use the colab demo that implements the algorithm in a simple and versatile manner.
colab demo : https://colab.research.google.com/drive/1agNE52eUevgdJ3KL3ytv5Y9JBbfJRYqd?usp=copy

## Step by step installation instructions
This repository uses several libraries but some are optional depending on what you want to plot with PhyloLM.

- Install the base of PhyloLM
```
pip install -r requirements.txt
```
- The unrooted trees are no longer supported by the latest biopython version and require a previous version. This can be tricky to install as it requires old library verions and depending on your OS and your python version this may not work correctly (ours worked with python 3.8). These are NOT necessary to run the code as you can still plot rooted trees instead. Here are the lines to install the libraries to plot unrooted trees :

```
pip install biopython==1.69 numpy==1.23.0 matplotlib==1.5.3 networkx==1.7 pygraphviz --use-pep517
```

- Prepare the respository (unzip data files + download open-web-math/mbxp + HuggingFace leaderboard results)
```
bash setup.sh
```

- If you want to use Mistral, OPENAI or Claude API you can fill the .api_[NAME] with the API tokens. To access VertexAI you'll need to follow instructions on how to setup vertexAI authentification and you should update the .api_vertexai with the project name.

## Documentation
This project is built on top of lanlab, a simple library to automate queries to LLMs. The lanlab folder contains the basic materials to make the framework run. Using this framework only 5 notebooks are used to generate all the figures in the study. These files reuse the dataset acquired through the study. You can delete the data folder to recompute everything from scratch but it will require very expensive hardware and this version of the code isn't optimized for this. We didn't include the efficient LLM loading scripts as they may depend on the hardware users are using.

📓 Here are the 5 notebooks :
- phylolm\_figures\_dendrograms.ipynb that is a notebook used to make all the figures related to dendrograms and similarity matrices (see Figures 3 and 4 in the paper)
- phylolm\_figures\_benchmarks.ipynb that contains the code to make benchmark prediction (see Figure 5 in the paper)
- phylolm\_figures\_hyperparameters.ipynb contains the code for the hyperparameter estimate. (see Figure 2 in the paper)
- gene_maker.ipynb that contains the code for making genes out of benchmarks
- colab_notebook.ipynb that is a simple notebook to upload on colab and provides a very simple interface with PhyloLM to run on the LLMs you want without needing much compute yourself. Free GPU is enough to run it quite fast. It can also be accessed using this link : https://colab.research.google.com/drive/1agNE52eUevgdJ3KL3ytv5Y9JBbfJRYqd?usp=copy

🗃️ The data folder contains 4 sets of genes :
- 'math_params' is a large set of genes extracted from open-web-math used to run the hyperparameter estimation (see phylolm\_figures\_hyperparameters.ipynb and Figure 2 in the paper).
- 'math' is extracted from open-web-math and is used to plot the dendrograms on the math genome (see phylolm\_figures\_dendrograms.ipynb and Figures 3 and 4 in the paper)
- 'code_params' is a large set of genes extracted from mbxp used to run the hyperparameter estimation (see phylolm\_figures\_hyperparameters.ipynb and Figure 2 in the paper).
- 'code' is extracted from mbxp and is used to plot the dendrograms on the code genome (see phylolm\_figures\_dendrograms.ipynb and Figures 3 and 4 in the paper)
