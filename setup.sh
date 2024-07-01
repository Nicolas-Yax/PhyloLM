# Unzip data files
unzip -q data/kl_pop/code_params.zip -d data/kl_pop
unzip -q data/kl_pop/code.zip -d data/kl_pop
unzip -q data/kl_pop/math.zip -d data/kl_pop
unzip -q data/kl_pop/math_params/genes.zip -d data/kl_pop/math_params
unzip -q data/kl_pop/math_params/probes_batch/probes_batch1.zip -d data/kl_pop/math_params/probes_batch
unzip -q data/kl_pop/math_params/probes_batch/probes_batch2.zip -d data/kl_pop/math_params/probes_batch

# Download data
mkdir gene_benchmarks

#- open-web-math
wget -O gene_benchmarks/open-web-math.jsonl.zst https://huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/main/open-web-math/test/test.jsonl.zst?download=true

#- MBXP
mkdir gene_benchmarks/mbxp
wget -O "gene_benchmarks/mbxp/0000(0).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbcpp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(1).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbcsp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(2).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbgp_release_v1.1.jsonl
wget -O "gene_benchmarks/mbxp/0000(3).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbplp_release_v1.jsonl
wget -O "gene_benchmarks/mbxp/0000(4).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbscp_release_v1.jsonl
wget -O "gene_benchmarks/mbxp/0000(5).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbswp_release_v1.jsonl
wget -O "gene_benchmarks/mbxp/0000(6).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbpp_release_v1.jsonl
wget -O "gene_benchmarks/mbxp/0000(7).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbjp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(8).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbjsp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(9).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbtsp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(10).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbkp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(11).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbrbp_release_v1.2.jsonl
wget -O "gene_benchmarks/mbxp/0000(12).jsonl" https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbphp_release_v1.2.jsonl

#Clone the leaderboard results
git clone https://huggingface.co/datasets/open-llm-leaderboard-old/results